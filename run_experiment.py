import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from data_loader import get_cifar10_loaders
from simple_cnn import SimpleCNN
from vgg_network import VGGLike
from resnet_network import ResNetLike
from train_test_eval import train, validate, test

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Hyperparameter Experiments')
    # Model architecture
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'vgg', 'resnet'],
                        help='Model architecture')
    parser.add_argument('--depth', type=int, default=2, help='Depth (conv layers for vgg, blocks for resnet)')
    # Activation, pooling
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu', 'sigmoid', 'tanh'])
    parser.add_argument('--pooling', type=str, default='max', choices=['max', 'avg', 'stochastic'])
    # Optimization
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for SGD)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step_size', type=int, default=20, help='LR scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR scheduler gamma')
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--batch_norm', action='store_true', help='Use batch normalization')

    parser.add_argument('--experiment_name', type=str, default='exp', help='Name for saving logs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train_fraction', type=float, default=1.0,
                        help='Fraction of training data to use (e.g., 0.5 for half)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/{args.experiment_name}.json'
    model_dir = f'logs/{args.experiment_name}_model.pth'

    device = torch.device(args.device)
    print(f'Using device: {device}')

    # Data
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, val_split=0.1, train_fraction=args.train_fraction, num_workers=2
    )

    # Build model
    if args.model == 'simple':
        model = SimpleCNN(
            num_classes=10,
            activation=args.activation,
            pooling=args.pooling,
            dropout=args.dropout,
            batch_norm=args.batch_norm
        )
    elif args.model == 'vgg':
        model = VGGLike(
            depth=args.depth,
            num_classes=10,
            activation=args.activation,
            pooling=args.pooling,
            dropout=args.dropout,
            batch_norm=args.batch_norm
        )
    elif args.model == 'resnet':
        model = ResNetLike(
            num_blocks=args.depth,
            num_classes=10,
            activation=args.activation,
            pooling=args.pooling,
            dropout=args.dropout,
            batch_norm=args.batch_norm
        )
    else:
        raise ValueError('Unknown model')

    model = model.to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch, args.log_interval)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_dir)
            print(f'*** New best model saved (val_acc={val_acc:.2f}) ***')

    # Load best model and test
    model.load_state_dict(torch.load(model_dir))
    test_loss, test_acc = test(model, device, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # Save results
    results = {
        'args': vars(args),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'history': history
    }
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {log_file}')

if __name__ == '__main__':
    main()