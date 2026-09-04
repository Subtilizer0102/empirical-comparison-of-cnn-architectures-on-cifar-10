import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Subset

def get_cifar10_loaders(batch_size=128, val_split=0.1, train_fraction=1.0, num_workers=2):
    """Return train_loader, val_loader, test_loader for CIFAR-10.
       train_fraction: fraction of full training set to use (e.g., 0.5 for half).
    """
    # Mean and std for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    # Reduce training set if train_fraction < 1.0
    if train_fraction < 1.0:
        num_total = len(full_trainset)
        num_keep = int(num_total * train_fraction)
        # Randomly select indices (shuffle first)
        indices = torch.randperm(num_total)[:num_keep].tolist()
        full_trainset = Subset(full_trainset, indices)

    # Split into train and validation
    train_size = int((1 - val_split) * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader