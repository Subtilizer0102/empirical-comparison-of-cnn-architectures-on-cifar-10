import torch.nn as nn
from pooling import StochasticPool2d

def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {name}")

def get_pooling(name, kernel_size=2, stride=2):
    if name == 'max':
        return nn.MaxPool2d(kernel_size, stride)
    elif name == 'avg':
        return nn.AvgPool2d(kernel_size, stride)
    elif name == 'stochastic':
        return StochasticPool2d(kernel_size, stride)
    else:
        raise ValueError(f"Unsupported pooling: {name}")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, activation='relu', pooling='max', dropout=0.0, batch_norm=False):
        super().__init__()
        self.activation = get_activation(activation)
        pool_layer = get_pooling(pooling)

        layers = []
        layers.append(nn.Conv2d(3, 6, kernel_size=5, padding=2))
        if batch_norm:
            layers.append(nn.BatchNorm2d(6))
        layers.append(self.activation)
        layers.append(pool_layer)  # 32 -> 16

        layers.append(nn.Conv2d(6, 16, kernel_size=5))
        if batch_norm:
            layers.append(nn.BatchNorm2d(16))
        layers.append(self.activation)
        layers.append(pool_layer)  # 12 -> 6

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x