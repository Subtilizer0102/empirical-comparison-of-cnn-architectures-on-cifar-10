import torch.nn as nn
from simple_cnn import get_activation, get_pooling

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, activation='relu', batch_norm=False):
        super().__init__()
        self.activation = get_activation(activation)
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if batch_norm else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes) if batch_norm else nn.Identity(),
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResNetLike(nn.Module):
    def __init__(self, num_blocks=2, num_classes=10, activation='relu', pooling='max', dropout=0.0, batch_norm=False):
        super().__init__()
        self.in_planes = 16
        self.activation = get_activation(activation)
        self.pool = get_pooling(pooling)  # not used directly in forward (but kept for compatibility)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) if batch_norm else nn.Identity()

        self.layer1 = self._make_layer(16, num_blocks, stride=1, batch_norm=batch_norm)

        self.layer2 = self._make_layer(32, num_blocks, stride=2, batch_norm=batch_norm)

        self.layer3 = self._make_layer(64, num_blocks, stride=2, batch_norm=batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def _make_layer(self, planes, num_blocks, stride, batch_norm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride,
                                     activation=self.activation.__class__.__name__.lower(),
                                     batch_norm=batch_norm))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out