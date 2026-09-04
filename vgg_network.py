import torch.nn as nn
from simple_cnn import get_activation, get_pooling  # reuse helpers

class VGGLike(nn.Module):
    def __init__(self, depth=4, num_classes=10, activation='relu', pooling='max', dropout=0.0, batch_norm=False):
        super().__init__()
        self.activation = get_activation(activation)
        pool_layer = get_pooling(pooling)

        # Extended channel list to support deeper configurations
        channels = [32, 32, 64, 64, 128, 128, 256, 256, 512, 512]
        conv_layers = []
        in_ch = 3
        pool_count = 0
        max_poolings = 5  # for 32x32 input (2^5 = 32)

        for i in range(depth):
            out_ch = channels[i] if i < len(channels) else 512
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(self.activation)
            in_ch = out_ch

            # Add pooling after every 2 convs, but stop after max_poolings
            if (i + 1) % 2 == 0 and pool_count < max_poolings:
                conv_layers.append(pool_layer)
                pool_count += 1

        self.features = nn.Sequential(*conv_layers)

        # Adaptive pooling to get a fixed-size vector regardless of final spatial size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, 512),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x