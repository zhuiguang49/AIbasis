import torch.nn as nn
from typing import List

class SimpleCNN(nn.Module):
    def __init__(self, 
                 in_channels: int = 1,
                 channels: List[int] = [32, 64],
                 num_classes: int = 7,
                 dropout_rate: float = 0.5):
        super(SimpleCNN, self).__init__()
        
        layers = []
        prev_channels = in_channels
        
        # 构建卷积层
        for channel in channels:
            layers.extend([
                nn.Conv2d(prev_channels, channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout_rate)
            ])
            prev_channels = channel
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算展平后的特征维度
        self.flat_features = prev_channels * (48 // (2 ** len(channels))) ** 2
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x