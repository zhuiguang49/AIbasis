import torch.nn as nn
from typing import List

class MLP(nn.Module):
    def __init__(self, 
                 input_size: int = 48*48,
                 hidden_sizes: List[int] = [512, 128],
                 output_size: int = 7,
                 dropout_rate: float = 0.5):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(
            nn.Flatten(),
            *layers
        )

    def forward(self, x):
        return self.model(x)