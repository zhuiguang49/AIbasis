from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    # MLP配置
    mlp_hidden_sizes: List[int] = (512, 128)
    mlp_dropout_rate: float = 0.5
    
    # CNN配置
    cnn_channels: List[int] = (32, 64)
    cnn_dropout_rate: float = 0.5
    
    # 通用配置
    num_classes: int = 7
    input_size: int = 48 * 48

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 20
    early_stopping_patience: int = 5
    lr_scheduler_patience: int = 3
    
    # 数据增强参数
    rotation_degrees: int = 10
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2

@dataclass
class PathConfig:
    data_dir: str = "data/fer2013"
    model_dir: str = "saved_models"
    log_dir: str = "logs"

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.paths = PathConfig()

config = Config()