import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
import json

def save_model_with_config(
    model: nn.Module,
    config: Dict[str, Any],
    save_dir: str,
    filename: str = "model.pth"
) -> None:
    """保存模型和配置"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), save_path / filename)
    
    # 保存配置
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

def load_model_with_config(
    model_class: nn.Module,
    load_dir: str,
    filename: str = "model.pth",
    device: Optional[torch.device] = None
) -> tuple:
    """加载模型和配置"""
    load_path = Path(load_dir)
    
    # 加载配置
    with open(load_path / "config.json", "r") as f:
        config = json.load(f)
    
    # 创建并加载模型
    model = model_class(**config)
    model.load_state_dict(torch.load(load_path / filename))
    
    if device:
        model = model.to(device)
    
    return model, config