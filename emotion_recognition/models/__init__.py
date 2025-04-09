from .mlp import MLP
from .simple_cnn import SimpleCNN

def create_model(model_name: str, **kwargs):
    """
    工厂函数：创建模型实例
    """
    models = {
        'MLP': MLP,
        'SimpleCNN': SimpleCNN
    }
    
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")
        
    return models[model_name](**kwargs)

__all__ = ['MLP', 'SimpleCNN', 'create_model']