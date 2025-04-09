import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Callable

from utils.training import EarlyStopping, TrainingHistory
from config import config

def train_epoch(model: nn.Module,
               loader: DataLoader,
               criterion: nn.Module,
               optimizer: optim.Optimizer,
               device: torch.device) -> dict:
    """单个训练周期"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc='Training')
    for batch in progress_bar:
        images, labels = batch['image'].to(device), batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'loss': total_loss/len(loader), 
                                'acc': 100.*correct/total})
    
    return {
        'train_loss': total_loss/len(loader),
        'train_acc': 100.*correct/total
    }

def validate(model: nn.Module,
            loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> dict:
    """模型验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return {
        'val_loss': total_loss/len(loader),
        'val_acc': 100.*correct/total
    }

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                config: object,
                device: torch.device,
                status_callback: Optional[Callable] = None,
                stop_flag: Optional[list] = None) -> TrainingHistory:
    """模型训练主函数"""
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=config.training.lr_scheduler_patience
    )
    criterion = nn.CrossEntropyLoss()
    
    # 初始化训练工具
    history = TrainingHistory()
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        verbose=True
    )
    
    # 创建保存目录
    save_dir = Path(config.paths.model_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 训练循环
    for epoch in range(config.training.num_epochs):
        if stop_flag and stop_flag[0]:
            break
            
        # 训练和验证
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_metrics['val_acc'])
        
        # 记录历史
        metrics = {**train_metrics, **val_metrics}
        history.update(metrics, optimizer.param_groups[0]['lr'])
        
        # 更新状态
        if status_callback:
            status_callback(
                f"Epoch [{epoch+1}/{config.training.num_epochs}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} "
                f"Train Acc: {train_metrics['train_acc']:.2f}% "
                f"Val Loss: {val_metrics['val_loss']:.4f} "
                f"Val Acc: {val_metrics['val_acc']:.2f}%"
            )
        
        # 早停检查
        if early_stopping(val_metrics['val_loss'], model, 
                        str(save_dir / 'best_model.pth')):
            break
    
    # 保存训练历史
    history.save(str(save_dir / 'training_history.json'))
    
    return history