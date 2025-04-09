import torch
from typing import Dict, Optional, List
import logging
from pathlib import Path
import json

class EarlyStopping:
    """早停机制实现"""
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, path: str):
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class TrainingHistory:
    """训练历史记录"""
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

    def update(self, metrics: Dict[str, float], lr: float):
        for k, v in metrics.items():
            self.history[k].append(v)
        self.history['learning_rates'].append(lr)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f)

    def load(self, path: str):
        with open(path, 'r') as f:
            self.history = json.load(f)