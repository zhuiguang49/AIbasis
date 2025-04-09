from PyQt5.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json
from pathlib import Path
import numpy as np

class TrainingPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # 创建图表
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 初始化子图
        self.ax1 = self.figure.add_subplot(121)  # 损失图
        self.ax2 = self.figure.add_subplot(122)  # 准确率图
        
        self.figure.tight_layout()

    def update_plot(self, history):
        """更新训练历史图表"""
        self.ax1.clear()
        self.ax2.clear()
        
        # 绘制损失曲线
        self.ax1.plot(history['train_loss'], label='Train Loss')
        self.ax1.plot(history['val_loss'], label='Val Loss')
        self.ax1.set_title('Loss History')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        
        # 绘制准确率曲线
        self.ax2.plot(history['train_acc'], label='Train Acc')
        self.ax2.plot(history['val_acc'], label='Val Acc')
        self.ax2.set_title('Accuracy History')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()