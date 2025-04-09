import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model: torch.nn.Module,
                   loader: DataLoader,
                   device: torch.device) -> Dict:
    """完整的模型评估"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        'labels': np.array(all_labels),
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs)
    }

def generate_metrics(results: Dict) -> Dict:
    """生成评估指标"""
    labels = results['labels']
    preds = results['predictions']
    probs = results['probabilities']
    
    # 定义类别名称
    class_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    # 计算每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # 为每个类别计算二分类的ROC
    for i in range(len(class_names)):
        # 将当前类别转换为二分类问题
        y_true = (labels == i).astype(int)  # 1表示当前类别，0表示其他类别
        y_score = probs[:, i]  # 当前类别的预测概率
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return {
        'classification_report': classification_report(
            labels, preds,
            target_names=class_names,
            zero_division=0
        ),
        'confusion_matrix': confusion_matrix(labels, preds),
        'roc_data': {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        },
        'class_names': class_names
    }

def plot_metrics(results: Dict, save_dir: Path) -> None:
    """绘制评估指标可视化图"""
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=results['class_names'],
        yticklabels=results['class_names']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_dir / 'confusion_matrix.png')
    plt.close()

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']
    
    for i, (name, color) in enumerate(zip(results['class_names'], colors)):
        plt.plot(
            results['roc_data']['fpr'][i],
            results['roc_data']['tpr'][i],
            color=color,
            label=f'{name} (AUC = {results["roc_data"]["auc"][i]:.2f})'
        )

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_dir / 'roc_curves.png')
    plt.close()