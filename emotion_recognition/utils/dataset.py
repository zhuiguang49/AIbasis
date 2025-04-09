import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image


class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None, usage_filter=None):
        """
        初始化数据集
        :param csv_file: 数据集的路径 (FER2013 CSV 文件)
        :param transform: 数据增强的变换
        :param usage_filter: 数据用途 (Training, PublicTest, PrivateTest)
        """
        self.data = pd.read_csv(csv_file)

        # 过滤数据用途
        if usage_filter:
            self.data = self.data[self.data["Usage"] == usage_filter]

        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        :param idx: 数据索引
        :return: 一个包含图像和标签的字典
        """
        # 提取图像像素和表情标签
        pixels = self.data.iloc[idx, 1]  # 图像像素
        emotion = int(self.data.iloc[idx, 0])  # 表情标签

        # 将像素字符串转为 numpy 数组并调整形状
        pixels = np.fromstring(pixels, dtype=int, sep=" ").reshape(48, 48).astype(np.uint8)

        # 将 numpy 数组转为 PIL 图像对象
        image = Image.fromarray(pixels)

        # 应用数据增强
        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": emotion}


def get_transforms():
    """
    定义图像预处理和数据增强
    :return: torchvision.transforms.Compose 对象
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# 测试代码
if __name__ == "__main__":
    dataset = FER2013Dataset(
        csv_file="data/fer2013/fer2013.csv",
        transform=get_transforms(),
        usage_filter="Training"
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample image shape: {sample['image'].shape}, Label: {sample['label']}")
