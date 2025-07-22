import os
import time
import json
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

# 设置日志记录
logging.basicConfig(level=logging.INFO)

class PairLoader(Dataset):
    """
    成对图像数据加载器类，用于加载 hazy 和 GT 图像。
    支持将图像增强、归一化和格式转换迁移到 GPU 上执行。
    """

    def __init__(self, data_dir, sub_dir, mode, size=256):
        """
        初始化数据加载器。

        参数:
            data_dir (str): 数据集根目录。
            sub_dir (str): 子目录（如 'train', 'test'）。
            mode (str): 数据模式（'train', 'valid', 'test'）。
            size (int): 输出图像尺寸，默认为 256x256。
        """
        assert mode in ['train', 'valid', 'test']  # 确保模式有效

        self.mode = mode  # 数据模式
        self.size = size  # 输出图像尺寸

        self.root_dir = os.path.join(data_dir, sub_dir)  # 数据子目录路径
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))  # 目标图像文件名列表
        self.img_num = len(self.img_names)  # 图像数量

        # 根据模式选择预处理操作

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((720, 1280), interpolation=transforms.InterpolationMode.BILINEAR),  # 先将图像缩放到 1024
                transforms.ToTensor(),  # 转换为张量
                transforms.RandomHorizontalFlip(),  # 水平翻转
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])


        if mode == 'valid':
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # 转换为 Tensor 并自动归一化到 [0, 1]
                transforms.CenterCrop(size),  # 中心裁剪
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
            ])
        if mode == 'test':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    def __len__(self):
        """返回数据集中图像的数量。"""
        return self.img_num

    def __getitem__(self, idx):
        """
        根据索引获取数据项。

        参数:
            idx (int): 数据索引。

        返回:
            dict: 包含源图像、目标图像和文件名的数据字典。
        """
        # 获取当前图像文件名
        img_name = self.img_names[idx]

        # 使用 PIL 读取图像
        source_img = Image.open(os.path.join(self.root_dir, 'hazy', img_name)).convert('RGB')
        target_img = Image.open(os.path.join(self.root_dir, 'GT', img_name)).convert('RGB')

        # 应用预处理操作
        source_img = self.transform(source_img)  # 不调用 .to(device)
        target_img = self.transform(target_img)

        # 返回处理后的数据
        return {'source': source_img, 'target': target_img, 'filename': img_name}


class SingleLoader(Dataset):
    """
    单图像数据加载器类，用于加载单张图像。
    支持将图像增强、归一化和格式转换迁移到 GPU 上执行。
    """

    def __init__(self, root_dir, size=512):
        """
        初始化数据加载器。

        参数:
            root_dir (str): 数据集根目录。
            size (int): 输出图像尺寸，默认为 256x256。
        """
        self.root_dir = root_dir  # 数据根目录
        self.img_names = sorted(os.listdir(root_dir))  # 图像文件名列表
        self.img_num = len(self.img_names)  # 图像数量
        self.size = size  # 输出图像尺寸

        # 定义图像预处理操作
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为 Tensor 并自动归一化到 [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
        ])

    def __len__(self):
        """返回数据集中图像的数量。"""
        return self.img_num

    def __getitem__(self, idx):
        """
        根据索引获取数据项。

        参数:
            idx (int): 数据索引。

        返回:
            dict: 包含图像和文件名的数据字典。
        """
        # 获取当前图像文件名
        img_name = self.img_names[idx]

        # 使用 PIL 读取图像
        img = Image.open(os.path.join(self.root_dir, img_name)).convert('RGB')

        # 应用预处理操作
        img = self.transform(img)  # 不调用 .to(device)

        # 返回处理后的数据
        return {'img': img, 'filename': img_name}