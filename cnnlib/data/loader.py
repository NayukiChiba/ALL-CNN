"""
DataLoader 工厂

根据模型名和数据集名，自动加载数据并构建 train/val/test 三个 DataLoader。

职责：
    1. 查注册表拿到数据集元信息（供 torchvision 类选择）
    2. 调 transform.build_transform 自动拼 transform
    3. 切分 train/val，构建 DataLoader

用法：
    from cnnlib.data.loader import build_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(
        model_name="vgg16",
        dataset_name="cifar10",
        batch_size=64,
        data_dir="datasets/",
    )
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from cnnlib.data.transform import build_transform

# 数据集名称 → torchvision 类
_TORCHVISION_CLASS = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
    "emnist": datasets.EMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "svhn": datasets.SVHN,
    "stl10": datasets.STL10,
    "caltech101": datasets.Caltech101,
    "gtsrb": datasets.GTSRB,
    "flowers102": datasets.Flowers102,
}


def build_dataloaders(
    model_name: str,
    dataset_name: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    data_dir: str = "datasets/",
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 train / val / test 三个 DataLoader

    流程：
        1. 拿到 torchvision 数据集类
        2. 分别构建 train 和 test 的 transform（train 含增强）
        3. 加载数据集
        4. 从 train 中切出 val
        5. 包成 DataLoader

    Args:
        model_name:   模型名称，如 "vgg16"
        dataset_name: 数据集名称，如 "cifar10"
        batch_size:   批次大小
        val_split:    验证集比例（从 train 中切）
        num_workers:  DataLoader 子进程数
        pin_memory:   是否 pin 内存（GPU 训练建议开启）
        data_dir:     数据存储根目录
        seed:         随机种子（控制 train/val 切分一致性）

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _TORCHVISION_CLASS:
        available = ", ".join(_TORCHVISION_CLASS.keys())
        raise KeyError(f"Unknown dataset: '{dataset_name}'. Available: {available}")

    dataset_cls = _TORCHVISION_CLASS[dataset_name]
    data_root = Path(data_dir)

    # 构建 transform
    train_transform = build_transform(model_name, dataset_name, augment=True)
    eval_transform = build_transform(model_name, dataset_name, augment=False)

    # 加载原始数据集
    train_full = dataset_cls(
        root=str(data_root), train=True, download=True, transform=train_transform
    )
    test_dataset = dataset_cls(
        root=str(data_root), train=False, download=True, transform=eval_transform
    )

    # 切分 train / val
    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_full, [train_size, val_size], generator=generator
    )

    # val 不能用增强 transform，覆盖
    val_dataset.dataset.transform = eval_transform

    # 构建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders(
        model_name="lenet",
        dataset_name="fashionmnist",
        batch_size=64,
    )
    print(f"train: {len(train_loader.dataset):,} samples, {len(train_loader)} batches")
    print(f"val:   {len(val_loader.dataset):,} samples, {len(val_loader)} batches")
    print(f"test:  {len(test_loader.dataset):,} samples, {len(test_loader)} batches")

    images, labels = next(iter(train_loader))
    print(f"batch shape: {images.shape}")
