# cnnlib/registry/datasets.py

"""
数据集注册表

只做三件事：
    1. 登记：每个数据集一行字典条目
    2. 查询：列出所有数据集、获取单个数据集元信息
    3. 告诉 transform 和 loader 该怎么处理数据

用法：
    from cnnlib.registry.datasets import list_datasets, get_dataset_info

    info = get_dataset_info("cifar10")
    # info["channels"] → 3
    # info["num_classes"] → 10
"""

from typing import Any, Dict, List

# 每个条目记录一个数据集的全部元信息，加一行即接入新数据集
datasets: Dict[str, Dict[str, Any]] = {
    # -- 灰度 28x28 --
    "mnist": {
        "channels": 1,
        "num_classes": 10,
        "image_size": 28,
        "mean": (0.1307,),
        "std": (0.3081,),
        "description": "MNIST handwritten digits (10 classes)",
    },
    "fashionmnist": {
        "channels": 1,
        "num_classes": 10,
        "image_size": 28,
        "mean": (0.2860,),
        "std": (0.3530,),
        "description": "Fashion-MNIST clothing (10 classes)",
    },
    "emnist": {
        "channels": 1,
        "num_classes": 47,
        "image_size": 28,
        "mean": (0.1736,),
        "std": (0.3317,),
        "description": "EMNIST letters + digits (47 classes)",
    },
    # -- RGB 32x32 --
    "cifar10": {
        "channels": 3,
        "num_classes": 10,
        "image_size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "description": "CIFAR-10 natural images (10 classes)",
    },
    "cifar100": {
        "channels": 3,
        "num_classes": 100,
        "image_size": 32,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "description": "CIFAR-100 natural images (100 classes)",
    },
    "svhn": {
        "channels": 3,
        "num_classes": 10,
        "image_size": 32,
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
        "description": "SVHN street view house numbers (10 classes)",
    },
    # -- RGB 96x96 --
    "stl10": {
        "channels": 3,
        "num_classes": 10,
        "image_size": 96,
        "mean": (0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
        "description": "STL-10 natural images (10 classes)",
    },
    # -- RGB 尺寸不固定 --
    "caltech101": {
        "channels": 3,
        "num_classes": 101,
        "image_size": None,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "description": "Caltech-101 objects (101 classes)",
    },
    "gtsrb": {
        "channels": 3,
        "num_classes": 43,
        "image_size": None,
        "mean": (0.3403, 0.3121, 0.3214),
        "std": (0.2724, 0.2608, 0.2669),
        "description": "GTSRB traffic signs (43 classes)",
    },
    "flowers102": {
        "channels": 3,
        "num_classes": 102,
        "image_size": None,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "description": "Oxford Flowers-102 (102 classes)",
    },
}


def list_datasets() -> List[str]:
    """返回所有已注册数据集的名称列表"""
    return list(datasets.keys())


def get_dataset_info(name: str) -> Dict[str, Any]:
    """
    获取数据集元信息

    Args:
        name: 数据集名称（大小写不敏感），如 "cifar10", "mnist"

    Returns:
        {"channels": int, "num_classes": int, "image_size": int|None,
         "mean": tuple, "std": tuple, "description": str}
    """
    key = name.lower()
    if key not in datasets:
        available = ", ".join(datasets.keys())
        raise KeyError(f"Unknown dataset: '{name}'. Available: {available}")
    return datasets[key]


def print_datasets():
    """打印所有已注册数据集（调试用）"""
    print(f"{'Name':<14} {'Channels':<5} {'Classes':<8} {'Size':<10} {'Description'}")
    print("-" * 70)
    for name, info in datasets.items():
        size_str = (
            f"{info['image_size']}x{info['image_size']}"
            if info["image_size"]
            else "variable"
        )
        print(
            f"  {name:<12} {info['channels']:<5} "
            f"{info['num_classes']:<8} {size_str:<10} {info['description']}"
        )


if __name__ == "__main__":
    print_datasets()
