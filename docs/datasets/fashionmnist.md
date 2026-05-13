# FashionMNIST

## 数据集简介

- **来源**: Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
- **描述**: Zalando 服饰图片，作为 MNIST 的直接替代品。与 MNIST 共享相同的图像尺寸、文件格式和训练/测试分割，但内容更难分类。
- **下载**: 首次使用自动从 torchvision 镜像下载。

## 基本统计

| 属性 | 值 |
|------|-----|
| 类别数 | 10 |
| 训练样本 | 60,000 |
| 测试样本 | 10,000 |
| 图像尺寸 | 28×28 像素 |
| 通道数 | 1（灰度） |

## 类别

| 标签 | 类别 | 标签 | 类别 |
|------|------|------|------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

> Shirt (6) 和 T-shirt/top (0)、Coat (4) 之间容易混淆——这是主要挑战。

## 注册信息

```python
"fashionmnist": {
    "channels": 1,
    "num_classes": 10,
    "image_size": 28,
    "mean": (0.2860,),
    "std": (0.3530,),
    "description": "Fashion-MNIST clothing (10 classes)",
}
```

## torchvision 加载方式

```python
from torchvision import datasets
train_set = datasets.FashionMNIST(root="datasets", train=True, download=True, transform=...)
test_set = datasets.FashionMNIST(root="datasets", train=False, download=True, transform=...)
```

## 预处理管线

与 MNIST 相同。

## 训练建议

- **推荐模型**: LeNet-5, NiN
- **典型准确率**: LeNet-5 ~90-93%, NiN ~92-94%
- **难度**: 中等-容易——比 MNIST 难但仍是入门级别
- **注意事项**: 某些类别（Shirt/Coat/Pullover）类间相似度高，混淆矩阵中容易看到这两类之间的错误

[TODO: 图片: FashionMNIST 样本网格展示（4×4）]
[TODO: 图片: FashionMNIST 类别分布柱状图]
[TODO: 图片: FashionMNIST 单类放大样本展示]
