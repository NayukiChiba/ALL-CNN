# CIFAR-100

## 数据集简介

- **来源**: Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
- **描述**: CIFAR-10 的细粒度版本。100 个类别，每个类别 500 张训练图 + 100 张测试图。类别分为 20 个超类（如"水生哺乳动物"、"交通工具"），每个超类含 5 个子类。
- **下载**: 首次使用自动从 torchvision 镜像下载。

## 基本统计

| 属性 | 值 |
|------|-----|
| 类别数 | 100（20 个超类 × 5 子类） |
| 训练样本 | 50,000（每类 500） |
| 测试样本 | 10,000（每类 100） |
| 图像尺寸 | 32×32 像素 |
| 通道数 | 3（RGB） |

## 注册信息

```python
"cifar100": {
    "channels": 3,
    "num_classes": 100,
    "image_size": 32,
    "mean": (0.5071, 0.4867, 0.4408),
    "std": (0.2675, 0.2565, 0.2761),
    "description": "CIFAR-100 natural images (100 classes)",
}
```

## torchvision 加载方式

```python
from torchvision import datasets
train_set = datasets.CIFAR100(root="datasets", train=True, download=True, transform=...)
test_set = datasets.CIFAR100(root="datasets", train=False, download=True, transform=...)
```

## 预处理管线

与 CIFAR-10 相同，但归一化参数不同。

## 训练建议

- **推荐模型**: GoogLeNet, VGG16
- **典型准确率**: GoogLeNet ~65-75%, VGG16 ~60-70%
- **难度**: 中等-困难——100 类且每类仅 500 张训练图
- **注意事项**: 
  - 样本量小（每类 500），容易过拟合，建议使用 Dropout + Weight Decay
  - 超类内的子类容易混淆（如 maple/oak/pine 树类）
  - 需要比 CIFAR-10 更大的模型容量

[TODO: 图片: CIFAR-100 样本网格展示（5×5）]
[TODO: 图片: CIFAR-100 类别分布柱状图]
[TODO: 图片: CIFAR-100 单类放大样本展示]
