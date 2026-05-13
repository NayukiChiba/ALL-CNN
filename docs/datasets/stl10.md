# STL-10

## 数据集简介

- **来源**: Coates, A., Ng, A., & Lee, H. (2011). An Analysis of Single-Layer Networks in Unsupervised Feature Learning.
- **描述**: 受 CIFAR-10 启发但更高分辨率（96×96）的图像数据集。特点是有大量无标签数据（100,000 张），适合半监督学习研究。
- **下载**: 首次使用自动从 torchvision 镜像下载。

## 基本统计

| 属性 | 值 |
|------|-----|
| 类别数 | 10（与 CIFAR-10 相同类别） |
| 训练样本 | 5,000（每类 500） |
| 测试样本 | 8,000（每类 800） |
| 无标签样本 | 100,000 |
| 图像尺寸 | 96×96 像素 |
| 通道数 | 3（RGB） |

## 注册信息

```python
"stl10": {
    "channels": 3,
    "num_classes": 10,
    "image_size": 96,
    "mean": (0.4467, 0.4398, 0.4066),
    "std": (0.2603, 0.2566, 0.2713),
    "description": "STL-10 natural images (10 classes)",
    "train_kwargs": {"split": "train"},
    "test_kwargs": {"split": "test"},
}
```

**关键**: STL-10 使用 `split=` 参数！

## torchvision 加载方式

```python
from torchvision import datasets
train_set = datasets.STL10(root="datasets", split="train", download=True, transform=...)
test_set = datasets.STL10(root="datasets", split="test", download=True, transform=...)
```

## 预处理管线

```
原始 PIL 图像 (96×96, RGB)
  → ToTensor()                    → (3, 96, 96) float32 [0,1]
  → Resize(model_input_size)      → 缩放至 224×224 或 32×32
  → [RandomHorizontalFlip]        → 仅训练时
  → [RandomRotation(±10°)]        → 仅训练时
  → Normalize(...)
```

## 训练建议

- **推荐模型**: GoogLeNet, VGG16
- **典型准确率**: 65-80%
- **难度**: 中等-困难——训练集极小（仅 5,000 张有标签），容易过拟合
- **注意事项**:
  - 极小的训练集意味着数据增强至关重要
  - 96×96 的高分辨率需要更大感受野的网络（VGG/GoogLeNet 比 NiN 好）
  - 可考虑利用 100K 无标签数据做半监督学习（本项目未实现）

[TODO: 图片: STL-10 样本网格展示（4×4）]
[TODO: 图片: STL-10 类别分布柱状图]
[TODO: 图片: STL-10 单类放大样本展示]
