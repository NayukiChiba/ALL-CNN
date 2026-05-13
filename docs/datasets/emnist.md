# EMNIST

## 数据集简介

- **来源**: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.
- **描述**: NIST 手写字符的扩展版，包含数字和大小写字母。"balanced" split 包含 47 个均衡类别。
- **下载**: 首次使用自动从 torchvision 镜像下载。

## 基本统计

| 属性 | 值 |
|------|-----|
| 类别数 | 47（balanced split） |
| 训练样本 | 112,800 |
| 测试样本 | 18,800 |
| 图像尺寸 | 28×28 像素 |
| 通道数 | 1（灰度） |

## 类别

47 个类别包括：10 个数字（0-9）+ 26 个大写字母（A-Z）+ 11 个小写字母（部分易混淆字母被合并，如 C/c、I/l、O/o 等）。

## 注册信息

```python
"emnist": {
    "channels": 1,
    "num_classes": 47,
    "image_size": 28,
    "mean": (0.1736,),
    "std": (0.3317,),
    "description": "EMNIST letters + digits (47 classes)",
    "train_kwargs": {"split": "balanced", "train": True},
    "test_kwargs": {"split": "balanced", "train": False},
}
```

## torchvision 加载方式

```python
from torchvision import datasets
train_set = datasets.EMNIST(root="datasets", split="balanced", train=True, download=True, transform=...)
test_set = datasets.EMNIST(root="datasets", split="balanced", train=False, download=True, transform=...)
```

**重要**: EMNIST 需要同时指定 `split="balanced"` 和 `train=True/False`。这是 torchvision EMNIST API 的特殊要求。

## 预处理管线

与 MNIST 相同。注意 EMNIST 图像的像素分布与 MNIST 不同（均值和标准差不同），使用正确的归一化参数至关重要。

## 训练建议

- **推荐模型**: LeNet-5, NiN
- **典型准确率**: 85-90%
- **难度**: 中等——47 类且存在易混淆字符对
- **注意事项**: 类别数较多（47），确保模型输出层 `num_classes=47`。某些字符类间相似度高（I/1/l, O/0, S/5, etc.）

[TODO: 图片: EMNIST 样本网格展示（5×5）]
[TODO: 图片: EMNIST 类别分布柱状图]
[TODO: 图片: EMNIST 单类放大样本展示]
