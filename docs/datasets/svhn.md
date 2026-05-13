# SVHN

## 数据集简介

- **来源**: Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., & Ng, A. Y. (2011). Reading Digits in Natural Images with Unsupervised Feature Learning.
- **描述**: Google 街景门牌号码。从街景图片中裁剪出的数字，包含各种光照、遮挡、模糊条件。
- **下载**: 首次使用自动从 torchvision 镜像下载。

## 基本统计

| 属性 | 值 |
|------|-----|
| 类别数 | 10（数字 0-9） |
| 训练样本 | 73,257 |
| 测试样本 | 26,032 |
| 额外样本 | 531,131（较弱标注，本项目不使用） |
| 图像尺寸 | 32×32 像素（从原始可变尺寸裁剪） |
| 通道数 | 3（RGB） |

## 注册信息

```python
"svhn": {
    "channels": 3,
    "num_classes": 10,
    "image_size": 32,
    "mean": (0.4377, 0.4438, 0.4728),
    "std": (0.1980, 0.2010, 0.1970),
    "description": "SVHN street view house numbers (10 classes)",
    "train_kwargs": {"split": "train"},
    "test_kwargs": {"split": "test"},
}
```

**关键**: SVHN 使用 `split=` 参数而非 `train=`！

## torchvision 加载方式

```python
from torchvision import datasets
train_set = datasets.SVHN(root="datasets", split="train", download=True, transform=...)
test_set = datasets.SVHN(root="datasets", split="test", download=True, transform=...)
```

## 预处理管线

与 CIFAR-10 相同，但归一化参数不同（SVHN 的均值和方差差异较大，街景色彩分布与自然图像不同）。

## 训练建议

- **推荐模型**: NiN, GoogLeNet
- **典型准确率**: 90-95%
- **难度**: 中等-容易——10 类但条件多样（光照、模糊、遮挡）
- **注意事项**:
  - 多个数字可能同时出现在裁剪区域中，需要网络辨别目标数字
  - 数字 "1" 和 "7"、"3" 和 "8" 容易混淆
  - 训练集较大（73K），batch size 可以设大一些

[TODO: 图片: SVHN 样本网格展示（4×4）]
[TODO: 图片: SVHN 类别分布柱状图]
[TODO: 图片: SVHN 单类放大样本展示]
