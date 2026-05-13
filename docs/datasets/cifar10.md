# CIFAR-10

## 数据集简介

- **来源**: Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
- **描述**: 10 类自然彩色图像，是计算机视觉最常用的中等规模基准数据集。图像采集自 80 million tiny images 数据集。
- **下载**: 首次使用自动从 torchvision 镜像下载（约 170MB）。

## 基本统计

| 属性 | 值 |
|------|-----|
| 类别数 | 10 |
| 训练样本 | 50,000 |
| 测试样本 | 10,000 |
| 图像尺寸 | 32×32 像素 |
| 通道数 | 3（RGB） |
| 每类样本 | 5,000 训练 + 1,000 测试 |

## 类别

| 标签 | 类别 | 标签 | 类别 |
|------|------|------|------|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

## 注册信息

```python
"cifar10": {
    "channels": 3,
    "num_classes": 10,
    "image_size": 32,
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2470, 0.2435, 0.2616),
    "description": "CIFAR-10 natural images (10 classes)",
}
```

## torchvision 加载方式

```python
from torchvision import datasets
train_set = datasets.CIFAR10(root="datasets", train=True, download=True, transform=...)
test_set = datasets.CIFAR10(root="datasets", train=False, download=True, transform=...)
```

## 预处理管线

```
原始 PIL 图像 (32×32, RGB)
  → ToTensor()                    → (3, 32, 32) float32 [0,1]
  → Resize(model_input_size)      → 缩放至 224×224 或保持 32×32
  → [RandomHorizontalFlip]        → 仅训练时
  → [RandomRotation(±10°)]        → 仅训练时
  → Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
```

## 训练建议

- **推荐模型**: NiN, GoogLeNet, VGG
- **典型准确率**: NiN ~85%, GoogLeNet ~90%, VGG16 ~88%
- **难度**: 中等——图像小（32×32）但物体多样性高
- **注意事项**: 部分类别（cat/dog, automobile/truck, bird/frog）视觉相似，是主要错误来源

[TODO: 图片: CIFAR-10 样本网格展示（4×4）]
[TODO: 图片: CIFAR-10 类别分布柱状图]
[TODO: 图片: CIFAR-10 单类放大样本展示]
