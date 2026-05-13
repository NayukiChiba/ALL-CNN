# Flowers-102

## 数据集简介

- **来源**: Nilsback, M-E. & Zisserman, A. (2008). Automated Flower Classification over a Large Number of Classes.
- **描述**: 牛津大学采集的 102 种花卉图像数据集。类别多、类间差异细微，是细粒度分类的经典基准。
- **下载**: 首次使用自动从 torchvision 镜像下载。

## 基本统计

| 属性 | 值 |
|------|-----|
| 类别数 | 102 |
| 总样本数 | 6,149（官方分 train/val/test） |
| 图像尺寸 | 可变（~500×500 典型值） |
| 通道数 | 3（RGB） |
| 每类样本 | 40-258 张（不均衡） |

## 官方分割

Flowers-102 提供官方三路分割：
- `split="train"`: 训练集
- `split="val"`: 验证集
- `split="test"`: 测试集

**本项目使用**: `split="train"` 作为训练源（进一步切出验证集），`split="test"` 作为测试集。

## 注册信息

```python
"flowers102": {
    "channels": 3,
    "num_classes": 102,
    "image_size": None,     # 可变尺寸
    "mean": (0.485, 0.456, 0.406),    # ImageNet 统计量
    "std": (0.229, 0.224, 0.225),
    "description": "Oxford Flowers-102 (102 classes)",
    "train_kwargs": {"split": "train"},
    "test_kwargs": {"split": "test"},
}
```

## torchvision 加载方式

```python
from torchvision import datasets
train_set = datasets.Flowers102(root="datasets", split="train", download=True, transform=...)
test_set = datasets.Flowers102(root="datasets", split="test", download=True, transform=...)
```

## 预处理管线

```
原始 PIL 图像 (可变尺寸, RGB)
  → ToTensor()                    → (3, H, W) float32 [0,1]
  → Resize(model_input_size)      → 统一缩放
  → [RandomHorizontalFlip]        → 仅训练时
  → [RandomRotation(±10°)]        → 仅训练时
  → Normalize(ImageNet统计量)
```

## 归一化

使用 ImageNet 统计量。Flowers-102 同样样本量不大且图像尺寸不一，通用 RGB 归一化往往比自身统计量更稳定。

## 训练建议

- **推荐模型**: GoogLeNet, VGG16
- **典型准确率**: 60-80%
- **难度**: 困难——102 类、总样本少（~6K）、细粒度差异
- **注意事项**:
  - 细粒度分类：某些花卉种类极为相似（不同品种的玫瑰/雏菊），需要捕获花蕊、花瓣形状等细微特征
  - 背景干扰：花朵在自然场景中拍摄，背景千差万别
  - 建议使用高分辨率模型（224×224）以保留细节特征
  - 数据增强不要过于激进——强旋转可能破坏花朵的判别特征

[TODO: 图片: Flowers-102 样本网格展示（5×5）]
[TODO: 图片: Flowers-102 类别分布柱状图]
[TODO: 图片: Flowers-102 单类放大样本展示]
