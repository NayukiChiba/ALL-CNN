# Caltech-101

## 数据集简介

- **来源**: Fei-Fei, L., Fergus, R., & Perona, P. (2004). Learning Generative Visual Models from Few Training Examples: An Incremental Bayesian Approach Tested on 101 Object Categories.
- **描述**: 101 个物体类别 + 1 个背景类别的图像数据集，是物体识别领域的重要基准。
- **下载**: 首次使用自动从 torchvision 镜像下载。

## 基本统计

| 属性 | 值 |
|------|-----|
| 类别数 | 101（+ 1 背景类别，共 102 但 torchvision 仅返回 101 类） |
| 总样本数 | ~9,000（各类别 40-800 张不等） |
| 图像尺寸 | 可变（~200×300 典型值） |
| 通道数 | 3（RGB） |
| 类别均衡 | 否（各类样本数差异大） |

## 注册信息

```python
"caltech101": {
    "channels": 3,
    "num_classes": 101,
    "image_size": None,     # 可变尺寸
    "mean": (0.485, 0.456, 0.406),    # ImageNet 统计量
    "std": (0.229, 0.224, 0.225),
    "description": "Caltech-101 objects (101 classes)",
    "train_kwargs": {},      # 无内置 train/test 分集
    "test_kwargs": {},
}
```

## torchvision 加载方式

```python
from torchvision import datasets
# Caltech101 没有内置 train/test 分集，全量加载
full_set = datasets.Caltech101(root="datasets", download=True, transform=...)
```

`train_kwargs={}` 表示无额外构造参数。加载后由 `random_split` 按 `val_split` 比例切分训练/验证/测试集。

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

使用 ImageNet 统计量而非 Caltech-101 自身统计量——这是通用做法，因为 Caltech-101 样本量小且尺寸不统一，自身统计量不够可靠。同时，使用 ImageNet 统计量也便于与预训练模型兼容。

## 训练建议

- **推荐模型**: GoogLeNet, VGG16
- **典型准确率**: 50-70%
- **难度**: 困难——101 类，总样本少（~9K），类别不均衡
- **注意事项**:
  - 类别不均衡（某些类仅 40 张），需注意 per-class 指标
  - 可变尺寸需统一 Resize 到模型输入尺寸
  - 建议使用较大的 Dropout 和 Weight Decay 防止过拟合
  - 图像背景较干净，与真实场景有差距（数据集偏置）

[TODO: 图片: Caltech-101 样本网格展示（5×5）]
[TODO: 图片: Caltech-101 类别分布柱状图]
[TODO: 图片: Caltech-101 单类放大样本展示]
