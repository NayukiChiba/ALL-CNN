# GTSRB

## 数据集简介

- **来源**: Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2011). The German Traffic Sign Recognition Benchmark: A multi-class classification competition.
- **描述**: 德国交通标志识别基准。从真实驾驶场景中采集的交通标志图像，包含严重的光照变化、运动模糊、遮挡、低对比度等挑战。
- **下载**: 首次使用自动从 torchvision 镜像下载。

## 基本统计

| 属性 | 值 |
|------|-----|
| 类别数 | 43（各类交通标志） |
| 训练样本 | 39,209 |
| 测试样本 | 12,630 |
| 图像尺寸 | 可变（15×15 ~ 250×250） |
| 通道数 | 3（RGB） |
| 类别均衡 | 否（各类 200~2,250 张不等） |

## 注册信息

```python
"gtsrb": {
    "channels": 3,
    "num_classes": 43,
    "image_size": None,     # 可变尺寸
    "mean": (0.3403, 0.3121, 0.3214),
    "std": (0.2724, 0.2608, 0.2669),
    "description": "GTSRB traffic signs (43 classes)",
    "train_kwargs": {"split": "train"},
    "test_kwargs": {"split": "test"},
}
```

## torchvision 加载方式

```python
from torchvision import datasets
train_set = datasets.GTSRB(root="datasets", split="train", download=True, transform=...)
test_set = datasets.GTSRB(root="datasets", split="test", download=True, transform=...)
```

## 预处理管线

```
原始 PIL 图像 (可变尺寸, RGB)
  → ToTensor()                    → (3, H, W) float32 [0,1]
  → Resize(model_input_size)      → 统一缩放
  → [RandomHorizontalFlip]        → 仅训练时（注意：部分交通标志翻转后含义变化！）
  → [RandomRotation(±10°)]        → 仅训练时
  → Normalize(...)
```

## 归一化

GTSRB 使用自身训练集的各通道统计量。由于图像采集自真实道路场景（自然光照），均值和标准差与自然图像数据集不同——蓝色通道均值较高（天空和路牌背景常常带蓝色调）。

## 训练建议

- **推荐模型**: GoogLeNet, VGG16
- **典型准确率**: 90-95%
- **难度**: 中等——类别多（43）、尺寸不一、但交通标志设计规范统一
- **注意事项**:
  - 类别不均衡（最少 200 张，最多 2,250 张），注意 per-class 指标
  - **水平翻转需谨慎**: 某些交通标志（如左转 ↔ 右转）翻转后含义完全相反
  - 尺寸差异极大（15×15 到 250×250），Resize 可能导致小图模糊
  - 速度限制标志（多个类别）彼此相似，是混淆的主要来源

[TODO: 图片: GTSRB 样本网格展示（5×5）]
[TODO: 图片: GTSRB 类别分布柱状图]
[TODO: 图片: GTSRB 单类放大样本展示]
