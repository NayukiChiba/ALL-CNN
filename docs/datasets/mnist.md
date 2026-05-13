# MNIST

## 数据集简介

- **来源**: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition.
- **描述**: 手写数字 0-9，是机器学习和计算机视觉领域最经典、最广泛使用的基准数据集之一。
- **下载**: 首次使用自动从 torchvision 镜像下载（约 11MB）。

## 基本统计

| 属性 | 值 |
|------|-----|
| 类别数 | 10（数字 0-9） |
| 训练样本 | 60,000 |
| 测试样本 | 10,000 |
| 图像尺寸 | 28×28 像素 |
| 通道数 | 1（灰度） |
| 像素范围 | 0-255 uint8 |
| 类别均衡 | 是（每类约 6,000 训练 + 1,000 测试） |

## 注册信息

```python
"mnist": {
    "channels": 1,
    "num_classes": 10,
    "image_size": 28,
    "mean": (0.1307,),
    "std": (0.3081,),
    "description": "MNIST handwritten digits (10 classes)",
}
```

## torchvision 加载方式

```python
from torchvision import datasets
train_set = datasets.MNIST(root="datasets", train=True, download=True, transform=...)
test_set = datasets.MNIST(root="datasets", train=False, download=True, transform=...)
```

构造参数: `train=True/False`（标准 API）。

## 预处理管线

```
原始 PIL 图像 (28×28, 灰度)
  → ToTensor()                    → (1, 28, 28) float32 [0,1]
  → Resize(32) 或 Resize(224)     → 根据模型输入尺寸
  → [通道转换: 如需 3ch]          → x.repeat(3,1,1)
  → Normalize((0.1307,), (0.3081,))
```

## 训练/验证/测试分割

| 集合 | 样本数 | 用途 |
|------|--------|------|
| 训练 | 54,000 | 模型参数更新 |
| 验证 | 6,000 | 每 epoch 评估 + 早停 + 调度器 |
| 测试 | 10,000 | 最终评估 |

## 训练建议

- **推荐模型**: LeNet-5（原生 1ch 32×32 输入）
- **典型准确率**: LeNet-5 可达 99%+（20 epochs）
- **难度**: 极低——现代 CNN 几乎都能达到 99%+
- **注意事项**: 数字居中、尺寸小，避免使用过大模型（容易过拟合）

[TODO: 图片: MNIST 样本网格展示（4×4）]
[TODO: 图片: MNIST 类别分布柱状图]
[TODO: 图片: MNIST 单类放大样本展示]
