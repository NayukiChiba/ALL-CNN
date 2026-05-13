# 数据集总览

## 10 个数据集一览

| 名称 | 来源 | 类别数 | 图像尺寸 | 通道 | 训练样本 | 测试样本 | 灰度/RGB |
|------|------|--------|---------|------|---------|---------|---------|
| [MNIST](/datasets/mnist) | LeCun, 1998 | 10 | 28×28 | 1 | 60,000 | 10,000 | 灰度 |
| [FashionMNIST](/datasets/fashionmnist) | Xiao, 2017 | 10 | 28×28 | 1 | 60,000 | 10,000 | 灰度 |
| [EMNIST](/datasets/emnist) | Cohen, 2017 | 47 | 28×28 | 1 | 112,800 | 18,800 | 灰度 |
| [CIFAR-10](/datasets/cifar10) | Krizhevsky, 2009 | 10 | 32×32 | 3 | 50,000 | 10,000 | RGB |
| [CIFAR-100](/datasets/cifar100) | Krizhevsky, 2009 | 100 | 32×32 | 3 | 50,000 | 10,000 | RGB |
| [SVHN](/datasets/svhn) | Netzer, 2011 | 10 | 32×32 | 3 | 73,257 | 26,032 | RGB |
| [STL-10](/datasets/stl10) | Coates, 2011 | 10 | 96×96 | 3 | 5,000 | 8,000 | RGB |
| [Caltech-101](/datasets/caltech101) | Fei-Fei, 2004 | 101 | 可变 | 3 | ~9,000 (full) | — | RGB |
| [GTSRB](/datasets/gtsrb) | Stallkamp, 2011 | 43 | 可变 | 3 | 39,209 | 12,630 | RGB |
| [Flowers-102](/datasets/flowers102) | Nilsback, 2008 | 102 | 可变 | 3 | 6,149 (full) | — | RGB |

[TODO: 图片: 10 个数据集样本网格（每个数据集取前 16 张）]

---

## 按属性分类

### 按通道

| 类型 | 数据集 |
|------|--------|
| **灰度 (1ch)** | MNIST, FashionMNIST, EMNIST |
| **RGB (3ch)** | CIFAR-10, CIFAR-100, SVHN, STL-10, Caltech-101, GTSRB, Flowers-102 |

### 按图像尺寸

| 尺寸 | 数据集 |
|------|--------|
| **28×28** | MNIST, FashionMNIST, EMNIST |
| **32×32** | CIFAR-10, CIFAR-100, SVHN |
| **96×96** | STL-10 |
| **可变** | Caltech-101, GTSRB, Flowers-102 |

### 按类别数

| 类别数 | 数据集 |
|--------|--------|
| **10** | MNIST, FashionMNIST, CIFAR-10, SVHN, STL-10 |
| **43** | GTSRB |
| **47** | EMNIST |
| **100** | CIFAR-100 |
| **101** | Caltech-101 |
| **102** | Flowers-102 |

[TODO: 图片: 数据集类别数分布饼图]

---

## 归一化统计量

| 数据集 | 均值 (μ) | 标准差 (σ) | 来源 |
|--------|---------|-----------|------|
| MNIST | (0.1307,) | (0.3081,) | MNIST 训练集统计 |
| FashionMNIST | (0.2860,) | (0.3530,) | FashionMNIST 训练集统计 |
| EMNIST | (0.1736,) | (0.3317,) | EMNIST (balanced) 训练集统计 |
| CIFAR-10 | (0.4914, 0.4822, 0.4465) | (0.2470, 0.2435, 0.2616) | CIFAR-10 训练集各通道统计 |
| CIFAR-100 | (0.5071, 0.4867, 0.4408) | (0.2675, 0.2565, 0.2761) | CIFAR-100 训练集各通道统计 |
| SVHN | (0.4377, 0.4438, 0.4728) | (0.1980, 0.2010, 0.1970) | SVHN 训练集各通道统计 |
| STL-10 | (0.4467, 0.4398, 0.4066) | (0.2603, 0.2566, 0.2713) | STL-10 训练集各通道统计 |
| Caltech-101 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | ImageNet 统计量（通用 RGB 归一化） |
| GTSRB | (0.3403, 0.3121, 0.3214) | (0.2724, 0.2608, 0.2669) | GTSRB 训练集各通道统计 |
| Flowers-102 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | ImageNet 统计量（通用 RGB 归一化） |

[TODO: 图片: 归一化统计量对比柱状图]

---

## 特殊构造参数

部分数据集使用非标准的 torchvision 构造参数：

| 数据集 | 构造参数 | 说明 |
|--------|---------|------|
| MNIST | `train=True/False` | 标准 API |
| FashionMNIST | `train=True/False` | 标准 API |
| EMNIST | `split="balanced", train=True/False` | 需同时指定 split 和 train |
| CIFAR-10 | `train=True/False` | 标准 API |
| CIFAR-100 | `train=True/False` | 标准 API |
| SVHN | `split="train"/"test"` | 使用 split= 而非 train= |
| STL-10 | `split="train"/"test"` | 使用 split= 而非 train= |
| Caltech-101 | (无) | 无内置分集，全量加载后自行切分 |
| GTSRB | `split="train"/"test"` | 使用 split= 而非 train= |
| Flowers-102 | `split="train"/"test"` | 使用 split=，另有 split="val" |

---

## 通道/尺寸兼容矩阵

以下展示各数据集与各模型之间是否需要通道转换或尺寸缩放：

| 数据集 | LeNet (1ch, 32) | AlexNet (3ch, 224) | VGG (3ch, 224) | NiN (3ch, 32) | GoogLeNet (3ch, 224) |
|--------|:---:|:---:|:---:|:---:|:---:|
| MNIST (1ch, 28) | 无转换/Resize 32 | 通道转换/Resize 224 | 通道转换/Resize 224 | 通道转换/Resize 32 | 通道转换/Resize 224 |
| FashionMNIST (1ch, 28) | 无转换/Resize 32 | 通道转换/Resize 224 | 通道转换/Resize 224 | 通道转换/Resize 32 | 通道转换/Resize 224 |
| EMNIST (1ch, 28) | 无转换/Resize 32 | 通道转换/Resize 224 | 通道转换/Resize 224 | 通道转换/Resize 32 | 通道转换/Resize 224 |
| CIFAR-10 (3ch, 32) | ❌ | Resize 224 | Resize 224 | 无转换/无Resize | Resize 224 |
| CIFAR-100 (3ch, 32) | ❌ | Resize 224 | Resize 224 | 无转换/无Resize | Resize 224 |
| SVHN (3ch, 32) | ❌ | Resize 224 | Resize 224 | 无转换/无Resize | Resize 224 |
| STL-10 (3ch, 96) | ❌ | Resize 224 | Resize 224 | Resize 32 | Resize 224 |
| Caltech-101 (3ch, var) | ❌ | Resize 224 | Resize 224 | Resize 32 | Resize 224 |
| GTSRB (3ch, var) | ❌ | Resize 224 | Resize 224 | Resize 32 | Resize 224 |
| Flowers-102 (3ch, var) | ❌ | Resize 224 | Resize 224 | Resize 32 | Resize 224 |

- **通道转换**: 灰度 1ch → RGB 3ch（`x.repeat(3,1,1)`）
- **Resize**: 数据集尺寸 → 模型输入尺寸
- **❌**: LeNet 仅支持 1 通道输入，无法直接使用 RGB 数据集

---

## 难度估计

数据集按相对难度排序（基于典型 CNN 表现）：

| 难度 | 数据集 | 典型准确率范围 | 主要挑战 |
|------|--------|--------------|---------|
| 容易 | MNIST | 99%+ | 几乎已解决 |
| 容易 | FashionMNIST | 90-95% | 服饰纹理变异 |
| 中等-容易 | EMNIST | 85-90% | 类间相似性（I/1/l） |
| 中等-容易 | SVHN | 90-95% | 多数字共存，需裁剪 |
| 中等 | CIFAR-10 | 85-95% | 图像小、物体多样性 |
| 中等 | GTSRB | 90-95% | 光照/遮挡/运动模糊 |
| 中等-难 | CIFAR-100 | 60-75% | 类别多（100），每类仅 500 张 |
| 中等-难 | STL-10 | 65-80% | 训练集极小（5,000 张） |
| 困难 | Caltech-101 | 50-70% | 类别多（101），样本不均，可变尺寸 |
| 困难 | Flowers-102 | 60-80% | 类别多（102），类间细微差异 |

---

## 数据预处理管线

所有数据集通过统一的 [build_transform()](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/data/transform.py) 处理：

```
原始图像
  → ToTensor()                                  # PIL → Tensor [0,1]
  → [通道转换: 如需]                             # 灰度 1ch → RGB 3ch
  → Resize(model_input_size)                    # 缩放到模型期望尺寸
  → [数据增强: 仅训练]                           # RandomHorizontalFlip + RandomRotation
  → Normalize(dataset_mean, dataset_std)        # Z-score 标准化
```

详见 [数据管道](/architecture/data-pipeline)

---

## 训练/验证/测试分割

大部分数据集的原始训练集会被二次分割（90% 训练 + 10% 验证集）：

| 数据集 | 原始训练集 | 训练 (90%) | 验证 (10%) | 测试集 |
|--------|----------|-----------|-----------|--------|
| MNIST | 60,000 | 54,000 | 6,000 | 10,000 |
| FashionMNIST | 60,000 | 54,000 | 6,000 | 10,000 |
| EMNIST | 112,800 | 101,520 | 11,280 | 18,800 |
| CIFAR-10 | 50,000 | 45,000 | 5,000 | 10,000 |
| CIFAR-100 | 50,000 | 45,000 | 5,000 | 10,000 |
| SVHN | 73,257 | 65,931 | 7,326 | 26,032 |
| STL-10 | 5,000 | 4,500 | 500 | 8,000 |
| Caltech-101* | ~9,000 | ~8,100 | ~900 | (同验证集) |
| GTSRB | 39,209 | 35,288 | 3,921 | 12,630 |
| Flowers-102* | ~6,149 | (train split) | (val split) | (test split) |

> \* Caltech-101 全量加载后自行切分。Flowers-102 使用官方提供的 train/val/test split。

---

## 相关文档

- [注册系统](/architecture/registry) — 数据集注册表的设计与使用
- [数据管道](/architecture/data-pipeline) — transform 管线和数据加载器
- [各数据集详情](/datasets/mnist) — 独立页面逐一详解
