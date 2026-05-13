# 数据集对比

## 综合对比表

| 数据集 | 类别数 | 图像尺寸 | 通道 | 训练集 | 测试集 | 来源 |
|------|:---:|:---:|:---:|------:|------:|------|
| MNIST | 10 | 28×28 | 1 | 60,000 | 10,000 | 手写数字 |
| FashionMNIST | 10 | 28×28 | 1 | 60,000 | 10,000 | 服饰类别 |
| EMNIST | 47 | 28×28 | 1 | 112,800 | 18,800 | 手写字母+数字 |
| CIFAR-10 | 10 | 32×32 | 3 | 50,000 | 10,000 | 自然物体 |
| CIFAR-100 | 100 | 32×32 | 3 | 50,000 | 10,000 | 自然物体细分类 |
| SVHN | 10 | 32×32 | 3 | 73,257 | 26,032 | 街景门牌号 |
| STL-10 | 10 | 96×96 | 3 | 5,000 | 8,000 | 自然物体（高分辨率） |
| Caltech-101 | 101 | 可变 | 3 | ~3,000 | ~6,000 | 自然物体 |
| GTSRB | 43 | 可变 | 3 | 39,209 | 12,630 | 交通标志 |
| Flowers-102 | 102 | 可变 | 3 | 2,040 | 6,149 | 花卉细分类 |

---

## 按属性分类

### 按通道

| 灰度（1 通道） | RGB（3 通道） |
|:---|:---|
| MNIST | CIFAR-10 / CIFAR-100 |
| FashionMNIST | SVHN / STL-10 |
| EMNIST | Caltech-101 / GTSRB / Flowers-102 |

灰度→RGB 模型时，`build_transform()` 自动将单通道复制为 3 通道。

### 按图像尺寸

| 尺寸 | 数据集 |
|:---:|------|
| 28×28 | MNIST, FashionMNIST, EMNIST |
| 32×32 | CIFAR-10, CIFAR-100, SVHN |
| 96×96 | STL-10 |
| 可变（需 Resize） | Caltech-101, GTSRB, Flowers-102 |

所有数据集通过 `transforms.Resize()` 统一到模型要求的输入尺寸。

### 按类别数

| 类别数 | 数据集 | 任务难度 |
|:---:|------|:---:|
| 10 | MNIST, FashionMNIST, CIFAR-10, SVHN, STL-10 | 标准 |
| 43 | GTSRB | 中等 |
| 47 | EMNIST | 中等 |
| 100 | CIFAR-100 | 较难 |
| 101 | Caltech-101 | 较难 |
| 102 | Flowers-102 | 最难（细粒度） |

---

## 难度排序

基于典型 CNN 能达到的验证准确率：

| 难度 | 数据集 | 典型 val_acc | 说明 |
|:---:|------|:---:|------|
| ★☆☆☆☆ | MNIST | 99%+ | 最简单——手写数字区别大 |
| ★★☆☆☆ | FashionMNIST | 93-95% | 比 MNIST 难——服饰比数字复杂 |
| ★★☆☆☆ | SVHN | 95-97% | 数字清晰但背景复杂 |
| ★★★☆☆ | CIFAR-10 | 90-95% | 自然物体、背景多样 |
| ★★★☆☆ | GTSRB | 95-98% | 类别虽多但模式固定 |
| ★★★★☆ | EMNIST | 85-88% | 47 类、部分字母相似 |
| ★★★★☆ | CIFAR-100 | 70-80% | 100 类、每类仅 500 训练样本 |
| ★★★★☆ | STL-10 | 70-75% | 训练样本极少（500/类） |
| ★★★★★ | Caltech-101 | 60-80% | 样本少、类别多、分辨率不一 |
| ★★★★★ | Flowers-102 | 60-80% | 细粒度——区分 102 种花卉极难 |

---

## 模型-数据集兼容性矩阵

| 模型 / 数据集 | MNIST | Fashion | EMNIST | CIFAR10 | CIFAR100 | SVHN | STL10 | Caltech | GTSRB | Flowers |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LeNet-5 (1ch) | ✓ | ✓ | ✓ | — | — | — | — | — | — | — |
| NiN (3ch) | △ | △ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AlexNet (3ch) | △ | △ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| VGG11-19 (3ch) | △ | △ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GoogLeNet (3ch) | △ | △ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

- ✓ = 原生兼容（通道和尺寸匹配）
- △ = 自动适配（通道转换 + Resize，性能可能略降）
- — = 不兼容（模型只支持 1 通道，RGB 数据无法直接使用）

---

## 数据集特殊构造参数

部分数据集的 torchvision 接口与标准 `train=True/False` 不同：

| 数据集 | 训练集参数 | 测试集参数 |
|------|------|------|
| MNIST / FashionMNIST / CIFAR10/100 | `train=True` | `train=False` |
| EMNIST | `split="balanced", train=True` | `split="balanced", train=False` |
| SVHN | `split="train"` | `split="test"` |
| STL-10 | `split="train"` | `split="test"` |
| Caltech-101 | `download=True` | `download=True` |
| GTSRB | `split="train"` | `split="test"` |
| Flowers-102 | `download=True` | `download=True` |

这些特殊参数在 `cnnlib/registry/datasets.py` 注册表中统一管理，由 `build_dataloaders()` 自动查表使用。

---

## 归一化统计量

| 数据集 | mean | std |
|------|------|------|
| MNIST | (0.1307,) | (0.3081,) |
| FashionMNIST | (0.2860,) | (0.3530,) |
| EMNIST | (0.1751,) | (0.3332,) |
| CIFAR-10 | (0.4914, 0.4822, 0.4465) | (0.2470, 0.2435, 0.2616) |
| CIFAR-100 | (0.5071, 0.4866, 0.4409) | (0.2673, 0.2564, 0.2761) |
| SVHN | (0.4377, 0.4438, 0.4728) | (0.1980, 0.2010, 0.1970) |
| STL-10 | (0.4850, 0.4560, 0.4060) | (0.2290, 0.2240, 0.2250) |
| Caltech-101 | (0.4850, 0.4560, 0.4060) | (0.2290, 0.2240, 0.2250) |
| GTSRB | (0.4850, 0.4560, 0.4060) | (0.2290, 0.2240, 0.2250) |
| Flowers-102 | (0.4850, 0.4560, 0.4060) | (0.2290, 0.2240, 0.2250) |

注：Caltech-101/GTSRB/Flowers-102 使用 ImageNet 统计量（近似），因 torchvision 未提供各数据集专属的官方统计值。

---

## 训练建议

| 数据集 | Epochs | 推荐模型 | 注意事项 |
|------|:---:|------|------|
| MNIST | 10-20 | LeNet-5 | 几乎任何模型都能 99%+ |
| FashionMNIST | 20-30 | LeNet-5 / NiN | 使用 BN 加速收敛 |
| EMNIST | 30-50 | NiN | 47 类需扩大 num_classes |
| CIFAR-10 | 50-100 | VGG16 / GoogLeNet | 数据增强关键 |
| CIFAR-100 | 50-100 | VGG16 / GoogLeNet | 少量样本/类，需强正则化 |
| SVHN | 20-30 | GoogLeNet | 裁剪到数字区域效果好 |
| STL-10 | 50-100 | GoogLeNet | 训练样本极少，发挥 GAP 优势 |
| Caltech-101 | 50-100 | GoogLeNet | 可变尺寸需 resize |
| GTSRB | 30-50 | GoogLeNet | 类别多但模式固定 |
| Flowers-102 | 100+ | VGG16 / GoogLeNet | 细粒度最挑战 |

---

## 相关文档

- [数据集总览](/datasets/overview) — 10 数据集注册信息
- [MNIST](/datasets/mnist) ~ [Flowers-102](/datasets/flowers102) — 各数据集详情
- [模型对比](/guides/model-comparison) — 8 模型对比与选择
- [数据管道](/architecture/data-pipeline) — transform 与 DataLoader
