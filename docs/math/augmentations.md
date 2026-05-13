# 数据增强

## 为什么需要数据增强

训练神经网络的本质是从有限样本中学习可泛化的模式。如果训练集只有原始样本，模型可能记住特定像素位置、朝向等无关特征——即**过拟合**。

数据增强通过对现有样本施加保留语义的变换，人工扩展训练集的有效规模，迫使模型学习**不变性**（invariance）：

- **平移不变性**: 物体在图中移动不应改变分类结果（由卷积天然提供）
- **翻转不变性**: 左右翻转马/猫仍是马/猫（多数自然图像）
- **旋转不变性**: 物体轻微旋转仍是同一物体

---

## 本项目的增强策略

**源码**: [cnnlib/data/transform.py:64-66](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/data/transform.py#L64-L66)

```python
if augment:
    ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.RandomRotation(degrees=10))
```

仅使用两种增强（简洁、有效），且**只在训练时**应用——验证和测试使用原始图像。

---

## 1. RandomHorizontalFlip

### 数学定义

设输入图像为 $X \in \mathbb{R}^{C \times H \times W}$，水平翻转操作定义为：

$$X'_{c, i, j} = X_{c, i, W-1-j}, \quad \forall c \in [0, C), i \in [0, H), j \in [0, W)$$

以概率 $p = 0.5$ 应用（PyTorch 默认）：

$$\tilde{X} = \begin{cases} X' & \text{以概率 } 0.5 \\ X & \text{以概率 } 0.5 \end{cases}$$

### 适合 vs 不适合的数据集

| 适合（自然图像） | 不适合（方向敏感） |
|:---|:---|
| CIFAR-10/100 | MNIST（数字 6↔9 会混淆） |
| STL-10 | SVHN（门牌号数字方向固定） |
| Caltech-101 | GTSRB（交通标志方向固定） |
| Flowers-102 | EMNIST（字母方向敏感） |

项目采用统一增强策略——所有数据集一律使用 RandomHorizontalFlip。对于 MNIST 等灰度数据集，水平翻转的影响有限（0.5 概率），实际训练中未观察到明显副作用。

---

## 2. RandomRotation

### 数学定义

设旋转角度 $\theta \sim \text{Uniform}(-\theta_{\max}, \theta_{\max})$，本项目 $\theta_{\max} = 10^\circ$。

旋转矩阵：

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

对图像每个像素 $(i, j)$，其在原图中的位置：

$$\begin{bmatrix} i' \\ j' \end{bmatrix} = R(\theta) \begin{bmatrix} i - H/2 \\ j - W/2 \end{bmatrix} + \begin{bmatrix} H/2 \\ W/2 \end{bmatrix}$$

然后用双线性插值从 $X$ 采样得到 $X'_{c, i, j}$。

### 为什么是 ±10°

| 旋转范围 | 效果 | 风险 |
|---------|------|------|
| ±5° | 极微弱正则化 | 效果不明显 |
| ±10° | 适度正则化（本项目选择） | 很少损坏语义 |
| ±20° | 强正则化 | 可能改变标签（如旋转 6 变 9） |
| ±30°+ | 极端增强 | 高概率改变标签 |

±10° 是一个保守且有效的选择——在不改变类别标签的前提下提供旋转不变性。

---

## 3. 训练 vs 推理管线差异

| | 训练管线 | 推理/评估管线 |
|---|:---|:---|
| ToTensor | ✓ | ✓ |
| 通道转换（1→3） | ✓ | ✓ |
| Resize | ✓ | ✓ |
| RandomHorizontalFlip | ✓ | ✗ |
| RandomRotation(±10°) | ✓ | ✗ |
| Normalize | ✓ | ✓ |

推理时不使用增强——因为我们需要模型对**固定的**输入做出一致的预测。增强仅在训练时引入随机性以提升泛化能力。

---

## 4. 增强的泛化效果

训练时每个 epoch 看到的是**不同的**图像变体：

- 同一张猫图：epoch 1 可能是原图，epoch 2 是水平翻转版，epoch 3 是旋转 +3° 版...
- 模型被迫学到"猫就是猫，无论左右还是轻微旋转"
- 等价于用更多样化的数据训练，但无需额外存储


---

## 5. 其他常见增强（本项目未使用）

| 增强方法 | 效果 | 未使用原因 |
|---------|------|-----------|
| RandomCrop | 随机裁剪后 resize | 对已 resize 的小图（32×32）意义不大 |
| ColorJitter | 随机调整亮度/对比度/饱和度 | 灰度数据集无效，且 CIFAR 等小图受益有限 |
| RandomErasing | 随机遮挡矩形区域 | 与 Dropout 原理重叠 |
| MixUp/CutMix | 混合两张图及其标签 | 训练策略复杂化，非本项目目标 |

---

## 相关文档

- [数据管道](/architecture/data-pipeline) — build_transform 完整流程
- [Dropout](/math/dropout) — 另一种正则化（随机丢弃神经元）
- [L1/L2/Weight Decay](/math/regularization) — 参数空间正则化
