# 完整前向传播方程链

本文展示从输入图像到分类 logits 的完整前向传播过程，覆盖 ALL-CNN 项目的 5 种核心架构。

---

## 1. LeNet-5 前向传播

**源码**: [cnnlib/models/lenet.py:93-102](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/lenet.py#L93-L102)

```
输入 X₀ : (B, 1, 32, 32)
  │
  ├─ C1: Conv2d(1→6, 5×5) → Tanh   → (B, 6, 28, 28)
  ├─ S2: AvgPool2d(2×2, s=2)        → (B, 6, 14, 14)
  ├─ C3: Conv2d(6→16, 5×5) → Tanh  → (B, 16, 10, 10)
  ├─ S4: AvgPool2d(2×2, s=2)        → (B, 16, 5, 5)
  ├─ C5: Conv2d(16→120, 5×5) → Tanh → (B, 120, 1, 1)
  ├─ Flatten                        → (B, 120)
  ├─ F6: Linear(120→84) → Tanh     → (B, 84)
  ├─ Output: Linear(84→num_classes)  → (B, num_classes)
  │
输出 logits Z : (B, num_classes)
```

### 逐层方程

**C1 — 卷积 + Tanh**：

$$Z_1[c, i, j] = b_1[c] + \sum_{u=0}^{4} \sum_{v=0}^{4} W_1[c, 0, u, v] \cdot X_0[0, i+u, j+v]$$

$$A_1[c, i, j] = \tanh(Z_1[c, i, j])$$

其中 $c = 0,\dots,5$，没有 padding 所以 $i,j \in [0,27]$。输出: $(B, 6, 28, 28)$。

**S2 — 平均池化**：

$$P_1[c, i, j] = \frac{1}{4}\sum_{u=0}^{1}\sum_{v=0}^{1} A_1[c, 2i+u, 2j+v], \quad i,j \in [0,13]$$

LeNet 使用平均池化（非最大池化），对局部区域取均值。输出: $(B, 6, 14, 14)$。

**C3 — 卷积 + Tanh**：

$$Z_2[k, i, j] = b_2[k] + \sum_{c=0}^{5} \sum_{u=0}^{4} \sum_{v=0}^{4} W_2[k, c, u, v] \cdot P_1[c, i+u, j+v]$$

$$A_2[k, i, j] = \tanh(Z_2[k, i, j])$$

输出: $(B, 16, 10, 10)$。

**S4** → **C5** → **Flatten** → **F6** → **Output** 继续依序推进。最终输出 $\text{logits} \in \mathbb{R}^{B \times \text{num\_classes}}$。

---

## 2. AlexNet 前向传播

**源码**: [cnnlib/models/alexnet.py:135-146](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/alexnet.py#L135-L146)

```
输入 X₀ : (B, 3, 224, 224)
  │
  ├─ Conv1: 11×11, s=4, pad=2 → ReLU  → (B, 96, 55, 55)
  ├─ MaxPool1: 3×3, s=2               → (B, 96, 27, 27)
  ├─ Conv2: 5×5, pad=2 → ReLU        → (B, 256, 27, 27)
  ├─ MaxPool2: 3×3, s=2               → (B, 256, 13, 13)
  ├─ Conv3: 3×3, pad=1 → ReLU        → (B, 384, 13, 13)
  ├─ Conv4: 3×3, pad=1 → ReLU        → (B, 384, 13, 13)
  ├─ Conv5: 3×3, pad=1 → ReLU        → (B, 256, 13, 13)
  ├─ MaxPool5: 3×3, s=2               → (B, 256, 6, 6)
  ├─ Flatten                          → (B, 9216)
  ├─ FC1: Linear → BN1d → ReLU → Dropout(0.5) → (B, 4096)
  ├─ FC2: Linear → BN1d → ReLU → Dropout(0.5) → (B, 4096)
  ├─ Output: Linear(4096→num_classes)  → (B, num_classes)
  │
输出 logits Z : (B, num_classes)
```

### 关键层的方程

**Conv1 — 大核大步长**（与 LeNet 的小核无步长形成对比）：

$$Z_1[c, i, j] = b_1[c] + \sum_{d=0}^{2} \sum_{u=0}^{10} \sum_{v=0}^{10} W_1[c, d, u, v] \cdot X_0[d, 4i+u-2, 4j+v-2]$$

$$A_1[c, i, j] = \max(0, Z_1[c, i, j]) \quad (\text{ReLU})$$

其中 $c = 0,\dots,95$，输出 $55 \times 55$。

**FC1 — 全连接 + BN + ReLU + Dropout**:

$$H_0[j] = b_{fc1}[j] + \sum_{i=0}^{9215} F[i] \cdot W_{fc1}[j, i], \quad j = 0,\dots,4095$$

$$\hat{H}_0[j] = \gamma_{bn1}[j] \cdot \frac{H_0[j] - \mu_{bn1}[j]}{\sqrt{\sigma_{bn1}^2[j] + \epsilon}} + \beta_{bn1}[j]$$

$$A_{fc1}[j] = \max(0, \hat{H}_0[j])$$

$$D_{fc1}[j] = \begin{cases} 0 & \text{以概率 } 0.5 \\ 2 \cdot A_{fc1}[j] & \text{以概率 } 0.5 \end{cases} \quad (\text{Dropout, 仅训练时})$$

FC2 同理。训练时 Dropout 随机丢弃一半神经元并 2× 缩放存活神经元；推理时 Dropout 关闭，所有神经元正常工作。

---

## 3. VGG 前向传播

**源码**: [cnnlib/models/vgg.py:116-120](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/vgg.py#L116-L120)

以 VGG16 为例：

```
输入 X₀ : (B, 3, 224, 224)
  │
  ├─ Stage 1: [Conv3-64 → BN → ReLU] ×2 → MaxPool(2×2) → (B, 64, 112, 112)
  ├─ Stage 2: [Conv3-128 → BN → ReLU] ×2 → MaxPool(2×2) → (B, 128, 56, 56)
  ├─ Stage 3: [Conv3-256 → BN → ReLU] ×3 → MaxPool(2×2) → (B, 256, 28, 28)
  ├─ Stage 4: [Conv3-512 → BN → ReLU] ×3 → MaxPool(2×2) → (B, 512, 14, 14)
  ├─ Stage 5: [Conv3-512 → BN → ReLU] ×3 → MaxPool(2×2) → (B, 512, 7, 7)
  ├─ Flatten                                           → (B, 25088)
  ├─ FC1: Linear → BN1d → ReLU → Dropout(0.5)         → (B, 4096)
  ├─ FC2: Linear → BN1d → ReLU → Dropout(0.5)         → (B, 4096)
  ├─ Output: Linear(4096→num_classes)                   → (B, num_classes)
  │
输出 logits Z : (B, num_classes)
```

### 3×3 卷积的堆叠方程

VGG 的核心：**连续 $N$ 个 $3\times3$ 卷积 + same padding，其感受野等价于一个 $(2N+1) \times (2N+1)$ 卷积**。

单个 vgg_conv 模块的方程：

$$Z[c, i, j] = b[c] + \sum_{d=0}^{C_{\text{in}}-1} \sum_{u=0}^{2} \sum_{v=0}^{2} W[c, d, u, v] \cdot X[d, i+u-1, j+v-1]$$

$$\hat{Z}[c, i, j] = \gamma[c] \cdot \frac{Z[c, i, j] - \mu[c]}{\sqrt{\sigma^2[c] + \epsilon}} + \beta[c]$$

$$A[c, i, j] = \max(0, \hat{Z}[c, i, j])$$

padding=1 保证空间尺寸不变（$H_{\text{out}} = H_{\text{in}}$, $W_{\text{out}} = W_{\text{in}}$）。空间缩小仅由 MaxPool 负责。

### 通道翻倍策略

每个 stage 后通道数翻倍（64→128→256→512→512），空间减半。这使得：
- 计算量在各 stage 间保持大致均衡
- 空间缩小 → 每个特征图的计算量减少
- 通道翻倍 → 特征图数量增加补偿

$$
\text{FLOPs per stage} \propto C_{\text{in}} \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}}
$$
$$
\text{Stage 1}: 3\times64 \times 224^2 \quad \text{Stage 5}: 512\times512 \times 14^2
$$

---

## 4. NiN 前向传播

**源码**: [cnnlib/models/nin.py:76-82](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/nin.py#L76-L82)

```
输入 X₀ : (B, 3, 32, 32)
  │
  ├─ Stage 1: nin_block(3→192, k=5) → MaxPool(3×3, s=2)  → (B, 192, 15, 15)
  ├─ Stage 2: nin_block(192→160, k=5) → MaxPool(3×3, s=2) → (B, 160, 6, 6)
  ├─ Stage 3: nin_block(160→96, k=3) → MaxPool(3×3, s=2)  → (B, 96, 2, 2)
  ├─ Classifier: nin_block(96→num_classes, k=3)             → (B, num_classes, 2, 2)
  ├─ AdaptiveAvgPool2d(1)                                    → (B, num_classes, 1, 1)
  ├─ Flatten                                                 → (B, num_classes)
  │
输出 logits Z : (B, num_classes)
```

### mlpconv (nin_block) 方程

一个 nin_block 包含三层卷积（每层后跟 ReLU）：

**第一层 — 标准卷积 (k×k)**：

$$Z_1[c, i, j] = b_1[c] + \sum_{d} \sum_{u,v} W_1[c, d, u, v] \cdot X[d, i+u, j+v]$$

$$A_1[c, i, j] = \max(0, Z_1[c, i, j])$$

**第二层 — 1×1 卷积**：

$$Z_2[c, i, j] = b_2[c] + \sum_{d} W_2[c, d, 0, 0] \cdot A_1[d, i, j]$$

$$A_2[c, i, j] = \max(0, Z_2[c, i, j])$$

1×1 卷积等价于对每个像素位置 $(i,j)$ 做一次全连接变换——不改变空间尺寸，只做通道混叠。

**第三层 — 1×1 卷积**：

$$Z_3[c, i, j] = b_3[c] + \sum_{d} W_3[c, d, 0, 0] \cdot A_2[d, i, j]$$

$$A_3[c, i, j] = \max(0, Z_3[c, i, j])$$

三层连续的非线性变换等价于对每个像素位置用一个微型 MLP 替代单层线性卷积——因此称为 mlpconv（"多层感知机卷积"）。

### 全局平均池化 (GAP)

最后一个 nin_block 输出 `num_classes` 个通道的特征图，每个通道对应一个类别。GAP 对每个通道取空间平均：

$$Z[k] = \frac{1}{H \times W} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} A[k, i, j], \quad k = 0,\dots,\text{num\_classes}-1$$

这个 $Z$ 向量直接作为分类 logits——无需 FC 层。

---

## 5. GoogLeNet 前向传播

**源码**: [cnnlib/models/googlenet.py:135-141](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/googlenet.py#L135-L141)

```
输入 X₀ : (B, 3, 224, 224)
  │
  ├─ Stem:
  │   ├─ Conv(7×7, s=2) → BN → ReLU            → (B, 64, 112, 112)
  │   ├─ MaxPool(3×3, s=2)                       → (B, 64, 56, 56)
  │   ├─ Conv(1×1) → BN → ReLU                  → (B, 64, 56, 56)
  │   ├─ Conv(3×3) → BN → ReLU                  → (B, 192, 56, 56)
  │   └─ MaxPool(3×3, s=2)                       → (B, 192, 28, 28)
  │
  ├─ Inception 3a/3b + MaxPool                   → (B, 480, 14, 14)
  ├─ Inception 4a-4e + MaxPool                   → (B, 832, 7, 7)
  ├─ Inception 5a/5b                              → (B, 1024, 7, 7)
  │
  ├─ Head:
  │   ├─ AdaptiveAvgPool2d(1)                    → (B, 1024, 1, 1)
  │   ├─ Flatten                                  → (B, 1024)
  │   ├─ Dropout(0.4)                             → (B, 1024)
  │   └─ Linear(1024→num_classes)                 → (B, num_classes)
  │
输出 logits Z : (B, num_classes)
```

### Inception 模块四分支方程

Inception 是 GoogLeNet 的核心——同一输入经过 4 个分支并行处理，结果在通道维拼接。

设输入为 $X$，4 个分支同时计算：

**分支 1 — 1×1 卷积**：

$$B_1 = \text{ReLU}(\text{Conv}_{1\times1}^{c_1}(X))$$

**分支 2 — 1×1 降维 → 3×3 卷积**：

$$B_2 = \text{ReLU}(\text{Conv}_{3\times3}^{c_2}(\text{ReLU}(\text{Conv}_{1\times1}^{c_2^{\text{reduce}}}(X))))$$

**分支 3 — 1×1 降维 → 5×5 卷积**：

$$B_3 = \text{ReLU}(\text{Conv}_{5\times5}^{c_3}(\text{ReLU}(\text{Conv}_{1\times1}^{c_3^{\text{reduce}}}(X))))$$

**分支 4 — 3×3 MaxPool → 1×1 卷积**：

$$B_4 = \text{ReLU}(\text{Conv}_{1\times1}^{c_4}(\text{MaxPool}_{3\times3, s=1, p=1}(X)))$$

**拼接**：

$$\text{Inception}(X) = \text{Concat}(B_1, B_2, B_3, B_4) \quad \text{(沿通道维)}$$

输出通道数：$c_1 + c_2 + c_3 + c_4$

### 分类头（GAP）

与 NiN 类似，GoogLeNet 使用 GAP + 单层 Linear 替代大 FC 层：

$$Z[k] = \sum_{i=0}^{1023} W_{\text{out}}[k, i] \cdot \frac{1}{H\times W}\sum_{u=0}^{H-1}\sum_{v=0}^{W-1} A[i, u, v] + b_{\text{out}}[k]$$

---

## 五种架构前向传播对比



| 特性 | LeNet-5 | AlexNet | VGG16 | NiN | GoogLeNet |
|------|:---:|:---:|:---:|:---:|:---:|
| 激活函数 | Tanh | ReLU | ReLU | ReLU | ReLU |
| 池化方式 | AvgPool | MaxPool | MaxPool | MaxPool | MaxPool |
| BN | 无 | Conv 无, FC 有 | 每层 | 无 | 每层 |
| Dropout | 无 | FC 0.5 | FC 0.5 | 无 | Head 0.4 |
| 分类器 | 2 层 FC | 3 层 FC (含 BN+Dropout) | 3 层 FC (含 BN+Dropout) | GAP (无 FC) | GAP + 1 层 Linear |
| 前向方程复杂度 | 简单线性 | 中等 | 重复模式 | mlpconv 嵌套 | 多分支并行 |
| 并行分支 | 无 | 无 | 无 | 无 | 4 分支 Inception |

---

## 相关文档

- [卷积运算](/math/convolution) — 基础卷积公式
- [各层公式](/math/layers) — 激活函数、池化、BN 公式
- [LeNet-5](/models/lenet) — LeNet 逐层详解
- [AlexNet](/models/alexnet) — AlexNet 逐层详解
- [VGG](/models/vgg) — VGG 架构
- [NiN](/models/nin) — mlpconv 与 GAP
- [GoogLeNet](/models/googlenet) — Inception 模块
