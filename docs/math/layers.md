# 各层公式与设计原理

本文逐一分析项目中每一层的数学公式、设计目的，以及为什么选择这种设计。

---

## Conv2d：二维卷积

### 数学公式

输入 $X \in \mathbb{R}^{C_{in} \times H \times W}$，输出 $Y \in \mathbb{R}^{C_{out} \times H' \times W'}$：

$$Y[c_{out}, i, j] = b[c_{out}] + \sum_{c_{in}=0}^{C_{in}-1} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} W[c_{out}, c_{in}, u, v] \cdot X[c_{in}, i \cdot s + u - p, j \cdot s + v - p]$$

其中 $s$ 为步长（stride），$p$ 为填充量（padding），输出尺寸：

$$H' = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1$$

### Same Padding

本项目使用 `padding = kernel_size // 2`（[src/model/layers.py:91-94](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/model/layers.py#L91-L94)），即 $p = \lfloor k/2 \rfloor$。

当 $s=1$ 时，输出尺寸恰好等于输入尺寸：

$$H' = \left\lfloor \frac{H + 2 \times 1 - 3}{1} \right\rfloor + 1 = H$$

**为什么用 Same Padding？**

- 保留空间分辨率，让卷积只改变通道数，空间尺寸交给 MaxPool 主动控制
- 避免每层都"吃掉"2 像素的边缘，深层输出不会缩到太小
- 对于 MNIST 28×28 的小图尤其重要——没有 padding 的话两层 3×3 卷积就会从 28→26→24，损失大量边缘信息

### 为什么用奇数核（3×3）？

- 奇数核有明确的中心点，方便对齐和设计 padding（$\lfloor 3/2 \rfloor = 1$）
- 两个 3×3 的感受野等价于一个 5×5，但参数量更少（$2 \times 9 < 25$），且中间多了一层非线性

---

## BatchNorm2d：批归一化

### 数学公式

对每个通道 $c$，在 batch 维度上标准化：

$$\mu_c = \frac{1}{N} \sum_{i=1}^{N} x_i \quad \text{（批均值）}$$

$$\sigma_c^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu_c)^2 \quad \text{（批方差）}$$

$$\hat{x}_i = \frac{x_i - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} \quad \text{（标准化）}$$

$$y_i = \gamma_c \cdot \hat{x}_i + \beta_c \quad \text{（缩放与平移）}$$

其中 $\gamma_c$ 和 $\beta_c$ 是**可学习参数**，$\epsilon = 10^{-5}$ 防止除零。

**为什么需要 $\gamma$ 和 $\beta$？**

直接将输入标准化为均值 0、方差 1 可能破坏网络的表达能力——比如 ReLU 之后的输出本应是正值，标准化后会有一半变成负值。$\gamma$ 和 $\beta$ 让网络可以自己学出合适的均值和方差。

### 训练 vs 推理模式

| | 训练模式 | 推理模式 |
|---|---|---|
| $\mu$ | 当前 batch 计算 | 训练时累积的运行均值 |
| $\sigma^2$ | 当前 batch 计算 | 训练时累积的运行方差 |

推理时使用全局统计量，保证确定性输出。本项目在 `Model.eval()` 时 PyTorch 自动切换（[src/train/engine.py:159](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/engine.py#L159)）。

### 为什么 BN 放在 ReLU 之前？

本项目采用 Conv2d → BatchNorm → ReLU 的顺序（[src/model/layers.py:89-117](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/model/layers.py#L89-L117)）。

BN-before-ReLU 的优势：
- 标准化后的分布对称（均值 0），ReLU 的阈值 0 能有效产生稀疏性
- 实践中比 ReLU-before-BN 收敛更快，训练更稳定

---

## ReLU：线性整流单元

### 数学公式

$$\text{ReLU}(x) = \max(0, x)$$

导数：

$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

### 为什么不用 Sigmoid？

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Sigmoid 有三个致命问题：

1. **梯度消失**：当 $|x|$ 较大时，$\sigma'(x) \approx 0$。深层网络的反向传播中梯度连乘，指数级衰减

2. **非零均值输出**：Sigmoid 输出恒为正值（0~1），下一层输入的均值不是 0。这导致反向传播时梯度方向一致（全正或全负），参数更新呈锯齿状，收敛缓慢

3. **指数计算开销大**

ReLU 的优点：
- 正区间梯度恒为 1，无梯度消失
- 计算简单（一个 threshold 操作）
- 产生稀疏激活（约 50% 神经元输出 0），有利于特征解耦

### Dead Neuron 问题

如果某个神经元始终输出 0（所有输入都在负区间），ReLU 梯度恒为 0，该神经元永远不更新。

**本项目如何缓解：** BatchNorm 将输入标准化为均值 0 的分布，保证大约一半的样本落入 ReLU 的正区间，大幅降低 dead neuron 风险。

---

## MaxPool2d：最大池化

### 数学公式

对 2×2 窗口，步长 2：

$$\text{MaxPool}(X)[i, j] = \max\left\{ X[2i, 2j], X[2i, 2j+1], X[2i+1, 2j], X[2i+1, 2j+1] \right\}$$

### 三大作用

**1. 下采样（降维）**

每次 2×2 MaxPool 将空间分辨率减半（28→14→7），后续层参数所需输入维度指数级减少。两层池化后将 28×28=784 维降到 7×7×64=3136 维但保留了 64 个高级特征通道。

**2. 局部平移不变性**

如果数字"3"平移 1 个像素，2×2 窗口内的最大值大概率不变。这使得分类对微小平移不敏感——证明：

$$|\text{MaxPool}(\text{shift}_{\delta}(X)) - \text{MaxPool}(X)| = 0 \quad \text{当} \; \|\delta\| < \frac{\text{pool\_size}}{2}$$

这比数据增强（RandomAffine 10° 旋转 + 10% 平移）更高效地增强了鲁棒性。

**3. 增大感受野**

每次池化后，下一层卷积的实际感受野翻倍。两层池化后，第 3 个 ConvBlock 的 3×3 卷积可覆盖原始图像的 ~8×8 区域。

### 为什么不用 stride-2 卷积替代池化？

stride-2 卷积也能降维，但有可学习参数 = 可能过拟合。MaxPool 零参数，更简单，且取最大值提供了一种强形式的局部不变性——这是加权求和做不到的。

---

## Dropout：随机失活

### 数学公式

训练时，每个神经元以概率 $p$ 被置零，存活神经元放大 $1/(1-p)$ 倍：

$$y_j = \begin{cases} 0 & \text{以概率 } p \\ \frac{x_j}{1-p} & \text{以概率 } 1-p \end{cases}$$

即 $y = \frac{1}{1-p} \cdot r \odot x$，其中 $r_j \sim \text{Bernoulli}(1-p)$。

推理时 Dropout 关闭，所有神经元正常工作。

### 为什么 1/(1-p) 缩放？

保证输出的期望值不变：

$$\mathbb{E}[y_j] = \mathbb{E}\left[\frac{r_j \cdot x_j}{1-p}\right] = \frac{(1-p) \cdot x_j}{1-p} = x_j$$

### 集成解释（Ensemble View）

有 $n$ 个神经元时，训练时随机采样出 $2^n$ 种不同的子网络。每个 batch 训练一个不同的子网络，Dropout 等价于训练 $2^n$ 个网络，在推理时隐式地做了模型集成。

### 为什么只有全连接层用 Dropout？

卷积层天然有参数共享，参数量少，不易过拟合。全连接层参数量大（本项目 FC 层 401,536 个参数占总参数的 95%），Dropout 提供正则化。

本项目中 Dropout 仅应用于 LinearBlock 的全连接层之后（[src/model/layers.py:223-224](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/model/layers.py#L223-L224)，`dropout=0.5`）。

---

## Linear：全连接层

### 数学公式

$$Y = X W^T + b$$

展开：

$$y_j = b_j + \sum_{i=1}^{d_{in}} x_i \cdot W_{j,i}, \quad j = 1, \dots, d_{out}$$

### 为什么需要非线性激活？

多个线性层的叠加仍然是线性的：

$$f_2(f_1(X)) = (X W_1^T + b_1) W_2^T + b_2 = X (W_2 W_1)^T + (b_1 W_2^T + b_2)$$

两个线性层等价于一个线性层。必须在层间插入非线性激活（本项目用 ReLU），否则深度无意义。

---

## 源码位置

- ConvBlock（Conv2d + BatchNorm2d + ReLU + MaxPool2d）：`src/model/layers.py:20-145`
- LinearBlock（Linear + BatchNorm1d + ReLU + Dropout）：`src/model/layers.py:148-243`
- 卷积核大小默认值：`config/default_params.py:49`
- Dropout 默认值：`config/default_params.py:55`
