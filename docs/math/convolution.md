# 离散卷积原理

## 一维离散卷积

两个一维离散序列 $f$ 和 $g$ 的卷积定义为：

$$(f * g)[n] = \sum_{k=-\infty}^{\infty} f[k] \cdot g[n - k]$$

直观理解：将卷积核 $g$ 翻转后，在序列 $f$ 上滑动，每个位置计算加权和。

### 为什么叫"卷积"？

"卷"指将核函数翻转（反转）的操作，"积"指相乘再求和。在深度学习中，实际使用的是**互相关（cross-correlation）**：

$$(f \star g)[n] = \sum_{k=-\infty}^{\infty} f[k] \cdot g[n + k]$$

互相关不翻转核函数，直接滑动加权求和。由于神经网络的卷积核权重是从数据中学出来的，翻转与否不影响学习结果——网络会自动学到它需要的方向。

PyTorch 的 `nn.Conv2d` 实现的是互相关，但沿用习惯称为"卷积"。

---

## 二维离散卷积（图像）

将一维推广到二维，对图像 $I \in \mathbb{R}^{H \times W}$ 和卷积核 $K \in \mathbb{R}^{k_h \times k_w}$：

$$(I \star K)[i, j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I[i+m, \; j+n] \cdot K[m, n]$$

对于多通道输入 $X \in \mathbb{R}^{C_{in} \times H \times W}$，每个输出通道有一个 3D 卷积核 $W \in \mathbb{R}^{C_{in} \times k_h \times k_w}$：

$$Y[c_{out}, i, j] = b[c_{out}] + \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} W[c_{out}, c_{in}, m, n] \cdot X[c_{in}, i+m, j+n]$$

其中 $b[c_{out}]$ 是偏置项。

---

## 为什么对图像使用卷积？

### 1. 局部感受野（Local Receptive Field）

传统全连接层中，每个输出神经元连接所有输入像素。对于 $28 \times 28$ 的图像，单个神经元就有 784 个权重。而 $3 \times 3$ 卷积核只有 9 个权重，每个输出位置只"看到"输入的 3x3 邻域。

这符合图像的物理特性：**邻近像素相关性高，远处像素相关性低**。边缘检测不需要看整张图，看 3x3 邻域就够了。

### 2. 参数共享（Parameter Sharing / Translation Equivariance）

同一个卷积核在整个图像上滑动，所有位置共享同一组权重。这意味着：

- 参数量从 $O(H \times W)$ 降到 $O(k^2)$，大幅减少过拟合风险
- **平移等变性**：如果输入图像平移 $\Delta$，卷积输出也平移 $\Delta$。即：

$$\text{Conv}(\text{shift}(X)) = \text{shift}(\text{Conv}(X))$$

这一点至关重要——数字 "3" 出现在图像的左边还是右边，卷积都能用同一组权重检测到它。

### 3. 层级特征抽象

堆叠多层卷积，可以从低级特征逐步构建高级语义：

| 层级 | 感受野 | 学到的特征 |
|------|--------|-----------|
| 第 1 层 (3x3) | 3x3 | 边缘、角点、纹理方向 |
| 第 2 层 (3x3 在 2x池化后) | ~8x8 | 简单形状（弧线、交叉） |
| 全连接层 | 全局 | 数字级别的语义组合 |

在本项目中（`src/model/cnn.py:88-102`），两层 3x3 卷积 + 2x2 MaxPool 后，每个输出位置的有效感受野约为 8x8 像素，覆盖 MNIST 数字的大部分笔画结构。

---

## 卷积核的物理意义

以 Sobel 边缘检测算子为例，这是一种人工设计的卷积核：

$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

$G_x$ 检测垂直边缘，$G_y$ 检测水平边缘。

CNN 不依赖人工设计的卷积核——它通过反向传播从数据中学习最优的卷积核权重。第一层学到的 32 个 $3 \times 3$ 卷积核（[src/model/layers.py:89-94](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/model/layers.py#L89-L94)）可以理解为 32 种不同的"边缘检测器"，每种捕捉笔画的不同方向或粗细。

---

## 源码位置

- 卷积块定义：`src/model/layers.py:89-94`（Conv2d 子层）
- 模型组装：`src/model/cnn.py:65-76`（循环构建卷积层序列）
- 默认卷积核大小：`config/default_params.py:49`（`CONV_KERNEL_SIZE = 3`）
