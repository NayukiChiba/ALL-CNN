# 损失函数推导

## 从 Logits 到概率：Softmax

网络输出 10 个原始 logits $\mathbf{z} = (z_0, z_1, \dots, z_9)$。Softmax 将 logits 映射为概率分布：

$$p_i = \text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=0}^{9} e^{z_j}}, \quad i = 0, \dots, 9$$

**性质：**
- $p_i \in (0, 1)$ 且 $\sum_{i=0}^{9} p_i = 1$——构成合法概率分布
- 单调保序：$z_i > z_j \Rightarrow p_i > p_j$，不改变相对排序
- Softmax 不是平移不变的（加常数 $c$ 会改变分母），但 argmax 是

---

## 负对数似然损失（NLLLoss）

在 MNIST 分类中，每个样本只有一个正确类别 $y \in \{0, 1, \dots, 9\}$（one-hot 编码 $\mathbf{1}_y$）。我们希望最大化正确类别的概率 $p_y$。

等价于最小化 $-\log p_y$（对数函数单调递增，取负号将最大化转为最小化）：

$$\text{NLL} = -\log p_y = -\log\left(\frac{e^{z_y}}{\sum_{j} e^{z_j}}\right)$$

---

## CrossEntropyLoss 的数值稳定实现

直接计算 $-\log(\text{softmax}(z))$ 在数值上不稳定——当 $z_i$ 很大时 $e^{z_i}$ 会溢出。

PyTorch 的 `nn.CrossEntropyLoss` 将 LogSoftmax 和 NLLLoss 合并在一个算子中，使用 **Log-Sum-Exp 技巧**：

$$\log \sum_j e^{z_j} = \alpha + \log \sum_j e^{z_j - \alpha}$$

其中 $\alpha = \max_j z_j$。减去最大值后 $e^{z_j - \alpha} \leq 1$，不会溢出。

完整的 `CrossEntropyLoss` 计算：

$$\mathcal{L} = -z_y + \log\sum_{j=0}^{9} e^{z_j}$$

或等效地：

$$\mathcal{L} = -\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right) = \log\sum_j e^{z_j} - z_y$$

批平均（`reduction='mean'`，[src/train/loss.py:25](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/loss.py#L25)）：

$$\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{b=1}^{B} \left( \log\sum_j e^{z^{(b)}_j} - z^{(b)}_{y^{(b)}} \right)$$

---

## 梯度推导

CrossEntropyLoss 不仅数值稳定，而且梯度具有极为优美的形式：

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \delta_{i,y}$$

其中 $\delta_{i,y} = 1$ 当 $i = y$（正确类别），否则为 0。

**推导过程：**

首先写出损失：

$$\mathcal{L} = -\log p_y = -\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right) = \log\sum_j e^{z_j} - z_y$$

对于正确类别 $i = y$：

$$\frac{\partial \mathcal{L}}{\partial z_y} = \frac{e^{z_y}}{\sum_j e^{z_j}} - 1 = p_y - 1$$

对于错误类别 $i \neq y$：

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{e^{z_i}}{\sum_j e^{z_j}} - 0 = p_i$$

合写为：

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \delta_{i,y}$$

**解读：** 梯度 = 预测概率 - 真实概率（one-hot）。如果模型已经输出正确类别的高概率（$p_y \approx 1$），梯度接近 0，参数几乎不更新。如果模型错了（$p_y \approx 0$），梯度接近 -1，产生强烈的修正信号。

---

## 为什么不用 MSE？

均方误差（MSE）用于分类的梯度：

$$\frac{\partial \text{MSE}}{\partial z_i} = 2(p_i \cdot \sum_j p_j^2 - p_i^2) \cdot (\text{terms})$$

梯度中存在 $p_i(1-p_i)$ 因子——当预测接近 0 或 1 时，梯度消失，训练停滞。CrossEntropy 的梯度是线性的（$p_i - \delta_{i,y}$），学习信号始终有效。

---

## 源码位置

- 损失函数创建：`src/train/loss.py:15-26`
- 训练引擎中调用：`src/train/engine.py:111`（`loss = criterion(logits, labels)`）
