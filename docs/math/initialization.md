# Kaiming & Xavier 初始化

## 为什么权重初始化重要

深度网络的训练对初始权重极为敏感。不当的初始化会导致两大问题：

1. **梯度消失**: 激活值逐层缩小 → 反向梯度趋近于 0 → 浅层参数几乎不更新
2. **梯度爆炸**: 激活值逐层放大 → 反向梯度指数级增长 → 参数更新发散，loss 变为 NaN

好的初始化策略的目标：**保持各层激活值和梯度的方差不变**，使信号可以稳定地向前向后传播。

---

## Xavier (Glorot) 初始化

**提出者**: Glorot & Bengio (2010). *Understanding the difficulty of training deep feedforward neural networks.*

### 推导思路

对于全连接层 $y = Wx + b$（忽略偏置），假设：
- $x$ 的方差为 $\text{Var}(x)$
- $W$ 的所有元素独立同分布，均值为 0，方差为 $\text{Var}(W)$

则输出 $y$ 的方差为：

$$\text{Var}(y_j) = \text{Var}\left(\sum_{i=1}^{n_{in}} W_{ji} x_i\right) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)$$

（假设 $W$ 和 $x$ 独立，且 $x_i$ 之间独立同分布）

为了让 $\text{Var}(y) = \text{Var}(x)$（前向方差不变）：

$$\text{Var}(W) = \frac{1}{n_{in}}$$

但如果只考虑前向，反向传播时梯度方差会收缩。平衡前向和后向的需求：

$$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$$

### 均匀分布形式

对于均匀分布 $U[-a, a]$，方差为 $\frac{a^2}{3}$。令 $\frac{a^2}{3} = \frac{2}{n_{in} + n_{out}}$：

$$W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]$$

### 正态分布形式

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

### PyTorch 代码

```python
nn.init.xavier_uniform_(weight)   # 均匀分布
nn.init.xavier_normal_(weight)    # 正态分布
```

### 适用场景

- 激活函数在零附近近似线性：Tanh、Sigmoid（在零附近）
- **本项目使用**: LeNet-5（所有 Conv2d 和 Linear 层）

---

## Kaiming (He) 初始化

**提出者**: He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.*

### 问题

Xavier 假设激活函数是线性的（或近似线性）。但 **ReLU 将一半的激活值归零**（$x \leq 0$ 时输出为 0），方差会减半。

### 推导

对于 ReLU 激活后的值 $y = \text{ReLU}(Wx + b)$：

$$\text{Var}(y) = \frac{1}{2} \cdot n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)$$

（ReLU 将一半输出置零，方差减半——乘以 1/2）

为使 $\text{Var}(y) = \text{Var}(x)$：

$$\text{Var}(W) = \frac{2}{n_{in}} \quad \text{（fan_in 模式）}$$

或考虑反向传播：

$$\text{Var}(W) = \frac{2}{n_{out}} \quad \text{（fan_out 模式）}$$

### fan_in vs fan_out

| 模式 | 公式 | 含义 |
|------|------|------|
| `fan_in` | $\text{Var}(W) = \frac{2}{n_{in}}$ | 保持前向传播激活方差不变 |
| `fan_out` | $\text{Var}(W) = \frac{2}{n_{out}}$ | 保持反向传播梯度方差不变 |

实际中 `fan_out` 通常效果略好（本项目的默认选择）。

### 正态分布形式

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n}}\right)$$

其中 $n = n_{in}$（fan_in 模式）或 $n = n_{out}$（fan_out 模式）。

### PyTorch 代码

```python
nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")
nn.init.kaiming_uniform_(weight, mode="fan_in", nonlinearity="relu")
```

`nonlinearity` 参数用于调整修正因子：
- `"relu"` → 因子为 $\sqrt{2}$
- `"leaky_relu"` → 因子为 $\sqrt{2 / (1 + \text{negative\_slope}^2)}$

### 适用场景

- ReLU、LeakyReLU、PReLU 等非对称激活函数
- **本项目使用**: AlexNet（Conv）、VGG（Conv）、NiN（所有 Conv）、GoogLeNet（Conv）

---

## Xavier vs Kaiming 对比

| | Xavier (Glorot) | Kaiming (He) |
|---|---|---|
| **假设激活** | 线性 / Tanh / Sigmoid（零附近近似线性） | ReLU 及其变体 |
| **方差目标** | $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$ | $\text{Var}(W) = \frac{2}{n}$（$n$ 为 fan_in 或 fan_out） |
| **核心理由** | 平衡前后向信号传播 | 补偿 ReLU 的 50% 归零 |
| **项目中使用** | LeNet-5 | AlexNet, VGG, NiN, GoogLeNet |


---

## 本项目的初始化策略

| 模型 | Conv 层 | Linear 层 | BN 层 |
|------|---------|----------|-------|
| LeNet-5 | Xavier uniform | Xavier uniform | — |
| AlexNet | Kaiming normal (fan_out, relu) | Xavier normal | — |
| VGG | Kaiming normal (fan_out, relu) | Normal(0, 0.01) | γ=1, β=0 |
| NiN | Kaiming normal (fan_out, relu) | — | — |
| GoogLeNet | Kaiming normal (fan_out, relu) | Normal(0, 0.01) | γ=1, β=0 |

### 代码示例 (LeNet-5)

```python
def _initWeights(self):
    for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

### 代码示例 (AlexNet)

```python
def _initWeights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
```

### 代码示例 (VGG / GoogLeNet)

```python
# Conv: Kaiming
nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
# BatchNorm: gamma=1, beta=0
nn.init.constant_(m.weight, 1)
nn.init.constant_(m.bias, 0)
# Linear: Normal(0, 0.01) — 匹配原始 VGG 论文
nn.init.normal_(m.weight, mean=0, std=0.01)
```

---

## 不当初始化的后果


### 梯度消失

- 原因: 权重太小 → 各层激活值趋近于 0 → 梯度指数衰减
- 症状: 浅层参数几乎不变化，loss 下降极慢或停滞
- 常见于: Sigmoid/Tanh + 深层网络 + 不当初始化

### 梯度爆炸

- 原因: 权重太大 → 各层激活值指数增长 → 梯度指数增长
- 症状: loss 突然变为 NaN，参数值爆炸
- 常见于: RNN、深层 CNN + 不当初始化 + 无梯度裁剪

### 修复方案
- 使用适当的初始化（Kaiming / Xavier）
- 使用 Batch Normalization（减少对初始化的依赖）
- 使用梯度裁剪（防止梯度爆炸）
- 使用残差连接（ResNet 等）

---

## 相关文档

- [Batch Normalization](/math/batch-normalization) — BN 降低对初始化的依赖
- [梯度裁剪](/math/gradient-clipping) — 防止梯度爆炸的最后防线
- [各层公式](/math/layers) — ReLU/Tanh 的数学定义
- [LeNet-5](/models/lenet) — Xavier 初始化的使用者
- [AlexNet](/models/alexnet) — Kaiming 初始化的早期采用者
