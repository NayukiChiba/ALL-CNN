# LeNet-5 (1998)

## 论文来源

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE.

**历史地位**: CNN 的开山之作。确立了卷积→池化→全连接的基本范式，影响后续所有 CNN 架构。

---

## 架构图

```
输入 X₀ : (B, 1, 32, 32)            ← 原始 MNIST 28×28 需 padding 到 32×32
  │
  ├─ C1: Conv2d(1→6, 5×5, pad=0) + Tanh
  │     输出: (B, 6, 28, 28)         ← (32-5)/1+1 = 28
  │
  ├─ S2: AvgPool2d(2×2, stride=2)
  │     输出: (B, 6, 14, 14)         ← 28/2 = 14
  │
  ├─ C3: Conv2d(6→16, 5×5, pad=0) + Tanh
  │     输出: (B, 16, 10, 10)        ← (14-5)/1+1 = 10
  │
  ├─ S4: AvgPool2d(2×2, stride=2)
  │     输出: (B, 16, 5, 5)          ← 10/2 = 5
  │
  ├─ C5: Conv2d(16→120, 5×5, pad=0) + Tanh
  │     输出: (B, 120, 1, 1)         ← (5-5)/1+1 = 1  （等价于 FC）
  │
  ├─ Flatten                         → (B, 120)
  │
  ├─ F6: Linear(120→84) + Tanh       → (B, 84)
  │
  └─ Output: Linear(84→num_classes)  → (B, num_classes)
```



---

## 逐层详解

### C1 — 第一个卷积层

$$\text{C1}[c, i, j] = \text{Tanh}\left(b_1[c] + \sum_{u=0}^{4} \sum_{v=0}^{4} W_1[c, 0, u, v] \cdot X_0[0, i+u, j+v]\right)$$

- 6 个 5×5 卷积核，无 padding（原始 LeNet 设计）
- 输入 32×32 → 输出 28×28（每边损失 2 像素）
- **参数量**: $6 \times (1 \times 5 \times 5 + 1) = 156$

### S2 — 平均池化

$$\text{S2}[c, i, j] = \frac{1}{4} \sum_{u=0}^{1} \sum_{v=0}^{1} \text{C1}[c, 2i+u, 2j+v]$$

- 2×2 窗口，步长 2，无重叠
- 28×28 → 14×14
- **参数量**: 0

### C3 — 第二个卷积层

- 16 个 5×5 卷积核，每个接受 6 通道输入
- 14×14 → 10×10
- **参数量**: $16 \times (6 \times 5 \times 5 + 1) = 2,416$
- 原始论文中 C3 与 S2 之间有不完全的连接表（16 个输出通道并非连接全部 6 个输入通道），此实现使用全连接

### S4 — 平均池化

- 10×10 → 5×5
- **参数量**: 0

### C5 — 第三个卷积层（等价于全连接）

- 120 个 5×5 卷积核。由于输入已缩至 5×5，5×5 卷积输出 1×1——这在数学上等价于一个全连接层
- **参数量**: $120 \times (16 \times 5 \times 5 + 1) = 48,120$

### F6 — 全连接隐层

$$\text{F6}[j] = \text{Tanh}\left(b_{fc}[j] + \sum_{i=0}^{119} \text{Flatten}(\text{C5})[i] \cdot W_{fc}[j, i]\right)$$

- 120 → 84
- **参数量**: $84 \times 120 + 84 = 10,164$

### Output — 分类层

- 84 → num_classes
- **参数量**: $\text{num\_classes} \times 84 + \text{num\_classes}$
- 对 MNIST (10 类): $10 \times 84 + 10 = 850$

---

## 参数量明细

| 层 | 权重 | 偏置 | 小计 | 累计 |
|----|------|------|------|------|
| C1 (1→6, 5×5) | 1×5×5×6 = 150 | 6 | 156 | 156 |
| S2 | 0 | 0 | 0 | 156 |
| C3 (6→16, 5×5) | 6×5×5×16 = 2,400 | 16 | 2,416 | 2,572 |
| S4 | 0 | 0 | 0 | 2,572 |
| C5 (16→120, 5×5) | 16×5×5×120 = 48,000 | 120 | 48,120 | 50,692 |
| F6 (120→84) | 120×84 = 10,080 | 84 | 10,164 | 60,856 |
| Output (84→10) | 84×10 = 840 | 10 | 850 | **61,706** |

> MNIST (10 类别) 时总参数约 62K。大部分参数集中在 C5 和 F6 层。

---

## 关键设计决策

### 1. Tanh 激活函数

LeNet-5 使用 Tanh 而非 Sigmoid：

$$\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- **输出零均值**: 相比 Sigmoid（输出 0~1），Tanh 输出 (-1, 1)，均值为 0，有利于下一层的学习
- **梯度更强**: Tanh 的最大梯度为 1（而 Sigmoid 为 0.25），缓解梯度消失
- **历史局限性**: 现代网络普遍使用 ReLU（不饱和、稀疏激活、计算简单）

### 2. Average Pooling

LeNet-5 使用 Average Pooling 而非 Max Pooling：

- AvgPool 保留了邻域内的全部信息（加权平均），而不是只取最大值
- 在浅层网络中，AvgPool 的平滑效果可能比 MaxPool 的稀疏选择更适合手写数字的连续笔画
- 现代网络几乎全部使用 MaxPool（更强的特征选择和局部不变性）

### 3. Xavier 初始化

LeNet-5 使用 Xavier (Glorot) 均匀初始化：

$$W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]$$

- 设计目标是保持各层激活值和梯度的方差不变
- 假设激活函数是线性的（或 Tanh/Sigmoid 在零附近近似线性）
- 对于 ReLU 激活，Kaiming 初始化更合适

详见 [Kaiming & Xavier 初始化](/math/initialization)

### 4. 无 Batch Normalization

BatchNorm 在 2015 年才被提出，远超 LeNet-5 时代。缺乏 BN 使得训练对初始化质量和学习率更敏感。

### 5. 输入尺寸 32×32（非 MNIST 原生 28×28）

MNIST 原始图像为 28×28，但 LeNet-5 设计接受 32×32。这个差异来自原始论文——他们将 28×28 的数字居中放在 32×32 的画布上。本项目的 transform 管线通过 Resize(32) 自动处理此差异。

---

## 感受野计算

使用递推公式 $r_{out} = r_{in} + (k - 1) \times \prod s_{prev}$：

| 层 | 核大小 | 步长 | 感受野 |
|----|--------|------|--------|
| 输入 | — | — | 1×1 |
| C1 (5×5) | 5 | 1 | 5×5 |
| S2 (2×2) | 2 | 2 | 6×6 |
| C3 (5×5) | 5 | 1 | 14×14 |
| S4 (2×2) | 2 | 2 | 16×16 |
| C5 (5×5) | 5 | 1 | 32×32 |

C5 层的每个神经元（1×1 输出）的感受野恰好覆盖整个 32×32 输入——这正是设计的精妙之处。

<div style="max-width:520px;margin:1em auto;font-size:13px;line-height:1.8;">
  <div style="text-align:center;font-weight:600;margin-bottom:8px;">LeNet-5 感受野增长（各层输出神经元的输入感受野）</div>
  <div style="display:flex;align-items:center;">
    <span style="width:40px;text-align:right;margin-right:8px;flex-shrink:0;">Input</span>
    <span style="height:14px;background:#3498db;width:2.86%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">1 px</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:40px;text-align:right;margin-right:8px;flex-shrink:0;">C1</span>
    <span style="height:14px;background:#3498db;width:14.29%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">5 px</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:40px;text-align:right;margin-right:8px;flex-shrink:0;">S2</span>
    <span style="height:14px;background:#3498db;width:17.14%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">6 px</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:40px;text-align:right;margin-right:8px;flex-shrink:0;">C3</span>
    <span style="height:14px;background:#3498db;width:40%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">14 px</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:40px;text-align:right;margin-right:8px;flex-shrink:0;">S4</span>
    <span style="height:14px;background:#3498db;width:45.71%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">16 px</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:40px;text-align:right;margin-right:8px;flex-shrink:0;">C5</span>
    <span style="height:14px;background:#3498db;width:91.43%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">32 px</span>
  </div>
</div>

---

## forward() 方法

```python
def forward(self, x):
    x = self.tanh(self.conv1(x))   # C1: Conv2d → Tanh
    x = self.pool1(x)              # S2: AvgPool
    x = self.tanh(self.conv2(x))   # C3: Conv2d → Tanh
    x = self.pool2(x)              # S4: AvgPool
    x = self.tanh(self.conv3(x))   # C5: Conv2d → Tanh
    x = torch.flatten(x, 1)        # Flatten
    x = self.tanh(self.fc1(x))     # F6: Linear → Tanh
    x = self.fc2(x)                # Output: Linear → logits
    return x
```

**源码**: [cnnlib/models/lenet.py:93-102](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/lenet.py#L93-L102)

---

## 与现代 LeNet 的差异

如果今天重新实现 LeNet-5，通常会做以下现代化改造：

| 原始 LeNet-5 (1998) | 现代化改造 |
|---------------------|-----------|
| Tanh 激活 | ReLU 激活 |
| Average Pooling | Max Pooling |
| 无 BN | Conv → BN → ReLU → MaxPool |
| 无 Dropout | FC 层后加 Dropout(0.5) |
| Xavier 初始化 | Kaiming 初始化 |
| 5×5 卷积 | 两个 3×3 卷积（同感受野，更多非线性） |
| 手动计算展平维度 | `infer_feature_dim()` 自动计算 |

本项目保持原始 LeNet-5 设计以求历史准确性，但使用 Xavier 初始化代替原始论文的特定初始化方案。

---

## 训练建议

- **推荐数据集**: MNIST, FashionMNIST, EMNIST（灰度 28×28 或 32×32 图像）
- **典型表现**: MNIST 上可达 99%+ 准确率
- **不推荐**: RGB 数据集（需通道转换且模型容量不足）
- **训练时间**: CPU 上约 10 分钟（20 epochs），GPU 上约 2 分钟

---

## 相关文档

- [各层公式与设计原理](/math/layers) — Tanh/AvgPool/Linear 公式
- [Kaiming & Xavier 初始化](/math/initialization) — Xavier 推导
- [感受野计算](/math/receptive-field) — 感受野递推公式
- [模型工厂](/architecture/model-factory) — 如何使用 create_model("lenet", ...)
