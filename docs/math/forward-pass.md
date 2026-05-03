# 完整前向传播方程链

本文逐层展示从输入图像到分类 logits 的完整计算过程，包含每层的形状变化和参数量。

---

## 网络结构一览

```
输入 X₀ : (B, 1, 28, 28)
  │
  ├─ ConvBlock #1 ────────────────────────────────
  │   ├─ Conv2d(1→32, k=3, p=1)     → (B, 32, 28, 28)
  │   ├─ BatchNorm2d(32)             → (B, 32, 28, 28)
  │   ├─ ReLU                        → (B, 32, 28, 28)
  │   └─ MaxPool2d(k=2, s=2)        → (B, 32, 14, 14)
  │
  ├─ ConvBlock #2 ────────────────────────────────
  │   ├─ Conv2d(32→64, k=3, p=1)    → (B, 64, 14, 14)
  │   ├─ BatchNorm2d(64)             → (B, 64, 14, 14)
  │   ├─ ReLU                        → (B, 64, 14, 14)
  │   └─ MaxPool2d(k=2, s=2)        → (B, 64,  7,  7)
  │
  ├─ Flatten                         → (B, 3136)
  │
  ├─ LinearBlock ─────────────────────────────────
  │   ├─ Linear(3136→128)            → (B, 128)
  │   ├─ BatchNorm1d(128)            → (B, 128)
  │   ├─ ReLU                        → (B, 128)
  │   └─ Dropout(p=0.5)              → (B, 128)
  │
  ├─ Linear(128→10)                  → (B, 10)
  │
输出 logits Z : (B, 10)
```

---

## 逐层方程

### 输入

$$X_0 \in \mathbb{R}^{B \times 1 \times 28 \times 28}$$

每张图是经过 `Normalize((0.1307,), (0.3081,))` 标准化后的灰度图。

### ConvBlock #1

**Conv2d — 32 个 1×3×3 卷积核：**

$$Z_1[c, i, j] = b_1[c] + \sum_{u=0}^{2} \sum_{v=0}^{2} W_1[c, 0, u, v] \cdot X_0[0, i+u-1, j+v-1]$$

其中 $c = 0, \dots, 31$，padding=1 使得 $i \in [0, 27], j \in [0, 27]$。

输出形状：$(B, 32, 28, 28)$

**BatchNorm2d：**

$$\hat{Z}_1[c, i, j] = \gamma_{1}[c] \cdot \frac{Z_1[c, i, j] - \mu_{1}[c]}{\sqrt{\sigma_{1}^{2}[c] + 10^{-5}}} + \beta_{1}[c]$$

**ReLU：**

$$A_1[c, i, j] = \max(0, \hat{Z}_1[c, i, j])$$

**MaxPool2d（2×2 非重叠）：**

$$P_1[c, i, j] = \max\left\{A_1[c, 2i, 2j], A_1[c, 2i, 2j+1], A_1[c, 2i+1, 2j], A_1[c, 2i+1, 2j+1]\right\}$$

其中 $i, j \in [0, 13]$。

输出形状：$(B, 32, 14, 14)$

### ConvBlock #2

**Conv2d — 64 个 32×3×3 卷积核：**

$$Z_2[k, i, j] = b_2[k] + \sum_{c=0}^{31} \sum_{u=0}^{2} \sum_{v=0}^{2} W_2[k, c, u, v] \cdot P_1[c, i+u-1, j+v-1]$$

其中 $k = 0, \dots, 63$。

输出形状：$(B, 64, 14, 14)$，经过 BN → ReLU → MaxPool2d 后：

$$P_2 \in \mathbb{R}^{B \times 64 \times 7 \times 7}$$

### Flatten

$$F = \text{reshape}(P_2, \; [B, 64 \times 7 \times 7]) = \text{reshape}(P_2, \; [B, 3136])$$

### LinearBlock

**Linear — 128×3136 权重矩阵：**

$$H_0[j] = b_{fc}[j] + \sum_{i=0}^{3135} F[i] \cdot W_{fc}[j, i], \quad j = 0, \dots, 127$$

再经过 BatchNorm1d → ReLU → Dropout(0.5)：

$$H = \text{Dropout}_{0.5}\left(\text{ReLU}\left(\text{BN}(H_0)\right)\right) \in \mathbb{R}^{B \times 128}$$

训练时 Dropout 随机置零一半神经元并 2× 缩放存活神经元；推理时 Dropout 关闭。

### 输出层

$$\text{logits}[k] = b_{out}[k] + \sum_{i=0}^{127} H[i] \cdot W_{out}[k, i], \quad k = 0, \dots, 9$$

输出形状：$(B, 10)$—— 10 个原始得分（未经 softmax）。

---

## 参数量明细

| 层 | 参数 | 个数 | 累计 |
|----|------|------|------|
| Conv2d #1 | $W: 32 \times 1 \times 3 \times 3 = 288$, $b: 32$ | 320 | 320 |
| BatchNorm2d #1 | $\gamma: 32$, $\beta: 32$ | 64 | 384 |
| ReLU #1 | 无 | 0 | 384 |
| MaxPool2d #1 | 无 | 0 | 384 |
| Conv2d #2 | $W: 64 \times 32 \times 3 \times 3 = 18,432$, $b: 64$ | 18,496 | 18,880 |
| BatchNorm2d #2 | $\gamma: 64$, $\beta: 64$ | 128 | 19,008 |
| ReLU #2 | 无 | 0 | 19,008 |
| MaxPool2d #2 | 无 | 0 | 19,008 |
| Flatten | 无 | 0 | 19,008 |
| Linear(FC) | $W: 128 \times 3136 = 401,408$, $b: 128$ | 401,536 | 420,544 |
| BatchNorm1d | $\gamma: 128$, $\beta: 128$ | 256 | 420,800 |
| ReLU | 无 | 0 | 420,800 |
| Dropout | 无 | 0 | 420,800 |
| Linear(out) | $W: 10 \times 128 = 1,280$, $b: 10$ | 1,290 | **422,090** |

**关键观察：** 全连接层（401,536 / 422,090 ≈ 95.1%）占据了绝大多数参数。这是因为从卷积的 64×7×7=3136 维特征空间映射到 128 维隐空间是信息压缩的瓶颈。卷积层的参数份额 (< 5%) 体现了参数共享的巨大优势。

---

## 源码位置

- 模型 forward 方法：`src/model/cnn.py:88-102`
- 卷积块构建循环：`src/model/cnn.py:65-76`
- 扁平化尺寸自动计算：`src/model/cnn.py:79-80`
- ConvBlock forward：`src/model/layers.py:133-145`
- LinearBlock forward：`src/model/layers.py:227-243`
