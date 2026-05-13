# L1 / L2 / Weight Decay / Label Smoothing

## L2 正则化与 Weight Decay

### 经典 L2 正则化

在损失函数中附加参数平方和的惩罚项：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \frac{\lambda}{2} \sum_i \theta_i^2$$

梯度变为：

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \theta_i} = \frac{\partial \mathcal{L}_{\text{CE}}}{\partial \theta_i} + \lambda \cdot \theta_i$$

参数更新（SGD）：

$$\theta_i^{(t+1)} = \theta_i^{(t)} - \eta \left(g_i^{(t)} + \lambda \theta_i^{(t)}\right) = (1 - \eta\lambda) \theta_i^{(t)} - \eta g_i^{(t)}$$

每次更新，权重先衰减 $1 - \eta\lambda$（趋向于 0），再沿梯度方向移动——因此称为 Weight Decay。

### AdamW：解耦 Weight Decay

在 Adam 等自适应优化器中，直接将 L2 正则化加到梯度中会导致问题——自适应学习率的缩放也作用于正则化项，使得不同参数的正则化强度不一致。

**AdamW** (Loshchilov & Hutter, 2019) 将 Weight Decay 从梯度中**解耦**，直接在权重上做衰减：

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \cdot \theta_{t-1}$$

注意最后一项 $-\eta\lambda \cdot \theta_{t-1}$ 是独立于 Adam 的矩估计的权重衰减，不受 $m_t$ 和 $v_t$ 影响。

**本项目的实现**: [cnnlib/training/optimizer.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/optimizer.py) 中，`AdamW` 使用 PyTorch 内置实现（`AdamW` 类），`Adam` 的 `weight_decay` 参数在 PyTorch 中也采用解耦方式。

### 默认值

```python
weight_decay = 1e-4  # config/training.py
```

---

## L1 正则化

L1 正则化在损失函数中附加参数绝对值的惩罚：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \sum_i |\theta_i|$$

梯度：

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \theta_i} = \frac{\partial \mathcal{L}_{\text{CE}}}{\partial \theta_i} + \lambda \cdot \text{sign}(\theta_i)$$

L1 正则化倾向于产生**稀疏权重**（许多权重精确为 0）——因为参数在更新时会被减去一个恒定的值，而非按比例衰减。

本项目不使用 L1 正则化（PyTorch 优化器的 `weight_decay` 实现的是 L2 解耦衰减）。

---

## Label Smoothing

### 原理

传统分类使用 one-hot 标签 $y$：
- 正确类别 $y_{\text{true}} = 1$
- 所有其他类别 $y_{\text{false}} = 0$

这鼓励模型对正确类别输出极高的概率（$p \to 1$）——可能导致**过度自信**和泛化能力下降。

**Label Smoothing** 将 one-hot 目标"软化"：

$$y_{\text{smooth}} = (1 - \alpha) \cdot y_{\text{one-hot}} + \frac{\alpha}{K}$$

其中 $K$ 是类别总数，$\alpha$ 是平滑因子（如 0.1）。

例如对 $K=10$，$\alpha=0.1$：
- 原标签 `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]`
- 平滑后 `[0.01, 0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]`

### 效果

- 防止模型对训练数据过于自信（$p \to 1$ 是危险的——总有一些不确定性）
- 改善校准（confidence 更接近真实 accuracy）
- 等价于在 CrossEntropy 损失中加入了一个 KL 散度正则项

### 使用方式

```python
# 在 createLoss() 中支持 label_smoothing 参数
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # PyTorch >= 1.10
```

本项目通过 `createLoss()` → `**kwargs` 支持 label_smoothing 参数。

---

## 早期停止 (Early Stopping)

虽然不是数学意义上的正则化，但早期停止是防止过拟合最直接有效的方法。

**原理**: 当验证集指标不再改善时停止训练，防止模型继续记忆训练集的噪声。

**本项目实现**: [cnnlib/training/earlyStopping.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/earlyStopping.py)

```python
earlyStopping = EarlyStopping(patience=10, min_delta=0.001, mode="max")
# 每 epoch 后:
earlyStopping.step(val_accuracy)
if earlyStopping.shouldStop:
    break
```

- **patience**: 容忍多少个 epoch 无改善后停止
- **min_delta**: 视为"改善"的最小变化量
- **mode**: "max"（监控准确率等越大越好的指标）或 "min"（监控损失等越小越好的指标）

---

## 各模型中的正则化策略

| 模型 | Weight Decay | Dropout | BN | GAP | Label Smoothing |
|------|:---:|:---:|:---:|:---:|:---:|
| LeNet-5 | ✓ (1e-4) | — | — | — | 可选 |
| AlexNet | ✓ (1e-4) | 0.5 | FC 层 | — | 可选 |
| VGG | ✓ (1e-4) | 0.5 | ✓ | — | 可选 |
| NiN | ✓ (1e-4) | — | — | ✓（强正则） | 可选 |
| GoogLeNet | ✓ (1e-4) | 0.4 | ✓ | 部分 | 可选 |

---

## 相关文档

- [Dropout](/math/dropout) — 最常用的随机正则化
- [Batch Normalization](/math/batch-normalization) — BN 的隐式正则化
- [优化器](/math/optimizer) — AdamW 的解耦 weight decay
- [训练流程](/architecture/training) — EarlyStopping 的使用
