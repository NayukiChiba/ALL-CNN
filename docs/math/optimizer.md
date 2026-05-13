# 优化器

## 从 SGD 出发

最基本的随机梯度下降（SGD）更新规则：

$$\theta_t = \theta_{t-1} - \eta \cdot g_t$$

其中 $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ 是当前 batch 的梯度，$\eta$ 是学习率。

SGD 的问题：
- **学习率敏感**：太大震荡，太小收敛慢
- **各向异性**：不同参数的梯度尺度不同，单一学习率无法兼顾
- **鞍点陷阱**：梯度接近 0 的鞍点区域，SGD 几乎停滞

---

## 1. SGD + Momentum

### 公式

动量法引入一阶矩估计，给梯度加入"惯性"：

$$m_t = \beta \cdot m_{t-1} + (1 - \beta) \cdot g_t$$

$$\theta_t = \theta_{t-1} - \eta \cdot m_t$$

默认 $\beta = 0.9$（本项目默认 momentum=0.9）。

### 物理直觉

将优化过程看作小球在损失曲面上滚动。梯度是推力，动量是速度。小球在平坦方向累积速度加速下降，在陡峭方向震荡抵消。

### 本项目的 SGD 实现

**源码**: [cnnlib/training/optimizer.py:28-32](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/optimizer.py#L28-L32)

```python
def _buildSGD(model, lr, weightDecay, **kw):
    momentum = kw.pop("momentum", 0.9)
    return SGD(model.parameters(), lr=lr, momentum=momentum,
               weight_decay=weightDecay, **kw)
```

### 优点与缺点

| 优点 | 缺点 |
|------|------|
| 收敛方向更平滑 | 仍需手工调学习率 |
| 积累速度穿越平坦区域 | 对所有参数统一学习率 |
| 在 CV 竞赛中泛化通常更好 | 收敛慢于 Adam（特别是稀疏梯度场景） |

---

## 2. RMSprop

### 公式

不同参数需要不同学习率。RMSprop 维护梯度的二阶矩（方差）估计：

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

然后用 $\sqrt{v_t}$ 归一化每个参数的学习率：

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{g_t}{\sqrt{v_t} + \epsilon}$$

默认 $\beta_2 = 0.99$，$\epsilon = 10^{-8}$。

### 效果

梯度波动大的参数自动获得更小的有效学习率；梯度稳定小的参数获得更大的有效学习率。对 RNN 和非平稳目标特别有效。

### 本项目的 RMSprop 实现

**源码**: [cnnlib/training/optimizer.py:35-38](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/optimizer.py#L35-L38)

```python
def _buildRMSprop(model, lr, weightDecay, **kw):
    return RMSprop(model.parameters(), lr=lr,
                   weight_decay=weightDecay, **kw)
```

---

## 3. Adam：动量 + 自适应 = 两者之长

Adam（Adaptive Moment Estimation）结合动量（一阶矩）和 RMSprop（二阶矩）。

### 完整算法

$$\begin{aligned}
m_t &= \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \\[4pt]
v_t &= \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
\end{aligned}$$

其中 $g_t^2$ 表示逐元素平方。

### 偏差校正

$m_t$ 和 $v_t$ 初始化为 0，导致训练初期估计偏小。Adam 对矩估计做偏差校正：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

随着 $t \to \infty$，$\beta_1^t \to 0$，校正因子趋于 1。

### 参数更新

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**解读**：
- $\hat{m}_t$ 提供带惯性的方向（动量）
- $1/\sqrt{\hat{v}_t}$ 提供逐参数的自适应步长缩放
- $\epsilon = 10^{-8}$ 防止除零

### 本项目的 Adam 实现

**源码**: [cnnlib/training/optimizer.py:18-20](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/optimizer.py#L18-L20)

```python
def _buildAdam(model, lr, weightDecay, **kw):
    betas = kw.pop("betas", (0.9, 0.999))
    return Adam(model.parameters(), lr=lr, weight_decay=weightDecay,
                betas=betas, **kw)
```

### 超参数

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| $\eta$ (lr) | 0.001 | 初始学习率 |
| $\beta_1$ | 0.9 | 一阶矩衰减系数 |
| $\beta_2$ | 0.999 | 二阶矩衰减系数 |
| $\epsilon$ | $10^{-8}$ | 数值稳定项 |
| weight_decay | $10^{-4}$ | 权重衰减（L2 正则） |

---

## 4. AdamW：解耦 Weight Decay

### 为什么需要 AdamW

在标准 Adam 中，L2 正则化通过将 $\lambda \theta$ 加到梯度 $g_t$ 中实现。但 Adam 的自适应学习率 $1/\sqrt{\hat{v}_t}$ 也会作用于 $\lambda\theta$ 项——导致不同参数的正则化强度不一致。

**AdamW** (Loshchilov & Hutter, 2019) 将 Weight Decay 从梯度更新中解耦，直接在权重上施加衰减：

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \cdot \theta_{t-1}$$

注意最后一项 $-\eta\lambda \cdot \theta_{t-1}$ 是独立于 $m_t$ 和 $v_t$ 的权重衰减。

### Adam vs AdamW 的更新差异

$$\begin{aligned}
\text{Adam (L2 reg):} \quad & \theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t + \lambda\theta_{t-1}}{\sqrt{\hat{v}_t} + \epsilon} \\[4pt]
\text{AdamW (decoupled):} \quad & \theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta\lambda \cdot \theta_{t-1}
\end{aligned}$$

在 Adam 中，$\lambda\theta$ 被 $\sqrt{\hat{v}_t}$ 缩放；在 AdamW 中，衰减与自适应学习率无关。

### 本项目的 AdamW 实现

**源码**: [cnnlib/training/optimizer.py:23-25](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/optimizer.py#L23-L25)

```python
def _buildAdamW(model, lr, weightDecay, **kw):
    betas = kw.pop("betas", (0.9, 0.999))
    return AdamW(model.parameters(), lr=lr, weight_decay=weightDecay,
                 betas=betas, **kw)
```

**默认 weight_decay = 1e-4**。

---

## 四种优化器对比



| | SGD+Momentum | RMSprop | Adam | AdamW |
|---|:---:|:---:|:---:|:---:|
| 一阶矩（动量） | ✓ | ✗ | ✓ | ✓ |
| 二阶矩（自适应 LR） | ✗ | ✓ | ✓ | ✓ |
| 偏差校正 | ✗ | ✗ | ✓ | ✓ |
| Weight Decay 方式 | 加到梯度 | 加到梯度 | 加到梯度（PyTorch 已修正） | 解耦（Decoupled） |
| 额外内存 | 1× | 1× | 2× | 2× |
| 收敛速度 | 慢 | 中 | 快 | 快 |
| 泛化能力 | 可能最好 | 中 | 通常较好 | 通常最好（含 WD） |
| 推荐场景 | 大 batch CV 竞赛 | RNN/强化学习 | 快速原型 | **默认推荐** |

---

## 本项目的优化器工厂

**源码**: [cnnlib/training/optimizer.py:49-75](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/optimizer.py#L49-L75)

```python
def createOptimizer(model, name="adam", lr=0.001, weight_decay=0.0, **kwargs):
    """
    name: adam / adamw / sgd / rmsprop
    """
    # 查表返回对应优化器实例
```

CLI 使用: `python main.py train --optimizer adamw --lr 0.001 --weight-decay 1e-4`

---

## 优化器选择建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 快速实验/原型 | Adam | 默认参数通常 work，无需调参 |
| 正式训练/追求最优 | AdamW | 解耦 weight decay，泛化更好 |
| 大 batch 训练 | SGD+Momentum | 大 batch 下自适应方法优势减弱 |
| 特殊任务（RL/RNN） | RMSprop | 非平稳目标场景更稳定 |

---

## 相关文档

- [学习率调度器](/math/schedulers) — ReduceLROnPlateau、StepLR、Cosine 等
- [L1/L2/Weight Decay](/math/regularization) — Weight Decay 的数学基础
- [梯度裁剪](/math/gradient-clipping) — 防止梯度爆炸的最后防线
- [训练流程](/architecture/training) — optimizer.step() 在训练循环中的位置
