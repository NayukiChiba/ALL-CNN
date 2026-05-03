# Adam 优化器

## 从 SGD 出发

最基本的随机梯度下降（SGD）更新规则：

$$\theta_t = \theta_{t-1} - \eta \cdot g_t$$

其中 $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ 是当前 batch 的梯度，$\eta$ 是学习率。

SGD 的问题：
- **学习率敏感**：太大震荡，太小收敛慢
- **各向异性**：不同参数的梯度尺度不同，单一学习率无法兼顾
- **鞍点陷阱**：梯度接近 0 的鞍点区域，SGD 几乎停滞

---

## 动量（Momentum）

动量法引入一阶矩估计，给梯度加入"惯性"：

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$\theta_t = \theta_{t-1} - \eta \cdot m_t$$

默认 $\beta_1 = 0.9$。

**物理直觉：** 将优化过程看作小球在损失曲面上滚动。梯度是推力，动量是速度。小球在平坦方向累积速度加速下降，在陡峭方向震荡抵消。

---

## 自适应学习率：RMSprop

不同参数需要不同学习率。RMSprop 维护梯度的二阶矩（方差）估计：

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

然后用 $\sqrt{v_t}$ 归一化每个参数的学习率：

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{g_t}{\sqrt{v_t} + \epsilon}$$

默认 $\beta_2 = 0.999$，$\epsilon = 10^{-8}$。

**效果：** 梯度波动大的参数自动获得更小的有效学习率；梯度稳定小的参数获得更大的有效学习率。

---

## Adam：动量 + 自适应 = 两者之长

Adam（Adaptive Moment Estimation）结合动量（一阶矩）和 RMSprop（二阶矩）：

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

**解读：**
- $\hat{m}_t$ 提供带惯性的方向（动量）
- $1/\sqrt{\hat{v}_t}$ 提供逐参数的自适应步长缩放
- $\epsilon = 10^{-8}$ 防止除零

### 本项目的超参数

| 参数 | 值 | 源码位置 |
|------|-----|---------|
| $\eta$ (lr) | 0.001 | `config/default_params.py:63` |
| $\beta_1$ | 0.9 | PyTorch 默认 |
| $\beta_2$ | 0.999 | PyTorch 默认 |
| $\epsilon$ | 10⁻⁸ | PyTorch 默认 |

---

## Weight Decay（L2 正则化）

在损失函数中附加参数范数惩罚：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \frac{1}{2}\sum \theta_i^2$$

梯度变为：

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \theta} = g_t + \lambda \cdot \theta$$

Adam 的 weight decay 实现（[src/train/optimizer.py:29-33](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/optimizer.py#L29-L33)，`weight_decay=1e-4`）将 $\lambda\theta$ 项从梯度中解耦（AdamW 风格），效果等价但优化轨迹更好。

**作用：** 鼓励所有权重趋于小值，防止模型过度依赖少数强特征，提升泛化能力。

---

## 学习率调度：ReduceLROnPlateau

不是调优化器，而是调学习率。当验证损失不再下降时，减半学习率：

$$\eta_{\text{new}} = \eta_{\text{old}} \times 0.5$$

触发条件：验证损失在连续 `patience=3` 个 epoch 内没有改善（[src/train/optimizer.py:51-86](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/optimizer.py#L51-L86)）。

**为什么需要？** 训练后期大学习率会在最优解附近震荡跳过去。逐渐缩小学习率让优化在小邻域内精细搜素。

**下限保护：** `min_lr=1e-6`，防止学习率过小导致训练停滞。

**示例轨迹：**

```
Epoch  1: lr = 0.001
Epoch  5: val_loss 连续 3 epoch 不降 → lr = 0.0005
Epoch 10: val_loss 连续 3 epoch 不降 → lr = 0.00025
Epoch 14: val_loss 连续 3 epoch 不降 → lr = 0.000125
...
Epoch 20: lr 降至 1e-6 后不再降低
```

---

## SGD vs Adam 对比

| | SGD+Momentum | Adam |
|---|---|---|
| 学习率 | 全局统一，需手工调参 | 逐参数自适应 |
| 收敛速度 | 慢 | 快（特别是稀疏梯度场景） |
| 泛化能力 | 可能更好（隐式正则） | 通常在训练集上更好 |
| 适用场景 | 大 batch、CV 竞赛 | 快速原型、中小规模 |
| 额外内存 | 1× (动量) | 2× (一阶 + 二阶矩) |

对于 MNIST 这种相对简单的任务，Adam 可以在 5-10 个 epoch 内达到 99%+ 准确率，远快于 SGD。

---

## 源码位置

- Adam 创建：`src/train/optimizer.py:19-48`
- ReduceLROnPlateau 创建：`src/train/optimizer.py:51-86`
- 超参数默认值：`config/default_params.py:58-75`
- 训练循环中调用 scheduler.step()：`scripts/train.py:149`
