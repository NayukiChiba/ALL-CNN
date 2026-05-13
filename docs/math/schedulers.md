# 学习率调度器

## 为什么需要学习率调度

训练的不同阶段对学习率的需求不同：
- **早期**: 大学习率快速探索，逼近最优区域
- **后期**: 小学习率精细调优，避免在最优解附近震荡

固定学习率很难兼顾两者——太大后期不收敛，太小早期收敛慢。

---

## 1. ReduceLROnPlateau

**最常用的自适应调度器**。当验证指标停止改善时自动降低学习率。

### 公式

$$\eta_{\text{new}} = \eta_{\text{old}} \times \text{factor}$$

### 触发条件

连续 `patience` 个 epoch 验证指标没有改善（改善幅度超过 `min_delta`）时触发。

### 本项目的默认参数

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",       # 监控 val_loss（越小越好）
    factor=0.5,       # 每次降低到原来的一半
    patience=3,       # 连续 3 个 epoch 不降就降低学习率
    min_lr=1e-6,      # 学习率下限，防止过小导致训练停滞
)
```

### 典型轨迹

```
Epoch  1: lr = 0.001
Epoch  5: val_loss 连续 3 epoch 不降 → lr = 0.0005
Epoch 10: val_loss 连续 3 epoch 不降 → lr = 0.00025
Epoch 14: val_loss 连续 3 epoch 不降 → lr = 0.000125
...
Epoch 20: lr 降至 1e-6 后不再降低
```

<table style="margin:1em auto;font-size:13px;border-collapse:collapse;">
  <thead>
    <tr style="background:#f0f0f0;">
      <th style="padding:4px 12px;border:1px solid #ccc;">Epoch</th>
      <th style="padding:4px 12px;border:1px solid #ccc;">1-4</th>
      <th style="padding:4px 12px;border:1px solid #ccc;">5-7</th>
      <th style="padding:4px 12px;border:1px solid #ccc;">8-10</th>
      <th style="padding:4px 12px;border:1px solid #ccc;">11-13</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align:center;">
      <td style="padding:4px 12px;border:1px solid #ccc;font-weight:500;">lr</td>
      <td style="padding:4px 12px;border:1px solid #ccc;">0.001</td>
      <td style="padding:4px 12px;border:1px solid #ccc;">0.0005</td>
      <td style="padding:4px 12px;border:1px solid #ccc;">0.00025</td>
      <td style="padding:4px 12px;border:1px solid #ccc;">0.000125</td>
    </tr>
  </tbody>
</table>
<div style="text-align:center;font-size:12px;color:#666;margin-top:4px;">ReduceLROnPlateau 学习率阶梯下降 (factor=0.5, patience=3)</div>

### 优点 vs 缺点

| 优点 | 缺点 |
|------|------|
| 自适应——无需预设下降时机 | 被动响应——可能在鞍点浪费多个 epoch |
| 简单直观 | 不保证最优（可能提前进入精细搜索） |

---

## 2. StepLR

按固定 epoch 间隔降低学习率。

### 公式

$$\eta_t = \eta_0 \times \gamma^{\lfloor t / T \rfloor}$$

其中 $T$ 是 `step_size`（每 $T$ 个 epoch 降低一次），$\gamma$ 是衰减系数。

### 默认参数

```python
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
```

即每 10 个 epoch 学习率变为原来的 0.1 倍。

### 典型轨迹

```
Epoch  1-10: lr = 0.001
Epoch 11-20: lr = 0.0001
Epoch 21-30: lr = 0.00001
```

### 优点 vs 缺点

| 优点 | 缺点 |
|------|------|
| 简单、可预测 | 不感知训练状态——可能在不合适的时机降低学习率 |
| 适合使用固定训练 epoch 的场景 | 需要在训练前预设 step_size |

---

## 3. CosineAnnealingLR

以余弦曲线平滑降低学习率。

### 公式

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \cdot \frac{t}{T_{\max}}\right)\right)$$

当 $t = 0$ 时 $\eta = \eta_{\max}$（初始学习率）；当 $t = T_{\max}$ 时 $\eta = \eta_{\min}$。

### 默认参数

```python
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
```

### 为什么余弦曲线？

余弦曲线在两端变化缓慢、中间变化快——这恰好匹配训练的需求：
- 训练初期在较大学习率停留久（广泛探索）
- 训练中期快速下降（收敛到最优区域）
- 训练末期在较小学习率停留久（精细搜索）

### 典型轨迹

```
Epoch  1: lr ≈ 0.001
Epoch 25: lr ≈ 0.0005
Epoch 50: lr ≈ 0
```

---

## 4. CosineAnnealingWarmRestarts

余弦退火的增强版：周期性重启（warm restart），每次重启将学习率恢复到初始值。

### 公式

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \cdot \frac{t \bmod T_i}{T_i}\right)\right)$$

其中 $T_i$ 是第 $i$ 个周期的长度。通常 $T_{i+1} = T_i \times T_{\text{mult}}$（每个周期逐渐变长）。

### 默认参数

```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
```

### 优势

- **逃离局部最优**: 学习率突然增大让优化器有机会跳出局部极小值
- **Snapshot Ensemble**: 每个周期末尾的模型参数可以作为独立模型，取平均或投票得到集成效果
- 在图像分类任务中常比普通余弦退火效果更好

<div style="max-width:520px;margin:1em auto;font-size:13px;">
  <div style="text-align:center;font-weight:600;margin-bottom:8px;">CosineAnnealingWarmRestarts 周期性重启 (T_0=10)</div>
  <p style="text-align:center;color:#555;">余弦曲线从最大值平滑衰减至最小值，每 10 个 epoch 重启回到初始学习率。</p>
  <table style="margin:0 auto;font-size:13px;border-collapse:collapse;">
    <thead>
      <tr style="background:#f0f0f0;">
        <th style="padding:4px 12px;border:1px solid #ccc;">周期</th>
        <th style="padding:4px 12px;border:1px solid #ccc;">Epoch 范围</th>
        <th style="padding:4px 12px;border:1px solid #ccc;">lr 变化</th>
      </tr>
    </thead>
    <tbody>
      <tr style="text-align:center;">
        <td style="padding:4px 12px;border:1px solid #ccc;">第 1 周期</td>
        <td style="padding:4px 12px;border:1px solid #ccc;">0-10</td>
        <td style="padding:4px 12px;border:1px solid #ccc;">0.001 → 0.0001 (余弦下降)</td>
      </tr>
      <tr style="text-align:center;">
        <td style="padding:4px 12px;border:1px solid #ccc;">第 2 周期</td>
        <td style="padding:4px 12px;border:1px solid #ccc;">10-20</td>
        <td style="padding:4px 12px;border:1px solid #ccc;">0.001 → 0.0001 (余弦下降)</td>
      </tr>
      <tr style="text-align:center;">
        <td style="padding:4px 12px;border:1px solid #ccc;">第 3 周期</td>
        <td style="padding:4px 12px;border:1px solid #ccc;">20-30</td>
        <td style="padding:4px 12px;border:1px solid #ccc;">0.001 → 0.0001 (余弦下降)</td>
      </tr>
    </tbody>
  </table>
</div>

---

## 四种调度器对比



| 调度器 | 下降方式 | 自适应 | 需要预设 | 推荐场景 |
|--------|---------|:---:|------|---------|
| ReduceLROnPlateau | 阶梯状 | ✓ | patience | **默认选择**——简单可靠 |
| StepLR | 阶梯状 | ✗ | step_size | 固定 epoch 训练，清晰里程碑 |
| CosineAnnealing | 平滑 | ✗ | T_max | 已知总 epoch，追求平滑收敛 |
| CosineWarmRestarts | 周期性 | ✗ | T_0, T_mult | 追求最优结果，可能集成 |

---

## 本项目的调度器工厂

**源码**: [cnnlib/training/scheduler.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/scheduler.py)

```python
def createScheduler(optimizer, name="plateau", **kwargs):
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                 patience=3, min_lr=1e-6, **kwargs)
    elif name == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1, **kwargs)
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=50, **kwargs)
    elif name == "cosine_warm":
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, **kwargs)
```

CLI 使用: `python main.py train --scheduler cosine --lr-factor 0.5`

---

## 相关文档

- [优化器](/math/optimizer) — Adam/AdamW/SGD/RMSprop 的更新规则
- [训练流程](/architecture/training) — scheduler.step() 在训练循环中的位置
