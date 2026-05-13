# Softmax 与数值稳定性

## Softmax 函数定义

Softmax 将任意实数向量 $\mathbf{z} = (z_0, z_1, \dots, z_{K-1})$ 映射为概率分布：

$$p_i = \text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=0}^{K-1} e^{z_j}}, \quad i = 0, \dots, K-1$$

### 性质

1. **正性**: $p_i > 0$（指数函数恒正）
2. **归一化**: $\sum_{i=0}^{K-1} p_i = 1$（分母恰好是分子的和）
3. **单调保序**: 如果 $z_i > z_j$，则 $p_i > p_j$——Softmax 不改变相对排序
4. **平移不变性 (argmax)**: $\text{argmax}(z + c) = \text{argmax}(z)$，但 $\text{softmax}(z + c) \neq \text{softmax}(z)$

---

## 数值稳定性问题

直接计算 $\frac{e^{z_i}}{\sum e^{z_j}}$ 在数值上危险——当 $z_i$ 较大时 $e^{z_i}$ 溢出：

```python
# 数值危险的做法
z = [1000, 1000, 1000]
exp_z = [exp(1000), exp(1000), exp(1000)]  # → inf!
softmax(z) → nan
```

## Log-Sum-Exp 技巧

将分子分母同时除以 $e^{\max(\mathbf{z})}$：

$$p_i = \frac{e^{z_i - \alpha}}{\sum_j e^{z_j - \alpha}}, \quad \alpha = \max_j z_j$$

其中 $\alpha = \max(z_0, \dots, z_{K-1})$。

这样 $z_j - \alpha \leq 0$（因为 $\alpha$ 是最大值），$e^{z_j - \alpha} \leq 1$，永远不会溢出。

---

## LogSoftmax

通常我们只需要 $\log p_i$ 而非 $p_i$：

$$\log p_i = \log\left(\frac{e^{z_i}}{\sum_j e^{z_j}}\right) = z_i - \log\sum_j e^{z_j}$$

其中 $\log\sum_j e^{z_j}$ 称为 Log-Sum-Exp (LSE)：

$$\text{LSE}(\mathbf{z}) = \alpha + \log\sum_j e^{z_j - \alpha}, \quad \alpha = \max_j z_j$$

完整的 LogSoftmax：

$$\log p_i = z_i - \left(\max_j z_j + \log\sum_j e^{z_j - \max_j z_j}\right)$$

---

## 与 CrossEntropyLoss 的关系

PyTorch 的 `nn.CrossEntropyLoss` 将 LogSoftmax 和 NLLLoss 合并为一个算子的原因：

1. **数值稳定性**: 使用 Log-Sum-Exp 技巧避免溢出
2. **计算效率**: 合并算子避免了中间结果的存储和拷贝
3. **梯度融合**: 可以直接计算更精确的合并梯度

在项目中（[cnnlib/training/loss.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/loss.py)），模型输出原始 logits，`CrossEntropyLoss` 内部完成 softmax。

---

## 温度 Softmax (Temperature Scaling)

引入温度参数 $T$ 控制概率分布的"尖锐程度"：

$$p_i^{(T)} = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

| $T$ 值 | 效果 | 应用 |
|--------|------|------|
| $T \to 0$ | 分布趋近于 one-hot（argmax） | 不可导，无法用于训练 |
| $T = 1$ | 标准 Softmax | 常规分类训练 |
| $T > 1$ | 分布趋于均匀（平滑） | 知识蒸馏（从老师模型学软标签） |
| $T \to \infty$ | 均匀分布（$p_i = 1/K$） | 无信息 |

温度 Softmax 广泛用于知识蒸馏——老师模型使用 $T > 1$ 产生"软标签"，学生模型学习这些更丰富的概率分布而非硬标签。

本项目推理时使用标准 Softmax（$T=1$），不涉及温度缩放。

---

## 推理中的使用

在 [Predictor](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/inference/predictor.py) 中：

```python
logits = model(tensor)             # (1, num_classes)
probabilities = F.softmax(logits, dim=1)  # logits → 概率
```

训练时 CrossEntropyLoss 内部处理 softmax，推理时需显式调用 `F.softmax` 获得概率输出。

---

## 相关文档

- [损失函数推导](/math/loss-function) — CrossEntropy = LogSoftmax + NLLLoss
- [推理系统](/architecture/inference) — Predictor 中的 Softmax 后处理
