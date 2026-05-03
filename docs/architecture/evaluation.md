# 评估系统

## 评估流水线概览

[scripts/eval.py:27-132](https://github.com/NayukiChiba/MNIST-CNN/blob/main/scripts/eval.py#L27-L132) 中的评估脚本执行：

```
1. 加载 test DataLoader
2. 从检查点加载 model（仅权重，不含 optimizer）
3. evaluateModel() → 损失/准确率/混淆矩阵/分类报告
4. 格式化报告打印
5. 保存 eval_results.json
6. 生成可视化图表
```

---

## 收集预测：gatherPredictions

[gatherPredictions()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/metrics.py#L32-L106) 是整个评估的基础：

```python
@torch.no_grad()
def gatherPredictions(model, dataloader, device):
    model.eval()
    allLabels, allPredictions, allProbs = [], [], []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)        # logits → 概率
        _, predicted = logits.max(dim=1)

        allLabels.append(labels)
        allPredictions.append(predicted.cpu())
        allProbs.append(probs.cpu())

    return (torch.cat(allLabels),
            torch.cat(allPredictions),
            torch.cat(allProbs))               # 全部在 CPU 上
```

- `@torch.no_grad()` 确保不计算梯度，节省显存
- 结果全部收集到 CPU，后续计算不需要 GPU
- 除预测类别外也收集概率，便于后续分析（如置信度分布）

---

## 混淆矩阵：computeConfusionMatrix

[computeConfusionMatrix()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/metrics.py#L109-L146)：

```python
def computeConfusionMatrix(labels, predictions, num_classes=10):
    # 向量化：使用 torch.histogramdd，无显式循环
    indices = torch.stack([labels, predictions], dim=1)
    bins = [num_classes, num_classes]
    cm = torch.histogramdd(indices.float(), bins=bins).hist
    return cm.long()
```

$$\text{CM}[i, j] = \#\{\text{样本}: \text{真实类别} = i, \text{预测类别} = j\}$$

矩阵对角线上的值表示正确分类的样本数。

**实现技巧：** 使用 `torch.histogramdd` 而非 Python 循环。10 个类别的双重 for 循环是 100 次迭代；histogramdd 是一次 C 级别的向量化运算。

---

## 分类报告：classificationReport

[classificationReport()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/metrics.py#L149-L251) 从混淆矩阵计算 per-class 指标：

对每个类别 $c \in \{0, \dots, 9\}$：

$$\text{TP}_c = \text{CM}[c, c]$$
$$\text{FP}_c = \sum_{i \neq c} \text{CM}[i, c]$$
$$\text{FN}_c = \sum_{j \neq c} \text{CM}[c, j]$$
$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$
$$\text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
$$\text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

**宏平均（Macro Average）：** 简单平均所有类别的指标（不受各类别样本数影响）：

$$\text{Macro F1} = \frac{1}{10} \sum_{c=0}^{9} \text{F1}_c$$

**边角情况处理：** `precision_c` 和 `recall_c` 的分母可能为 0（某类别从未被预测或从未出现在数据中）。此时返回 0.0（[metrics.py:199-203](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/metrics.py#L199-L203)）。

### 格式化输出

[formatReport()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/metrics.py#L371-L410) 产出类似于 sklearn 风格的格式化字符串：

```
Classification Report
==================================================
Class    Precision    Recall    F1-score    Support
--------------------------------------------------
0        0.9921       0.9910    0.9915       980
1        0.9939       0.9950    0.9944      1135
...
--------------------------------------------------
Macro Avg  0.9912    0.9910    0.9911      10000
Accuracy: 0.9910
```

---

## 一键评估：evaluateModel

[evaluateModel()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/metrics.py#L259-L363) 将以上步骤打包为一次调用：

```python
def evaluateModel(model, dataloader, criterion=None, device=None):
    # 1. 收集预测
    labels, predictions, probabilities = gatherPredictions(...)

    # 2. 混淆矩阵
    cm = computeConfusionMatrix(labels, predictions)

    # 3. Per-class 报告
    report = classificationReport(labels, predictions)

    # 4. 准确率
    accuracy = (predictions == labels).float().mean().item()

    return {
        "loss": loss,           # 仅当提供 criterion 时计算
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "num_samples": len(labels),
    }
```

---

## 可视化

所有可视化函数位于 [src/eval/visualize.py](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/visualize.py)。

### 训练曲线：plotTrainingCurves

[visualize.py:65-184](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/visualize.py#L65-L184)

双子图布局：左图显示 train + val 损失曲线，右图显示 train + val 准确率曲线。最佳 epoch 用虚线标注：

![训练曲线](/visualizations/training_curves.png)

### 混淆矩阵热力图：plotConfusionMatrix

[visualize.py:192-320](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/visualize.py#L192-L320)

用 `imshow` 渲染混淆矩阵热力图，每个格子标注归一化后的比例：

![混淆矩阵](/visualizations/confusion_matrix.png)

### 错误样本网格：plotErrorGrid

[visualize.py:328-467](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/visualize.py#L328-L467)

展示模型分类错误的样本，红色标题标注真实类别与错误预测：

![错误样本](/visualizations/error_grid.png)

### 预测示例

从测试集随机抽取 20 张图展示预测结果，绿色为正确、红色为错误：

![预测示例](/visualizations/prediction_demo.png)

**反归一化**（[visualize.py:431-436](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/visualize.py#L431-L436)）：

$$\text{image} = \text{tensor} \times 0.3081 + 0.1307$$

### 收集错误样本：gatherErrorSamples

[visualize.py:475-549](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/visualize.py#L475-L549) 遍历 DataLoader 收集最多 `max_errors` 个错误分类的样本及其真实/预测标签。

### 批量生成：generateEvaluationPlots

[visualize.py:557-636](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/eval/visualize.py#L557-L636) 一键生成全部图表并保存到 `visualizations/`。

---

## 评估命令示例

```bash
# 基本评估
python main.py eval --checkpoint checkpoints/best_model.pth

# 跳过可视化（只输出文本报告）
python main.py eval --checkpoint checkpoints/best_model.pth --no-visualize
```
