# 评估系统

## 评估流水线概览

ALL-CNN 的评估系统（[cnnlib/evaluation/](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/evaluation/)）执行：

```
1. 加载 test DataLoader          — build_dataloaders(augment=False)
2. 从检查点加载 model             — loadCheckpoint(optimizer=None)
3. 收集预测                       — gatherPredictions()
4. 计算指标                       — accuracy + confusionMatrix + per-class report
5. 格式化报告打印                  — formatReport()
6. 生成可视化图表                  — generateAllCharts()
7. 保存 eval_results.json
```

---

## Evaluator 类

**源码**: [cnnlib/evaluation/evaluator.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/evaluation/evaluator.py)

```python
class Evaluator:
    def __init__(self, model, loader, criterion=None, device="cpu"):
        ...

    def evaluate(self) -> Dict:
        """
        Returns:
            {
                "loss": float,           # 仅 criterion 非 None
                "accuracy": float,
                "top5_accuracy": float,
                "confusion_matrix": Tensor (num_classes × num_classes),
                "report": {              # per-class metrics
                    "precision": List[float],
                    "recall": List[float],
                    "f1": List[float],
                    "support": List[int],
                    "macro_precision": float,
                    "macro_recall": float,
                    "macro_f1": float,
                },
                "predictions": Tensor,
                "probabilities": Tensor,
                "labels": Tensor,
                "num_samples": int,
            }
        """
```

---

## 收集预测：gatherPredictions

**源码**: [cnnlib/evaluation/metrics.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/evaluation/metrics.py)

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
            torch.cat(allProbs))
```

- `@torch.no_grad()` 禁用梯度计算，节省显存
- 结果全部收集到 CPU，后续计算不需要 GPU
- 同时收集概率以供置信度分析

---

## 混淆矩阵：computeConfusionMatrix

使用 `torch.histogramdd` 向量化计算：

$$\text{CM}[i, j] = \#\{\text{样本}: \text{真实类别} = i, \text{预测类别} = j\}$$

矩阵对角线为正确分类数；非对角线为各类混淆情况。

---

## 分类报告：classificationReport

从混淆矩阵计算 per-class 指标：

对每个类别 $c$：

$$\text{TP}_c = \text{CM}[c, c]$$

$$\text{FP}_c = \sum_{i \neq c} \text{CM}[i, c]$$

$$\text{FN}_c = \sum_{j \neq c} \text{CM}[c, j]$$

$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$

$$\text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

$$\text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### 宏平均 (Macro Average)

简单平均所有类别的指标（不受类别样本数影响）：

$$\text{Macro F1} = \frac{1}{K} \sum_{c=0}^{K-1} \text{F1}_c$$

### Top-K 准确率

$$\text{Top-K Acc} = \frac{\#\{\text{真实类别在预测的 top-K 中}\}}{\text{总样本数}}$$

---

## 格式化输出

输出类似于 sklearn 风格的报告：

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
Top-5 Acc: 0.9997
```

---

## 可视化

**源码**: [cnnlib/evaluation/visualize.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/evaluation/visualize.py)

### 生成的图表

| 图表 | 函数 | 内容 |
|------|------|------|
| 训练曲线 | `plotTrainingCurves()` | 双子图：loss + accuracy 曲线 |
| 混淆矩阵 | `plotConfusionMatrix()` | 热力图，每格标注归一化比例 |
| 错误样本 | `plotErrorGrid()` | 分类错误的样本网格 |
| 预测示例 | `plotPredictionDemo()` | 随机 20 张图预测结果 |

### 批量生成：generateAllCharts

```python
def generateAllCharts(model, loader, datasetInfo, saveDir,
                       history=None, device="cpu", titlePrefix=""):
    # 一键生成全部图表并保存到 saveDir
```

### 反归一化

可视化时需要将标准化后的张量还原为可视图像：

$$\text{image} = \text{tensor} \times \text{std} + \text{mean}$$

不同数据集使用各自的 mean/std（从注册表读取）。

---

## 评估命令示例

```bash
# 基本评估
python main.py --model lenet --dataset mnist eval \
    --checkpoint checkpoints/best_model.pth

# 跳过可视化（只输出文本报告）
python main.py --model vgg16 --dataset cifar10 eval \
    --checkpoint checkpoints/best_model.pth --no-visualize
```

---

## 相关文档

- [评估与指标](/architecture/evaluation) — 本文
- [损失函数](/math/loss-function) — CrossEntropy 推导
- [Softmax](/math/softmax) — logits → 概率
- [训练流程](/architecture/training) — 如何训练得到 checkpoint
- [基准测试系统](/architecture/benchmark) — 自动化评测
