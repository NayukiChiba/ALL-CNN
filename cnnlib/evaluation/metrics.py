"""
评估指标

分类任务常用指标:准确率、Top-K 准确率、混淆矩阵、精确率/召回率/F1.

用法:
    from cnnlib.evaluation.metrics import computeAllMetrics

    metrics = computeAllMetrics(outputs, labels, numClasses=10)
    # → {"accuracy": 0.93, "top5_accuracy": 0.99, "confusion_matrix": ..., ...}
"""

from typing import Dict

import torch


def computeAccuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 准确率"""
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    return correct / labels.size(0)


def computeTopKAccuracy(
    outputs: torch.Tensor, labels: torch.Tensor, k: int = 5
) -> float:
    """Top-K 准确率"""
    _, topkIndices = outputs.topk(k, dim=1)
    correct = topkIndices.eq(labels.view(-1, 1)).any(dim=1).sum().item()
    return correct / labels.size(0)


def computeConfusionMatrix(
    outputs: torch.Tensor, labels: torch.Tensor, numClasses: int
) -> torch.Tensor:
    """
    计算混淆矩阵 (numClasses x numClasses)

    行=真实标签,列=预测标签
    """
    _, predicted = outputs.max(1)
    cm = torch.zeros(numClasses, numClasses, dtype=torch.long)
    for t, p in zip(labels.cpu(), predicted.cpu()):
        cm[t.item(), p.item()] += 1
    return cm


def computePerClassAccuracy(cm: torch.Tensor) -> Dict[int, float]:
    """从混淆矩阵计算每类准确率"""
    perClass = {}
    for i in range(cm.size(0)):
        total = cm[i].sum().item()
        perClass[i] = cm[i, i].item() / total if total > 0 else 0.0
    return perClass


def computePrecisionRecallF1(cm: torch.Tensor) -> Dict[str, float]:
    """
    从混淆矩阵计算宏观/微观精确率、召回率、F1

    Returns:
        {"macro_precision", "macro_recall", "macro_f1",
         "micro_precision", "micro_recall", "micro_f1"}
    """
    tpPerClass = cm.diag().float()
    predPerClass = cm.sum(dim=0).float()  # 每列=预测为该类的总数
    truePerClass = cm.sum(dim=1).float()  # 每行=真实为该类的总数

    # 微观:全局 TP / 全局总数
    microTp = tpPerClass.sum().item()
    microPred = predPerClass.sum().item()
    microTrue = truePerClass.sum().item()

    microPrecision = microTp / microPred if microPred > 0 else 0.0
    microRecall = microTp / microTrue if microTrue > 0 else 0.0
    microF1 = (
        2 * microPrecision * microRecall / (microPrecision + microRecall)
        if (microPrecision + microRecall) > 0
        else 0.0
    )

    # 宏观:各类指标取平均
    eps = 1e-8
    perPrecision = tpPerClass / (predPerClass + eps)
    perRecall = tpPerClass / (truePerClass + eps)
    perF1 = 2 * perPrecision * perRecall / (perPrecision + perRecall + eps)

    return {
        "macro_precision": perPrecision.mean().item(),
        "macro_recall": perRecall.mean().item(),
        "macro_f1": perF1.mean().item(),
        "micro_precision": microPrecision,
        "micro_recall": microRecall,
        "micro_f1": microF1,
    }


def computeAllMetrics(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    numClasses: int,
    topK: int = 5,
) -> Dict:
    """
    一次性计算所有分类指标

    Args:
        outputs:    模型输出 logits (N, numClasses)
        labels:     真实标签 (N,)
        numClasses: 类别总数
        topK:       Top-K 的 K 值

    Returns:
        {"accuracy", "top{k}_accuracy", "confusion_matrix",
         "per_class_accuracy", "macro_precision", "macro_recall", "macro_f1",
         "micro_precision", "micro_recall", "micro_f1"}
    """
    cm = computeConfusionMatrix(outputs, labels, numClasses)
    perClassAcc = computePerClassAccuracy(cm)
    prf = computePrecisionRecallF1(cm)

    result = {
        "accuracy": computeAccuracy(outputs, labels),
        f"top{topK}_accuracy": computeTopKAccuracy(outputs, labels, k=topK),
        "confusion_matrix": cm,
        "per_class_accuracy": perClassAcc,
    }
    result.update(prf)
    return result
