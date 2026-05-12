"""
评估可视化

混淆矩阵图、预测样本展示、训练曲线、逐类准确率对比.

用法:
    from cnnlib.evaluation.visualize import (
        plotConfusionMatrix, plotPredictions, plotTrainingHistory
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.defaults import DefaultParams

matplotlib.use("Agg")  # 非交互后端


def _ensureSave(savePath: Optional[str | Path], defaultName: str) -> Path | None:
    """规范化保存路径"""
    if savePath is None:
        return None
    path = Path(savePath)
    if path.is_dir():
        path = path / defaultName
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plotConfusionMatrix(
    cm: torch.Tensor | np.ndarray,
    classNames: Optional[Sequence[str]] = None,
    savePath: Optional[str | Path] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    绘制混淆矩阵热力图

    Args:
        cm:         混淆矩阵 (numClasses x numClasses)
        classNames: 类别名称列表
        savePath:   保存路径(None=不保存)
        normalize:  是否按行归一化
        title:      图表标题

    Returns:
        matplotlib Figure
    """
    cm = np.asarray(cm).copy()
    numClasses = cm.shape[0]

    if normalize:
        rowSums = cm.sum(axis=1, keepdims=True)
        rowSums[rowSums == 0] = 1
        cm = cm / rowSums

    fig, ax = plt.subplots(figsize=(max(8, numClasses * 0.6), max(6, numClasses * 0.5)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    # 标注数值
    fmt = ".2f" if normalize else "d"
    threshold = cm.max() / 2
    for i in range(numClasses):
        for j in range(numClasses):
            val = cm[i, j]
            ax.text(
                j,
                i,
                format(val, fmt) if val > 0 else "",
                ha="center",
                va="center",
                color="white" if val > threshold else "black",
                fontsize=7,
            )

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if classNames and len(classNames) == numClasses:
        ticks = list(range(numClasses))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(classNames, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(classNames, fontsize=7)

    plt.colorbar(im, ax=ax)
    fig.tight_layout()

    path = _ensureSave(savePath, "confusion_matrix.png")
    if path:
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

    return fig


def plotPredictions(
    model: nn.Module,
    loader: DataLoader,
    mean: Sequence[float],
    std: Sequence[float],
    classNames: Optional[Sequence[str]] = None,
    numSamples: int = 8,
    savePath: Optional[str | Path] = None,
    device: str = DefaultParams.DEVICE,
) -> plt.Figure:
    """
    展示模型预测样本(真实标签 vs 预测标签)

    Args:
        model:      模型
        loader:     数据加载器
        mean:       数据集均值(用于反归一化)
        std:        数据集标准差
        classNames: 类别名称
        numSamples: 展示样本数
        savePath:   保存路径
        device:     计算设备

    Returns:
        matplotlib Figure
    """
    model.eval()

    images, labels = next(iter(loader))
    images = images[:numSamples]
    labels = labels[:numSamples]

    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)

    images = images.cpu()
    preds = preds.cpu()

    mean = np.array(mean)
    std = np.array(std)

    cols = min(4, numSamples)
    rows = (numSamples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(numSamples):
        img = images[i].permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)

        if img.shape[-1] == 1 or (img.ndim == 2):
            axes[i].imshow(img.squeeze(), cmap="gray")
        else:
            axes[i].imshow(img)

        trueLabel = (
            classNames[labels[i].item()] if classNames else str(labels[i].item())
        )
        predLabel = classNames[preds[i].item()] if classNames else str(preds[i].item())
        color = "green" if labels[i] == preds[i] else "red"
        axes[i].set_title(f"T: {trueLabel} | P: {predLabel}", color=color, fontsize=9)
        axes[i].axis("off")

    for i in range(numSamples, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()

    path = _ensureSave(savePath, "predictions.png")
    if path:
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

    return fig


def plotTrainingHistory(
    history: Dict[str, List[float]],
    savePath: Optional[str | Path] = None,
    title: str = "Training History",
) -> plt.Figure:
    """
    绘制训练曲线(loss + accuracy)

    Args:
        history:  {"train_loss": [...], "val_loss": [...],
                   "train_acc": [...], "val_acc": [...]}
        savePath: 保存路径
        title:    图表标题

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    ax1.plot(epochs, history.get("train_loss", []), "b-", label="Train Loss")
    ax1.plot(epochs, history.get("val_loss", []), "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.get("train_acc", []), "b-", label="Train Acc")
    ax2.plot(epochs, history.get("val_acc", []), "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    path = _ensureSave(savePath, "training_history.png")
    if path:
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

    return fig


def plotPerClassAccuracy(
    perClassAcc: Dict[int, float],
    classNames: Optional[Sequence[str]] = None,
    savePath: Optional[str | Path] = None,
    title: str = "Per-Class Accuracy",
) -> plt.Figure:
    """
    绘制逐类准确率柱状图

    Args:
        perClassAcc: {classIdx: accuracy, ...}
        classNames:  类别名称
        savePath:    保存路径
        title:       图表标题

    Returns:
        matplotlib Figure
    """
    indices = sorted(perClassAcc.keys())
    accs = [perClassAcc[i] * 100 for i in indices]
    labels = (
        [classNames[i] for i in indices]
        if classNames and len(classNames) >= len(indices)
        else [str(i) for i in indices]
    )

    fig, ax = plt.subplots(figsize=(max(8, len(indices) * 0.5), 5))
    colors = [
        "#2ecc71" if a >= 70 else "#f39c12" if a >= 40 else "#e74c3c" for a in accs
    ]
    ax.bar(labels, accs, color=colors)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=45, labelsize=7)

    for i, (label, acc) in enumerate(zip(labels, accs)):
        ax.text(i, acc + 1, f"{acc:.1f}", ha="center", fontsize=7)

    fig.tight_layout()

    path = _ensureSave(savePath, "per_class_accuracy.png")
    if path:
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

    return fig


def generateAllCharts(
    model: nn.Module,
    loader: DataLoader,
    datasetInfo: Dict,
    saveDir: str | Path,
    history: Optional[Dict[str, List[float]]] = None,
    classNames: Optional[Sequence[str]] = None,
    device: str = DefaultParams.DEVICE,
    titlePrefix: str = "",
    precomputedMetrics: Optional[Dict] = None,
) -> Dict:
    """
    生成全部评估图表（训练曲线、混淆矩阵、逐类准确率、预测样本）

    Args:
        model:             已训练的模型
        loader:            数据加载器（通常为测试集）
        datasetInfo:       数据集元信息 {"num_classes", "mean", "std", ...}
        saveDir:           保存目录
        history:           训练历史（有则出训练曲线，None 则跳过）
        classNames:        类别名称列表
        device:            计算设备
        titlePrefix:       图表标题前缀
        precomputedMetrics: 预先计算的指标（避免重复评估）

    Returns:
        评估指标字典
    """
    from cnnlib.evaluation.evaluator import Evaluator
    from cnnlib.training import createLoss

    saveDir = Path(saveDir)
    saveDir.mkdir(parents=True, exist_ok=True)

    # 训练曲线
    if history and history.get("train_loss") and history.get("val_loss"):
        plotTrainingHistory(
            history,
            savePath=saveDir / "training_history.png",
            title=f"{titlePrefix}Training History",
        )

    # 评估
    if precomputedMetrics is not None:
        metrics = precomputedMetrics
    else:
        lossFn = createLoss("cross_entropy")
        evaluator = Evaluator(
            model, loader, lossFn, torch.device(device), datasetInfo["num_classes"]
        )
        metrics = evaluator.evaluate()

    # 混淆矩阵
    if "confusion_matrix" in metrics:
        plotConfusionMatrix(
            metrics["confusion_matrix"],
            classNames=classNames,
            savePath=saveDir / "confusion_matrix.png",
            title=f"{titlePrefix}Confusion Matrix",
        )

    # 逐类准确率
    if "per_class_accuracy" in metrics:
        plotPerClassAccuracy(
            metrics["per_class_accuracy"],
            classNames=classNames,
            savePath=saveDir / "per_class_accuracy.png",
            title=f"{titlePrefix}Per-Class Accuracy",
        )

    # 预测样本
    plotPredictions(
        model,
        loader,
        mean=datasetInfo["mean"],
        std=datasetInfo["std"],
        classNames=classNames,
        savePath=saveDir / "predictions.png",
        device=device,
    )

    return metrics
