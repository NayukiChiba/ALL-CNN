"""
Checkpoint 管理

保存/加载训练检查点,包含模型权重、优化器状态、调度器状态和训练元信息.

用法:
    from cnnlib.training.checkpoint import saveCheckpoint, loadCheckpoint

    saveCheckpoint(path, model, optimizer, scheduler, epoch, bestMetric, metrics)
    ckpt = loadCheckpoint(path, model, optimizer, scheduler)
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from config.defaults import DefaultParams


def saveCheckpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    bestMetric: float,
    metrics: Dict[str, Any],
    scheduler: Optional[_LRScheduler] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    保存检查点

    Args:
        path:        保存路径
        model:       模型实例
        optimizer:   优化器实例
        epoch:       当前轮次
        bestMetric:  历史最佳验证指标
        metrics:     当前轮指标字典 {"train_loss": ..., "val_acc": ...}
        scheduler:   调度器实例(可选)
        metadata:    额外元信息(可选)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": bestMetric,
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, str(path))


def loadCheckpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: str = DefaultParams.DEVICE,
) -> Dict[str, Any]:
    """
    加载检查点

    Args:
        path:      检查点文件路径
        model:     模型实例(会被原地更新权重)
        optimizer: 优化器实例(可选,原地更新状态)
        scheduler: 调度器实例(可选,原地更新状态)
        device:    目标设备

    Returns:
        完整的 checkpoint 字典
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"检查点不存在: {path}")

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
