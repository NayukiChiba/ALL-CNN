"""
学习率调度器工厂

支持 ReduceLROnPlateau / StepLR / CosineAnnealingLR / CosineAnnealingWarmRestarts.

用法:
    from cnnlib.training.scheduler import createScheduler

    sched = createScheduler(optimizer, "plateau", factor=0.5, patience=3)
"""

from typing import Any, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
)


def _buildPlateau(optimizer: Optimizer, **kw: Any) -> ReduceLROnPlateau:
    factor = kw.pop("factor", 0.5)
    patience = kw.pop("patience", 3)
    min_lr = kw.pop("min_lr", 1e-6)
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        **kw,
    )


def _buildStep(optimizer: Optimizer, **kw: Any) -> StepLR:
    stepSize = kw.pop("step_size", 10)
    gamma = kw.pop("gamma", 0.1)
    return StepLR(optimizer, step_size=stepSize, gamma=gamma, **kw)


def _buildCosine(optimizer: Optimizer, **kw: Any) -> CosineAnnealingLR:
    tMax = kw.pop("T_max", 50)
    return CosineAnnealingLR(optimizer, T_max=tMax, **kw)


def _buildCosineWarm(optimizer: Optimizer, **kw: Any) -> CosineAnnealingWarmRestarts:
    t0 = kw.pop("T_0", 10)
    tMult = kw.pop("T_mult", 1)
    return CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=tMult, **kw)


_SCHEDULER_FACTORIES: Dict[str, Any] = {
    "plateau": _buildPlateau,
    "step": _buildStep,
    "cosine": _buildCosine,
    "cosine_warm": _buildCosineWarm,
}


def createScheduler(
    optimizer: Optimizer,
    name: str = "plateau",
    **kwargs: Any,
):
    """
    创建学习率调度器

    Args:
        optimizer: 优化器实例
        name:      调度器名称
                   plateau    — ReduceLROnPlateau(监控 loss,需要传 loss 值)
                   step       — 固定步长衰减
                   cosine     — 余弦退火
                   cosine_warm — 余弦退火 + 热重启
        **kwargs:  透传参数
                   plateau: factor=0.5, patience=3, min_lr=1e-6
                   step: step_size=10, gamma=0.1
                   cosine: T_max=50
                   cosine_warm: T_0=10, T_mult=1

    Returns:
        _LRScheduler 或 ReduceLROnPlateau 实例
    """
    name = name.lower()
    if name not in _SCHEDULER_FACTORIES:
        available = ", ".join(_SCHEDULER_FACTORIES.keys())
        raise ValueError(f"未知调度器: '{name}',可用: {available}")
    return _SCHEDULER_FACTORIES[name](optimizer, **kwargs)
