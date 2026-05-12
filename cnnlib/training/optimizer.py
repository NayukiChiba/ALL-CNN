"""
优化器工厂

根据名称创建优化器实例,统一管理模型参数传递和默认值.

用法:
    from cnnlib.training.optimizer import createOptimizer

    opt = createOptimizer(model, "adamw", lr=0.001, weight_decay=1e-4)
"""

from typing import Any, Dict

import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, RMSprop


def _buildAdam(model: nn.Module, lr: float, weightDecay: float, **kw: Any) -> Adam:
    betas = kw.pop("betas", (0.9, 0.999))
    return Adam(model.parameters(), lr=lr, weight_decay=weightDecay, betas=betas, **kw)


def _buildAdamW(model: nn.Module, lr: float, weightDecay: float, **kw: Any) -> AdamW:
    betas = kw.pop("betas", (0.9, 0.999))
    return AdamW(model.parameters(), lr=lr, weight_decay=weightDecay, betas=betas, **kw)


def _buildSGD(model: nn.Module, lr: float, weightDecay: float, **kw: Any) -> SGD:
    momentum = kw.pop("momentum", 0.9)
    return SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay, **kw
    )


def _buildRMSprop(
    model: nn.Module, lr: float, weightDecay: float, **kw: Any
) -> RMSprop:
    return RMSprop(model.parameters(), lr=lr, weight_decay=weightDecay, **kw)


_OPTIMIZER_FACTORIES: Dict[str, Any] = {
    "adam": _buildAdam,
    "adamw": _buildAdamW,
    "sgd": _buildSGD,
    "rmsprop": _buildRMSprop,
}


def createOptimizer(
    model: nn.Module,
    name: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs: Any,
):
    """
    创建优化器

    Args:
        model:        模型实例
        name:         优化器名称: adam / adamw / sgd / rmsprop
        lr:           初始学习率
        weight_decay: 权重衰减(L2 正则)
        **kwargs:     透传参数
                      adam/adamw: betas=(0.9, 0.999)
                      sgd: momentum=0.9

    Returns:
        torch.optim.Optimizer 实例
    """
    name = name.lower()
    if name not in _OPTIMIZER_FACTORIES:
        available = ", ".join(_OPTIMIZER_FACTORIES.keys())
        raise ValueError(f"未知优化器: '{name}',可用: {available}")
    return _OPTIMIZER_FACTORIES[name](model, lr=lr, weightDecay=weight_decay, **kwargs)
