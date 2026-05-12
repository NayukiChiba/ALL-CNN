"""
损失函数工厂

根据名称创建损失函数实例,所有参数显式透传.

用法:
    from cnnlib.training.loss import createLoss

    lossFn = createLoss("cross_entropy", label_smoothing=0.1)
"""

from typing import Literal

import torch.nn as nn

_LOSS_FACTORIES = {
    "cross_entropy": lambda **kw: nn.CrossEntropyLoss(**kw),
    "mse": lambda **kw: nn.MSELoss(**kw),
}


def createLoss(
    name: Literal["cross_entropy", "mse"] = "cross_entropy", **kwargs
) -> nn.Module:
    """
    创建损失函数

    Args:
        name: 损失函数名称,支持 cross_entropy / mse
        **kwargs: 透传给损失函数构造函数
                  cross_entropy: label_smoothing, weight, reduction 等
                  mse: reduction 等

    Returns:
        nn.Module 损失函数实例
    """
    name = name.lower()
    if name not in _LOSS_FACTORIES:
        available = ", ".join(_LOSS_FACTORIES.keys())
        raise ValueError(f"未知损失函数: '{name}',可用: {available}")
    return _LOSS_FACTORIES[name](**kwargs)
