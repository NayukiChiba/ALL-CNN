# cnnlib/registry/models.py

"""
模型注册表

只做三件事：
    1. 登记：装饰器注册模型类 + 元信息
    2. 查询：列出所有已注册模型、获取单个模型信息
    3. 创建：根据名称实例化模型

用法：
    from cnnlib.registry.models import register_model, list_models, create_model

    @register_model("lenet", input_size=32, channels=1, description="LeNet-5 (1998)")
    class LeNet(nn.Module):
        ...

    names = list_models()
    model = create_model("lenet", num_classes=10)
"""

from typing import Any, Dict, List, Type

import torch.nn as nn

# key=模型名称, value={class, input_size, channels, description}
_registry: Dict[str, Dict[str, Any]] = {}


def register_model(
    name: str,
    input_size: int,
    channels: int,
    description: str = "",
):
    """
    装饰器：将模型类注册到全局注册表

    Args:
        name:        模型简称，如 "lenet", "alexnet", "vgg16"
        input_size:  模型要求的输入尺寸（正方形），如 32, 224
        channels:    模型要求的输入通道数，1=灰度 3=RGB
        description: 一行描述
    """

    def wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
        _registry[name] = {
            "class": cls,
            "input_size": input_size,
            "channels": channels,
            "description": description,
        }
        return cls

    return wrapper


def list_models() -> List[str]:
    """返回所有已注册模型的名称列表"""
    return list(_registry.keys())


def get_model_info(name: str) -> Dict[str, Any]:
    """
    获取模型元信息（不实例化）

    Returns:
        {"class": cls, "input_size": int, "channels": int, "description": str}
    """
    if name not in _registry:
        available = ", ".join(_registry.keys()) or "(none)"
        raise KeyError(f"Unknown model: '{name}'. Available: {available}")
    return _registry[name]


def create_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    根据名称创建模型实例

    Args:
        name:        模型名称，如 "lenet"
        num_classes: 输出类别数
        **kwargs:    传递给模型构造函数的额外参数
    """
    info = get_model_info(name)
    return info["class"](num_classes=num_classes, **kwargs)


def print_registry():
    """打印所有已注册模型（调试用）"""
    print(f"{'Name':<14} {'Input':<10} {'Ch':<5} {'Description'}")
    print("-" * 60)
    for name, info in _registry.items():
        size = f"{info['input_size']}x{info['input_size']}"
        print(f"  {name:<12} {size:<10} {info['channels']:<5} {info['description']}")
