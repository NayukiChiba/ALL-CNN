"""
模型注册表

1. 登记: 装饰器注册模型类 + 元信息
2. 查询: 列出所有已经注册的模型, 获取单个模型的信息
3. 创建: 根据名称实例化模型

用法:
    from cnnlib.registry.models import registerModel, listModels, createModel

    @registerModel("lenet", inputSize=32, channels=1, description="LeNet-5 (1998)")
    class LeNet(nn.Module):
        ...

    # 列出所有模型
    names = listModels()

    # 创建模型
    model = createModel("lenet", numClasses=10)

"""

from typing import Any, Dict, List, Type

import torch.nn as nn

# 全局注册表, key 为模型名称, value为 {class, input_size, channels, description}
_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_model(name: str, input_size: int, channels: int, description: str = ""):
    """
    装饰器：将模型类注册到全局注册表

    Args:
        name:        模型简称，如 "lenet", "alexnet", "vgg16"
        inputSize:   模型要求的输入尺寸（正方形），如 32, 224
        channels:    模型要求的输入通道数，1=灰度 3=RGB
        description: 一行中文描述

    示例:
        @registerModel("lenet", inputSize=32, channels=1, description="LeNet-5 (1998)")
        class LeNet(nn.Module):
            ..

    """

    def wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
        _REGISTRY[name] = {
            "class": cls,
            "input_size": input_size,
            "channels": channels,
            "description": description,
        }
        return cls

    return wrapper


def list_models() -> List[str]:
    """返回所有已经注册的模型的名称列表"""
    return list(_REGISTRY)


def get_model_info(name: str) -> List[str]:
    """获取模型的元信息(不实例化)

    Returns:
        {"class": cls, "inputSize": int, "channels": int, "description": str}

    """
    if name not in _REGISTRY:
        available = ",".join(_REGISTRY.keys()) or "(None)"
        raise KeyError(f"Not Found :{name}, Available: {available}")
    return _REGISTRY[name]


def create_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    根据名称创建模型实例

    Args:
        name(str): 模型名称，如 "lenet"
        num_classes: 输出类别数
        **kwargs: 传递给模型构造函数的额外参数

    Returns:
        nn.Module: 模型实例
    """
    info = get_model_info(name)
    return info["class"](num_classes=num_classes, **kwargs)
