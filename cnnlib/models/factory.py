"""
模型工厂

根据名称创建模型实例，注册表的唯一消费方。

职责：
    1. 查注册表拿类引用和元信息
    2. 拼出 BaseModel 需要的 input_size / in_channels / num_classes
    3. 实例化并迁移到设备

用法：
    from cnnlib.models.factory import create_model

    model = create_model("lenet", num_classes=10, device="cuda")
    model = create_model("cnn", num_classes=10, conv_channels=[32, 64, 128])
"""

from typing import Any

import torch
import torch.nn as nn

from cnnlib.registry.models import get_model_info


def _get_device(device: str = "auto") -> str:
    """解析设备参数"""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def create_model(
    name: str, num_classes: int = 10, device: str = "auto", **kwargs
) -> nn.Module:
    """
    根据模型名称创建实例

    Args:
        name:        模型名称，如 "lenet", "cnn"
        num_classes: 输出类别数（从数据集注册表获取）
        device:      "cpu", "cuda" 或 "auto"（自动检测）
        **kwargs:    透传给模型构造函数的额外参数
                     - cnn: conv_channels, dropout
                     - alexnet: dropout
                     - vgg: variant ("11"/"13"/"16"/"19"), dropout

    Returns:
        模型实例，已在目标设备上

    示例:
        # 基础用法
        model = create_model("lenet", num_classes=10)

        # 带设备
        model = create_model("vgg16", num_classes=100, device="cuda")

        # 带模型专属参数
        model = create_model("cnn", num_classes=10, conv_channels=[32, 64, 128])
    """
    device = _get_device(device)
    info = get_model_info(name)

    # 注册表的 channels 映射为 BaseModel 的 in_channels
    base_args: dict[str, Any] = {
        "input_size": info["input_size"],
        "in_channels": info["channels"],
        "num_classes": num_classes,
    }

    # 合并基类参数和模型专属参数
    model = info["class"](**base_args, **kwargs)
    model = model.to(device)
    return model


def create_model_for_dataset(
    model_name: str,
    dataset_name: str,
    device: str = "auto",
    **kwargs,
) -> nn.Module:
    """
    快捷方法：从数据集注册表自动取 num_classes 创建模型

    Args:
        model_name:   模型名称，如 "lenet"
        dataset_name: 数据集名称，如 "cifar10"
        device:       "cpu", "cuda" 或 "auto"
        **kwargs:     透传给 create_model

    示例:
        model = create_model_for_dataset("vgg16", "cifar100", device="cuda")
    """
    from cnnlib.registry.datasets import get_dataset_info

    dataset_info = get_dataset_info(dataset_name)
    return create_model(
        name=model_name,
        num_classes=dataset_info["num_classes"],
        device=device,
        **kwargs,
    )


if __name__ == "__main__":
    # 需要先注册模型才能测试
    try:
        model = create_model("lenet", num_classes=10)
        print(model.summary())
    except KeyError as e:
        print(f"[skip] {e}")
