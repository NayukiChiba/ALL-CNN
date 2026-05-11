"""
src/model/factory.py

模型工厂 — 根据模型名称创建对应架构实例。

支持的模型：
    - cnn        → MNISTCNN（ConvBlock 堆叠式 CNN）
    - lenet      → LeNet-5（经典 1998 架构）
    - (后续扩展 alexnet, vgg, nin, googlenet)

Usage:
    from src.model.factory import createModel
    model = createModel("cnn", numClasses=10, device="cuda")
    model = createModel("lenet", numClasses=10, device="cuda")
"""

import torch.nn as nn

from config.default_params import DefaultParams
from src.model.cnn import MNISTCNN
from src.model.lenet import LeNet


def createModel(
    modelName: str = "cnn",
    numClasses: int = 10,
    conv_channels: list[int] = None,
    hidden_size: int = 128,
    dropout: float = 0.5,
    device: str = DefaultParams.DEVICE,
) -> nn.Module:
    """
    根据模型名称创建架构实例并迁移到指定设备。

    Args:
        modelName:     "cnn" 或 "lenet"
        numClasses:    分类类别数
        conv_channels: (仅 cnn) 卷积层输出通道列表
        hidden_size:   (仅 cnn) 隐藏层大小
        dropout:       (仅 cnn) Dropout 概率
        device:        "cpu" 或 "cuda"

    Returns:
        指定模型实例，已在目标设备上。
    """
    modelName = modelName.lower()

    if modelName == "lenet":
        model = LeNet(numClasses=numClasses)
    elif modelName == "cnn":
        model = MNISTCNN(
            conv_channels=conv_channels,
            fc_hidden_size=hidden_size,
            dropout=dropout,
        )
    else:
        raise ValueError(f"未知模型: {modelName}，支持: cnn, lenet")

    model = model.to(device)
    return model
