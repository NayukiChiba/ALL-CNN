"""
src/model/factory.py

Model factory — builds MNISTCNN from settings and moves it to device.

Usage:
    from src.model.factory import createModel
    model = createModel(conv_channels=[32, 64], hidden_size=128, dropout=0.5, device="cuda")
"""

import torch.nn as nn

from src.model.cnn import MNISTCNN


def createModel(
    conv_channels: list[int] = None,
    hidden_size: int = 128,
    dropout: float = 0.5,
    device: str = "cpu",
) -> nn.Module:
    """
    Build MNISTCNN and move to the specified device.

    Args:
        conv_channels: Output channels per ConvBlock. Default [32, 64].
        hidden_size:   Hidden units in the classifier head. Default 128.
        dropout:       Dropout probability. Default 0.5.
        device:        "cpu" or "cuda". The model is moved here before returning.

    Returns:
        MNISTCNN instance on the target device, ready for training or inference.
    """
    model = MNISTCNN(
        conv_channels=conv_channels,
        hidden_size=hidden_size,
        dropout=dropout,
    )
    model = model.to(device)
    return model
