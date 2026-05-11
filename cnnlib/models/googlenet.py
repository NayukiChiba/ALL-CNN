"""
GoogLeNet — Inception v1

Szegedy et al., "Going Deeper with Convolutions", CVPR 2015.

架构:
    Input (3, 224, 224)
      → Conv(7x7, stride=2) + BN + ReLU → MaxPool(3x3, stride=2)
      → Conv(1x1) + BN + ReLU
      → Conv(3x3) + BN + ReLU → MaxPool(3x3, stride=2)
      → inception(3a) → inception(3b) → MaxPool
      → inception(4a) → inception(4b) → inception(4c) → inception(4d) → inception(4e) → MaxPool
      → inception(5a) → inception(5b)
      → AdaptiveAvgPool2d(1) → Dropout(0.4) → Linear(1024→num_classes)

特点:
    - Inception 多分支并行，不同尺度感受野融合
    - 1x1 卷积降维控制参数量
    - 全局平均池化替代全连接，大幅减少参数
    - 22 层深度（含池化层）
"""

import torch
import torch.nn as nn

from cnnlib.models.base import BaseModel
from cnnlib.models.blocks import inception_block
from cnnlib.registry.models import register_model

# Inception 模块参数: (c1, c2_reduce, c2, c3_reduce, c3, c4)
INCEPTION_PARAMS = {
    "3a": (64, 96, 128, 16, 32, 32),
    "3b": (128, 128, 192, 32, 96, 64),
    "4a": (192, 96, 208, 16, 48, 64),
    "4b": (160, 112, 224, 24, 64, 64),
    "4c": (128, 128, 256, 24, 64, 64),
    "4d": (112, 144, 288, 32, 64, 64),
    "4e": (256, 160, 320, 32, 128, 128),
    "5a": (256, 160, 320, 32, 128, 128),
    "5b": (384, 192, 384, 48, 128, 128),
}


def _inception(in_channels: int, name: str) -> inception_block:
    """用查表参数构建一个 Inception 模块"""
    return inception_block(in_channels, *INCEPTION_PARAMS[name])


@register_model(
    "googlenet",
    input_size=224,
    channels=3,
    description="GoogLeNet / Inception v1 (2015)",
)
class GoogLeNet(BaseModel):
    """GoogLeNet (Inception v1)"""

    def __init__(
        self,
        input_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout: float = 0.4,
    ):
        super().__init__(
            input_size=input_size, in_channels=in_channels, num_classes=num_classes
        )

        # ── Stem ────────────────────────────────────────
        self.stem = nn.Sequential(
            # Conv1: (3, 224, 224) → (64, 112, 112)
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Conv2: (64, 56, 56) → (64, 56, 56)
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Conv3: (64, 56, 56) → (192, 56, 56)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # → (192, 28, 28)
        )

        # ── Inception 3a / 3b → MaxPool → 14x14 ────────
        self.inception3 = nn.Sequential(
            _inception(192, "3a"),  # 192 → 256
            _inception(256, "3b"),  # 256 → 480
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── Inception 4a-4e → MaxPool → 7x7 ────────────
        self.inception4 = nn.Sequential(
            _inception(480, "4a"),  # 480 → 512
            _inception(512, "4b"),  # 512 → 512
            _inception(512, "4c"),  # 512 → 512
            _inception(512, "4d"),  # 512 → 528
            _inception(528, "4e"),  # 528 → 832
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── Inception 5a / 5b → 7x7 ────────────────────
        self.inception5 = nn.Sequential(
            _inception(832, "5a"),  # 832 → 832
            _inception(832, "5b"),  # 832 → 1024
        )

        # ── 分类头 ──────────────────────────────────────
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

        self._initWeights()

    def _initWeights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.head(x)
        return x
