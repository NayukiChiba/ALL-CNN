"""
NiN — Network in Network

Lin et al., "Network In Network", ICLR 2014.

架构:
    Input (3, 32, 32)
      → nin_block(3→192,   k=5) → ReLU → MaxPool(3×3, stride=2)   → (192, 15, 15)
      → nin_block(192→160, k=5) → ReLU → MaxPool(3×3, stride=2)   → (160, 6, 6)
      → nin_block(160→96,  k=3) → ReLU → MaxPool(3×3, stride=2)   → (96, 2, 2)
      → nin_block(96→num_classes, k=3) → ReLU                     → (num_classes, 2, 2)
      → AdaptiveAvgPool2d(1)                                        → (num_classes, 1, 1)
      → flatten                                                     → (num_classes)

核心创新:
    - mlpconv 用多层感知机替代单层卷积，增强局部感受野内的非线性
    - 用全局平均池化替代全连接层，参数量极少，不易过拟合
"""

import torch
import torch.nn as nn

from cnnlib.models.base import BaseModel
from cnnlib.models.blocks import nin_block
from cnnlib.registry.models import register_model


@register_model(
    "nin", input_size=32, channels=3, description="Network In Network (2014)"
)
class NiN(BaseModel):
    """Network in Network"""

    def __init__(
        self,
        input_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        super().__init__(
            input_size=input_size, in_channels=in_channels, num_classes=num_classes
        )

        # ── 三个 mlpconv + MaxPool 阶段 ──────────────────
        self.stage1 = nn.Sequential(
            nin_block(in_channels, 192, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage2 = nn.Sequential(
            nin_block(192, 160, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage3 = nn.Sequential(
            nin_block(160, 96, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── 分类 mlpconv + 全局平均池化 ──────────────────
        self.classifier = nn.Sequential(
            nin_block(96, num_classes, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self._initWeights()

    def _initWeights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)  # (N, num_classes, 1, 1) → (N, num_classes)
        return x
