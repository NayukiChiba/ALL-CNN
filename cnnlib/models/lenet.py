"""
LeNet-5 经典卷积神经网络

Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition", 1998.

架构:
    Input (1, 32, 32)
      → Conv2d(1→6, 5×5) + Tanh     → (6, 28, 28)
      → AvgPool2d(2×2, stride=2)     → (6, 14, 14)
      → Conv2d(6→16, 5×5) + Tanh    → (16, 10, 10)
      → AvgPool2d(2×2, stride=2)     → (16, 5, 5)
      → Conv2d(16→120, 5×5) + Tanh  → (120, 1, 1)
      → Flatten                       → (120)
      → Linear(120→84) + Tanh        → (84)
      → Linear(84→num_classes)        → (num_classes)

特点:
    - Tanh 激活函数（原始 LeNet 风格）
    - Average Pooling（非 Max Pooling）
    - Xavier 权重初始化
"""

import torch
import torch.nn as nn

from cnnlib.models.base import BaseModel
from cnnlib.registry.models import register_model


@register_model("lenet", input_size=32, channels=1, description="LeNet-5 (1998)")
class LeNet(BaseModel):
    """
    LeNet-5 经典卷积神经网络

    Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition", 1998.

    架构:
        Input (1, 32, 32)
          → Conv2d(1→6, 5×5) + Tanh     → (6, 28, 28)
          → AvgPool2d(2×2, stride=2)     → (6, 14, 14)
          → Conv2d(6→16, 5×5) + Tanh    → (16, 10, 10)
          → AvgPool2d(2×2, stride=2)     → (16, 5, 5)
          → Conv2d(16→120, 5×5) + Tanh  → (120, 1, 1)
          → Flatten                       → (120)
          → Linear(120→84) + Tanh        → (84)
          → Linear(84→num_classes)        → (num_classes)
    """

    def __init__(
        self, input_size: int = 32, in_channels: int = 1, num_classes: int = 10
    ):
        super().__init__(
            input_size=input_size, in_channels=in_channels, num_classes=num_classes
        )

        # C1: Conv2d(1→6, 5×5, padding=0) — (N,1,32,32) → (N,6,28,28)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=6, kernel_size=5, padding=0
        )

        # S2: AvgPool2d(2×2, stride=2) — (N,6,28,28) → (N,6,14,14)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C3: Conv2d(6→16, 5×5, padding=0) — (N,6,14,14) → (N,16,10,10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)

        # S4: AvgPool2d(2×2, stride=2) — (N,16,10,10) → (N,16,5,5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C5: Conv2d(16→120, 5×5, padding=0) — (N,16,5,5) → (N,120,1,1)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=5, padding=0
        )

        # F6: Linear(120→84) + Tanh — (N,120) → (N,84)
        self.fc1 = nn.Linear(in_features=120, out_features=84)

        # Output: Linear(84→num_classes) — (N,84) → (N,num_classes)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

        self.tanh = nn.Tanh()

        self._initWeights()

    def _initWeights(self):
        # Xavier 初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tanh(self.conv1(x))  # C1
        x = self.pool1(x)  # S2
        x = self.tanh(self.conv2(x))  # C3
        x = self.pool2(x)  # S4
        x = self.tanh(self.conv3(x))  # C5
        x = torch.flatten(x, 1)  # Flatten
        x = self.tanh(self.fc1(x))  # F6
        x = self.fc2(x)  # Output
        return x
