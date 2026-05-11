"""
src/model/lenet.py

经典 LeNet-5 架构实现，来自 LeNet 学习项目。

Architecture:
    Input (1, 32, 32)
      → Conv2d(1→6, 5×5) + Tanh     → (6, 28, 28)
      → AvgPool2d(2×2)               → (6, 14, 14)
      → Conv2d(6→16, 5×5) + Tanh   → (16, 10, 10)
      → AvgPool2d(2×2)               → (16, 5, 5)
      → Conv2d(16→120, 5×5) + Tanh → (120, 1, 1)
      → Flatten                       → (120)
      → Linear(120→84) + Tanh       → (84)
      → Linear(84→numClasses)        → (numClasses)

特点：
    - 使用 Tanh 激活函数（原始 LeNet 风格）
    - Xavier 权重初始化
    - Average Pooling（非 Max Pooling）
    - 原始论文使用 32×32 输入，28×28 图像需 Resize

Usage:
    from src.model.lenet import LeNet
    model = LeNet(numClasses=10)
    logits = model(images)
"""

import torch
from torch import nn


class LeNet(nn.Module):
    """
    LeNet-5 经典卷积神经网络

    Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition", 1998.
    """

    def __init__(self, numClasses: int = 10):
        super().__init__()

        # C1: Conv2d, 输入通道数 1，输出通道数 6，卷积核大小 5x5，步长 1，填充 0
        # (N, 1, 28, 28) -> (N, 6, 28, 28)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0
        )

        # S2: AvgPool2d, 池化核大小 2x2，步长 2
        # (N, 6, 28, 28) -> (N, 6, 14, 14)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C3: Conv2d, 输入通道数 6，输出通道数 16，卷积核大小 5x5，步长 1，填充 0
        # (N, 6, 14, 14) -> (N, 16, 10, 10)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )

        # S4: AvgPool2d, 池化核为 2x2，步长为 2
        # (N, 16, 10, 10) -> (N, 16, 5, 5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C5: Conv2d, 输入通道为16，输出通道为120，卷积核为 5x5，步长 1， 填充0
        # (N, 16, 5, 5) -> (N, 120, 1, 1)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0
        )

        # F6: Linear, 全连接层，把整个conv全部拉直
        # (N, 120) -> (N, 84)
        self.fc1 = nn.Linear(120, 84)

        # Output: Linear，全连接层，把 84 个特征对应到 num_classes 个结果中
        # (N, 84) -> (N, num_classes)
        self.fc2 = nn.Linear(84, numClasses)

        # 激活函数
        self.tanh = nn.Tanh()

        # Xavier 初始化（Tanh 激活函数适用）
        self._initWeights()

    def _initWeights(self) -> None:
        """对 Conv 和 Linear 层使用 Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tanh(self.conv1(x))
        x = self.pool1(x)
        x = self.tanh(self.conv2(x))
        x = self.pool2(x)
        x = self.tanh(self.conv3(x))
        x = torch.flatten(x, 1)  # (N, 120, 1, 1) -> (N, 120)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
