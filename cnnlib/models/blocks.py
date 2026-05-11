"""
公共构建块

CNN 架构常用的可复用模块,每个模型按需取用:

    conv_block     — Conv2d → BN → ReLU → [MaxPool2d]
    linear_block   — Linear → BN → ReLU → [Dropout]
    inception_block — Inception 多分支并行(GoogLeNet 用)
    nin_block       — mlpconv: Conv → 1x1 Conv → 1x1 Conv(NiN 用)

用法:
    from cnnlib.models.blocks import conv_block, inception_block, nin_block
"""

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════
# 基础块:ConvBlock / LinearBlock
# ═══════════════════════════════════════════════════════════════


class conv_block(nn.Module):
    """
    Conv2d → BatchNorm2d → ReLU → [MaxPool2d]

    参数:
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  卷积核大小(默认 3)
        pool:         是否追加 2x2 MaxPool(默认 True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class linear_block(nn.Module):
    """
    Linear → BatchNorm1d → ReLU → [Dropout]

    参数:
        in_features:  输入维度
        out_features: 输出维度
        dropout:      Dropout 概率(None = 不启用)
    """

    def __init__(
        self, in_features: int, out_features: int, dropout: float | None = 0.5
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


# ═══════════════════════════════════════════════════════════════
# Inception 模块(GoogLeNet)
# ═══════════════════════════════════════════════════════════════


class inception_block(nn.Module):
    """
    Inception 模块:四条分支并行,结果在通道维拼接

    分支:
        1x1 Conv
        1x1 Conv → 3x3 Conv
        1x1 Conv → 5x5 Conv
        3x3 MaxPool → 1x1 Conv

    参数:
        in_channels:  输入通道数
        c1:           分支 1 的输出通道数(1x1)
        c2_reduce:    分支 2 的瓶颈通道数(1x1 降维)
        c2:           分支 2 的输出通道数(3x3)
        c3_reduce:    分支 3 的瓶颈通道数
        c3:           分支 3 的输出通道数(5x5)
        c4:           分支 4 的输出通道数(MaxPool + 1x1)

    参考:
        Szegedy et al., "Going Deeper with Convolutions", 2014
        (Inception v1 / GoogLeNet)
    """

    def __init__(
        self,
        in_channels: int,
        c1: int,
        c2_reduce: int,
        c2: int,
        c3_reduce: int,
        c3: int,
        c4: int,
    ):
        super().__init__()

        # 分支 1: 1x1 Conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # 分支 2: 1x1 → 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2_reduce, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 分支 3: 1x1 → 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3_reduce, c3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # 分支 4: 3x3 MaxPool → 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


# ═══════════════════════════════════════════════════════════════
# NiN 模块(Network in Network)
# ═══════════════════════════════════════════════════════════════


class nin_block(nn.Module):
    """
    mlpconv 块:Conv → ReLU → 1x1 Conv → ReLU → 1x1 Conv → ReLU

    用多层感知机替代单层卷积,增强局部感受野内的非线性表达能力.

    1x1 卷积的作用:
        - 等价于对每个像素位置做一次全连接变换
        - 不改变空间尺寸,只改变通道数
        - 参数量极少

    参数:
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  第一层卷积核大小(默认 5)
        padding:      填充大小(默认 same)

    参考:
        Lin et al., "Network In Network", 2014
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ═══════════════════════════════════════════════════════════════
# VGG 卷积块
# ═══════════════════════════════════════════════════════════════


class vgg_conv(nn.Module):
    """
    VGG 基础卷积块：Conv2d(3×3, pad=1) → BatchNorm2d → ReLU

    VGG 的核心设计：全部使用 3×3 卷积 + same padding，
    空间尺寸只由 MaxPool2d(2×2) 控制，卷积本身不改变 H×W。

    参数:
        in_channels:  输入通道数
        out_channels: 输出通道数
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ═══════════════════════════════════════════════════════════════
# 调试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 基础块
    conv = conv_block(1, 32)
    lin = linear_block(3136, 128)
    print(f"conv_block params:  {sum(p.numel() for p in conv.parameters()):,}")
    print(f"linear_block params: {sum(p.numel() for p in lin.parameters()):,}")

    # Inception
    inc = inception_block(192, 64, 96, 128, 16, 32, 32)
    x = torch.randn(2, 192, 28, 28)
    y = inc(x)
    print(
        f"inception_block: {x.shape} -> {y.shape}, params: {sum(p.numel() for p in inc.parameters()):,}"
    )

    # NiN
    nin = nin_block(3, 192)
    x = torch.randn(2, 3, 32, 32)
    y = nin(x)
    print(
        f"nin_block: {x.shape} -> {y.shape}, params: {sum(p.numel() for p in nin.parameters()):,}"
    )
