"""
AlexNet 深度卷积神经网络

Krizhevsky et al., "ImageNet Classification with Deep Convolutional
Neural Networks", NeurIPS 2012.

架构:
    Input (3, 224, 224)
      → Conv2d(3→96, 11×11, stride=4, pad=2) + ReLU       → (96, 55, 55)
      → MaxPool2d(3×3, stride=2)                            → (96, 27, 27)
      → Conv2d(96→256, 5×5, pad=2) + ReLU                  → (256, 27, 27)
      → MaxPool2d(3×3, stride=2)                            → (256, 13, 13)
      → Conv2d(256→384, 3×3, pad=1) + ReLU                 → (384, 13, 13)
      → Conv2d(384→384, 3×3, pad=1) + ReLU                 → (384, 13, 13)
      → Conv2d(384→256, 3×3, pad=1) + ReLU                 → (256, 13, 13)
      → MaxPool2d(3×3, stride=2)                            → (256, 6, 6)
      → Flatten                                              → (9216)
      → Linear(9216→4096) + ReLU + Dropout(0.5)            → (4096)
      → Linear(4096→4096) + ReLU + Dropout(0.5)            → (4096)
      → Linear(4096→num_classes)                             → (num_classes)

特点:
    - ReLU 激活(首次在 CNN 中大规模使用)
    - Dropout 正则化
    - 分组卷积(原始论文为双 GPU 设计,此处为单 GPU 完整版)
"""

import torch
import torch.nn as nn

from cnnlib.models.base import BaseModel
from cnnlib.models.blocks import linear_block
from cnnlib.registry.models import register_model


@register_model("alexnet", input_size=224, channels=3, description="AlexNet (2012)")
class AlexNet(BaseModel):
    """
    AlexNet 深度卷积神经网络

    """

    def __init__(
        self,
        input_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout: float = 0.5,
    ):
        super().__init__(
            input_size=input_size, in_channels=in_channels, num_classes=num_classes
        )

        # 卷积特提取层
        # conv1: (3, 224, 224) → (96, 55, 55)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding=2,
            ),
            nn.ReLU(inplace=True),
        )

        # maxpool1: (96, 55, 55) → (96, 27, 27)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # conv2: (96, 27, 27) → (256, 27, 27)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # maxpool2: (256, 27, 27) → (256, 13, 13)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # conv3: (256, 13, 13) → (384, 13, 13)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv4: (384, 13, 13) → (384, 13, 13)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # conv5: (384, 13, 13) → (256, 13, 13)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # maxpool5: (256, 13, 13) → (256, 6, 6)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 分类器层

        # 自动推导展平维度
        conv_stack = nn.Sequential(
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.pool5,
        )
        self.feature_dim = self.infer_feature_dim(conv_stack)

        self.classifier = nn.Sequential(
            linear_block(self.feature_dim, 4096, dropout=dropout),
            linear_block(4096, 4096, dropout=dropout),
            nn.Linear(4096, num_classes),
        )

        self._initWeights()

    def _initWeights(self) -> None:
        """Conv 层用 Kaiming 初始化,Linear 层用 Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
