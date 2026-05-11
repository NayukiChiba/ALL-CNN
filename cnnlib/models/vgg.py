"""
VGG 卷积神经网络

Simonyan & Zisserman, "Very Deep Convolutional Networks for
Large-Scale Image Recognition", ICLR 2015.

变体: VGG11, VGG13, VGG16, VGG19

架构:
    Input (3, 224, 224)
      → [vgg_conv x N1] → MaxPool2d(2)   通道数: 64
      → [vgg_conv x N2] → MaxPool2d(2)   通道数: 128
      → [vgg_conv x N3] → MaxPool2d(2)   通道数: 256
      → [vgg_conv x N4] → MaxPool2d(2)   通道数: 512
      → [vgg_conv x N5] → MaxPool2d(2)   通道数: 512
      → Flatten → Linear(25088→4096) → ReLU → Dropout
      → Linear(4096→4096) → ReLU → Dropout
      → Linear(4096→num_classes)

各变体每 stage 的卷积层数:
    变体     N1 N2 N3 N4 N5  总权重层
    VGG11   1  1  2  2  2    11
    VGG13   2  2  2  2  2    13
    VGG16   2  2  3  3  3    16
    VGG19   2  2  4  4  4    19
"""

import torch
import torch.nn as nn

from cnnlib.models.base import BaseModel
from cnnlib.models.blocks import linear_block, vgg_conv
from cnnlib.registry.models import register_model

# 各变体的每 stage 卷积层数
VGG_CONFIGS = {
    "11": [1, 1, 2, 2, 2],
    "13": [2, 2, 2, 2, 2],
    "16": [2, 2, 3, 3, 3],
    "19": [2, 2, 4, 4, 4],
}

# 每 stage 的输出通道数
VGG_CHANNELS = [64, 128, 256, 512, 512]


def _make_stage(in_channels: int, out_channels: int, num_convs: int) -> nn.Sequential:
    """构建一个 stage 的卷积层序列"""
    layers = []
    for _ in range(num_convs):
        layers.append(vgg_conv(in_channels, out_channels))
        in_channels = out_channels  # 后续卷积输入通道数等于输出通道数
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 每个 stage 以 MaxPool 结尾
    return nn.Sequential(*layers)


class _VGG(BaseModel):
    """
    VGG 基类
    """

    def __init__(
        self,
        input_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        config: str = "16",
        dropout: float = 0.5,
    ):
        super().__init__(
            input_size=input_size, in_channels=in_channels, num_classes=num_classes
        )

        layers_per_stage = VGG_CONFIGS[config]

        # 卷积特征提取器
        stages = []

        in_channels = in_channels
        for out_channels, num_convs in zip(VGG_CHANNELS, layers_per_stage):
            stages.append(_make_stage(in_channels, out_channels, num_convs))
            in_channels = (
                out_channels  # 下一 stage 的输入通道数等于当前 stage 的输出通道数
            )

        self.features = nn.Sequential(*stages)

        # 分类器

        self.feature_dim = self.infer_feature_dim(
            self.features
        )  # 卷积特征维度（全连接输入维度）

        self.classifier = nn.Sequential(
            linear_block(self.feature_dim, 4096, dropout),
            linear_block(4096, 4096, dropout),
            nn.Linear(4096, num_classes),
        )
        self._initWeights()

    def _initWeights(self) -> None:
        """Conv 用 Kaiming 初始化，Linear 用正态初始化"""
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
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@register_model("vgg11", input_size=224, channels=3, description="VGG-11 (2015)")
class VGG11(_VGG):
    def __init__(self, **kwargs):
        kwargs["config"] = "11"
        super().__init__(**kwargs)


@register_model("vgg13", input_size=224, channels=3, description="VGG-13 (2015)")
class VGG13(_VGG):
    def __init__(self, **kwargs):
        kwargs["config"] = "13"
        super().__init__(**kwargs)


@register_model("vgg16", input_size=224, channels=3, description="VGG-16 (2015)")
class VGG16(_VGG):
    def __init__(self, **kwargs):
        kwargs["config"] = "16"
        super().__init__(**kwargs)


@register_model("vgg19", input_size=224, channels=3, description="VGG-19 (2015)")
class VGG19(_VGG):
    def __init__(self, **kwargs):
        kwargs["config"] = "19"
        super().__init__(**kwargs)
