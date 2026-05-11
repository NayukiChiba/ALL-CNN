# cnnlib/models/base.py

"""
CNN 模型基类

不强制子类的内部结构, 只提供通用工具:
    - 自动推导 feature_dim(一次虚拟前传)
    - 参数量统计
    - 结构摘要打印

子类自由选择写法:
    - 简单模型:用 nn.Sequential 堆叠
    - 复杂模型:手写 forward, 用到 ModuleList / ModuleDict 都行

用法:
    from cnnlib.models.base import BaseModel

    class MyNet(BaseModel):
        def __init__(self, num_classes=10):
            super().__init__(input_size=32, in_channels=1, num_classes=num_classes)
            self.conv = nn.Sequential(...)
            self.fc = nn.Sequential(...)

        def forward(self, x):
            x = self.conv(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    CNN 模型基类

    子类需要在 __init__ 中调用 super().__init__() 并传入三个参数:
        input_size  — 输入图像尺寸(正方形边长), 如 32, 224
        in_channels — 输入通道数, 1=灰度 3=RGB
        num_classes — 分类类别数

    基类提供:
        self.input_size、self.in_channels、self.num_classes — 随处可用
        infer_feature_dim(module) — 自动计算展平后维度
        param_count() — 总参数量
        summary() — 结构摘要
    """

    def __init__(self, input_size: int, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._param_count = None

    # 工具方法

    def infer_feature_dim(self, module: nn.Module) -> int:
        """
        对 module 跑一次虚拟前传, 自动计算 flatten 后的维度

        Args:
            module: 特征提取部分(卷积层), 不包含 flatten

        用法:
            self.conv = nn.Sequential(...)
            self.feature_dim = self.infer_feature_dim(self.conv)
        """
        dummy = torch.zeros(1, self.in_channels, self.input_size, self.input_size)
        with torch.no_grad():
            x = module(dummy)
            return int(x.view(1, -1).size(1))

    def param_count(self) -> int:
        """返回模型总参数量"""
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters())
        return self._param_count

    def summary(self) -> str:
        """返回模型结构摘要"""
        return (
            f"{self.__class__.__name__}(\n"
            f"  input:  ({self.in_channels}, {self.input_size}, {self.input_size})\n"
            f"  params: {self.param_count():,}\n"
            f"  classes: {self.num_classes}\n"
            ")"
        )
