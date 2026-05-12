"""
推理预测器

对单张/批量图片进行推理,输出 top-K 预测结果.
支持文件路径、PIL Image、NumPy 数组等多种输入格式.

用法:
    from cnnlib.inference.predictor import Predictor

    predictor = Predictor(model, transform, classNames=classNames)
    result = predictor.predictFromFile("cat.jpg", topK=3)
    # → [{"class": "cat", "confidence": 0.92, "class_idx": 3}, ...]
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from config.defaults import DefaultParams


class Predictor:
    """
    推理预测器

    封装模型推理流程:预处理 → 前传 → softmax → top-K 输出.
    """

    def __init__(
        self,
        model: nn.Module,
        transform,
        device: str = DefaultParams.DEVICE,
        classNames: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            model:      模型实例
            transform:  预处理 transform(不含 batch 维)
            device:     计算设备
            classNames: 类别名称列表(可选,用于结果展示)
        """
        self.model = model.to(device)
        self.transform = transform
        self.device = device
        self.classNames = classNames
        self.model.eval()

    @torch.no_grad()
    def predict(
        self, image: Union[Image.Image, np.ndarray, torch.Tensor], topK: int = 3
    ) -> List[Dict]:
        """
        单张图片推理

        Args:
            image: 输入图片(PIL / numpy HWC / tensor CHW)
            topK:  返回 top-K 预测

        Returns:
            [{"class_idx": int, "class": str, "confidence": float}, ...]
        """
        imageTensor = self._preprocess(image)
        imageTensor = imageTensor.unsqueeze(0).to(self.device)

        outputs = self.model(imageTensor)
        probs = torch.softmax(outputs, dim=1)
        topProbs, topIndices = probs.topk(min(topK, probs.size(1)), dim=1)

        return self._formatResults(
            topIndices[0].cpu().numpy(),
            topProbs[0].cpu().numpy(),
        )

    @torch.no_grad()
    def predictBatch(
        self,
        images: List[Union[Image.Image, np.ndarray, torch.Tensor]],
        topK: int = 3,
    ) -> List[List[Dict]]:
        """
        批量推理

        Args:
            images: 图片列表
            topK:   返回 top-K 预测

        Returns:
            每张图片的 top-K 结果列表
        """
        tensors = [self._preprocess(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        outputs = self.model(batch)
        probs = torch.softmax(outputs, dim=1)
        topProbs, topIndices = probs.topk(min(topK, probs.size(1)), dim=1)

        results = []
        for i in range(len(images)):
            results.append(
                self._formatResults(
                    topIndices[i].cpu().numpy(),
                    topProbs[i].cpu().numpy(),
                )
            )
        return results

    def predictFromFile(self, imagePath: Union[str, Path], topK: int = 3) -> List[Dict]:
        """
        从文件路径推理

        Args:
            imagePath: 图片文件路径
            topK:      返回 top-K 预测

        Returns:
            top-K 预测结果列表
        """
        imagePath = Path(imagePath)
        if not imagePath.exists():
            raise FileNotFoundError(f"图片不存在: {imagePath}")

        image = Image.open(str(imagePath)).convert("RGB")
        return self.predict(image, topK)

    def _preprocess(
        self, image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """统一预处理:输入 → 标准化的 tensor (C, H, W)"""
        # numpy array: HWC → CHW
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[-1] in (1, 3):
                image = image.astype(np.float32)
            image = Image.fromarray(
                (image * 255).astype(np.uint8)
                if image.max() <= 1.0
                else image.astype(np.uint8)
            )
            return self.transform(image)

        # PIL Image
        if isinstance(image, Image.Image):
            return self.transform(image)

        # 已经是 tensor
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                return image
            raise ValueError(f"非预期的 tensor 维度: {image.shape}")

        raise TypeError(f"不支持的图片类型: {type(image)}")

    def _formatResults(self, indices: np.ndarray, probs: np.ndarray) -> List[Dict]:
        """格式化为字典列表"""
        results = []
        for idx, prob in zip(indices, probs):
            entry = {
                "class_idx": int(idx),
                "confidence": float(prob),
            }
            if self.classNames and int(idx) < len(self.classNames):
                entry["class"] = self.classNames[int(idx)]
            results.append(entry)
        return results
