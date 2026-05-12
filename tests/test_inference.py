"""
推理模块单元测试

覆盖: Predictor 单张/批量/文件推理
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from cnnlib.inference.predictor import Predictor


class _DummyModel(nn.Module):
    def __init__(self, numClasses=10):
        super().__init__()
        self.conv = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, numClasses)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv(x)  # → (B, 3, 1, 1)
        x = x.view(x.size(0), -1)  # → (B, 3)
        return self.fc(x)


def _dummyTransform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


# ── Predictor ─────────────────────────────────────────────────


class TestPredictor:
    def test_predict_from_pil(self):
        model = _DummyModel()
        transform = _dummyTransform()
        predictor = Predictor(model, transform, device="cpu")

        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        results = predictor.predict(img, topK=3)
        assert len(results) == 3
        assert "class_idx" in results[0]
        assert "confidence" in results[0]
        # 概率从高到低
        assert (
            results[0]["confidence"]
            >= results[1]["confidence"]
            >= results[2]["confidence"]
        )

    def test_predict_from_numpy(self):
        model = _DummyModel()
        transform = _dummyTransform()
        predictor = Predictor(model, transform, device="cpu")

        arr = np.random.rand(32, 32, 3).astype(np.float32)
        results = predictor.predict(arr, topK=5)
        assert len(results) == 5

    def test_predict_from_tensor(self):
        model = _DummyModel()
        transform = _dummyTransform()
        predictor = Predictor(model, transform, device="cpu")

        tensor = torch.randn(3, 32, 32)
        results = predictor.predict(tensor, topK=3)
        assert len(results) == 3

    def test_predict_batch(self):
        model = _DummyModel()
        transform = _dummyTransform()
        predictor = Predictor(model, transform, device="cpu")

        images = [
            Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            for _ in range(4)
        ]
        results = predictor.predictBatch(images, topK=3)
        assert len(results) == 4
        assert all(len(r) == 3 for r in results)

    def test_predict_from_file(self):
        model = _DummyModel()
        transform = _dummyTransform()
        predictor = Predictor(model, transform, device="cpu")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.jpg")
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            img.save(path)

            results = predictor.predictFromFile(path, topK=3)
            assert len(results) == 3

    def test_predict_file_not_found(self):
        model = _DummyModel()
        predictor = Predictor(model, _dummyTransform(), device="cpu")

        with pytest.raises(FileNotFoundError):
            predictor.predictFromFile("nonexistent.png")

    def test_class_names(self):
        model = _DummyModel()
        transform = _dummyTransform()
        classNames = [
            "airplane",
            "auto",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        predictor = Predictor(model, transform, device="cpu", classNames=classNames)

        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        results = predictor.predict(img, topK=3)
        for r in results:
            assert "class" in r
            assert isinstance(r["class"], str)

    def test_topk_exceeds_available(self):
        model = _DummyModel(numClasses=3)
        transform = _dummyTransform()
        predictor = Predictor(model, transform, device="cpu")

        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        results = predictor.predict(img, topK=10)
        assert len(results) == 3  # 只有 3 个类

    def test_predict_model_in_eval_mode(self):
        model = _DummyModel()
        model.train()
        transform = _dummyTransform()
        predictor = Predictor(model, transform, device="cpu")

        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        predictor.predict(img)
        # 推理后模型应为 eval 模式
        assert not model.training

    def test_predict_deterministic(self):
        model = _DummyModel()
        transform = _dummyTransform()
        predictor = Predictor(model, transform, device="cpu")

        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        r1 = predictor.predict(img, topK=3)
        r2 = predictor.predict(img, topK=3)
        assert r1 == r2
