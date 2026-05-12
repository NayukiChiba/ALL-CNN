"""
评估模块单元测试

覆盖: 指标计算、评估器、可视化
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cnnlib.evaluation import (
    Evaluator,
    computeAccuracy,
    computeAllMetrics,
    computeConfusionMatrix,
    computePerClassAccuracy,
    computePrecisionRecallF1,
    computeTopKAccuracy,
    plotConfusionMatrix,
    plotPerClassAccuracy,
    plotPredictions,
    plotTrainingHistory,
)


class _DummyModel(nn.Module):
    def __init__(self, inDim=10, numClasses=5):
        super().__init__()
        self.fc = nn.Linear(inDim, numClasses)

    def forward(self, x):
        return self.fc(x)


def _makeLoader(samples=64, inDim=10, numClasses=5, batchSize=16):
    x = torch.randn(samples, inDim)
    y = torch.randint(0, numClasses, (samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batchSize, shuffle=False)


def _dummyOutputs(samples=64, numClasses=5):
    """模拟模型输出"""
    outputs = torch.randn(samples, numClasses)
    labels = torch.randint(0, numClasses, (samples,))
    return outputs, labels


# ── 指标 ──────────────────────────────────────────────────────


class TestMetrics:
    def test_accuracy_perfect(self):
        outputs = torch.tensor(
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]
        )
        labels = torch.tensor([0, 1, 2])
        assert computeAccuracy(outputs, labels) == 1.0

    def test_accuracy_half(self):
        outputs = torch.tensor([[100.0, 0.0], [0.0, 100.0], [100.0, 0.0], [0.0, 100.0]])
        labels = torch.tensor([0, 1, 1, 0])  # 错 2 个 → 50%
        assert computeAccuracy(outputs, labels) == 0.5

    def test_topk_accuracy(self):
        outputs = torch.tensor(
            [
                [10.0, 5.0, 3.0],
                [3.0, 10.0, 5.0],
                [3.0, 5.0, 10.0],
            ]
        )
        labels = torch.tensor([2, 2, 2])  # 所有 true=2
        # top-1: 0/3=0, top-2: 只看 2 是否在前 2 里
        # sample0: [10,5,3] → top2=[0,1] 无2 → 错
        # sample1: [3,10,5] → top2=[1,2] 有2 → 对
        # sample2: [3,5,10] → top2=[2,1] 有2 → 对
        assert computeTopKAccuracy(outputs, labels, k=1) == 1.0 / 3.0
        assert computeTopKAccuracy(outputs, labels, k=2) == 2.0 / 3.0

    def test_confusion_matrix_shape(self):
        outputs, labels = _dummyOutputs(64, 5)
        cm = computeConfusionMatrix(outputs, labels, 5)
        assert cm.shape == (5, 5)
        assert cm.sum().item() == 64

    def test_confusion_matrix_values(self):
        outputs = torch.tensor([[10.0, 0.0], [0.0, 10.0], [0.0, 10.0], [10.0, 0.0]])
        labels = torch.tensor([0, 1, 1, 0])
        cm = computeConfusionMatrix(outputs, labels, 2)
        expected = torch.tensor([[2, 0], [0, 2]])  # 全对
        assert torch.equal(cm, expected)

    def test_per_class_accuracy(self):
        cm = torch.tensor([[8, 2], [3, 7]])  # class0: 8/10=0.8, class1: 7/10=0.7
        acc = computePerClassAccuracy(cm)
        assert acc[0] == pytest.approx(0.8)
        assert acc[1] == pytest.approx(0.7)

    def test_per_class_accuracy_empty_class(self):
        cm = torch.tensor([[5, 0], [0, 0]])  # class1 has no samples
        acc = computePerClassAccuracy(cm)
        assert acc[0] == 1.0
        assert acc[1] == 0.0  # no samples → 0

    def test_precision_recall_f1(self):
        cm = torch.tensor([[5, 0], [0, 5]])  # perfect
        prf = computePrecisionRecallF1(cm)
        assert prf["macro_f1"] == 1.0
        assert prf["micro_f1"] == 1.0

    def test_precision_recall_f1_imperfect(self):
        cm = torch.tensor([[5, 5], [5, 5]])  # random
        prf = computePrecisionRecallF1(cm)
        assert 0.4 <= prf["macro_f1"] <= 0.6

    def test_compute_all_metrics(self):
        outputs, labels = _dummyOutputs(100, 5)
        metrics = computeAllMetrics(outputs, labels, 5, topK=3)
        assert "accuracy" in metrics
        assert "top3_accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert "per_class_accuracy" in metrics
        assert "macro_f1" in metrics
        assert "micro_f1" in metrics
        # top-3 应 >= top-1
        assert metrics["top3_accuracy"] >= metrics["accuracy"]


# ── 评估器 ────────────────────────────────────────────────────


class TestEvaluator:
    def test_evaluate_returns_all_metrics(self):
        model = _DummyModel()
        loader = _makeLoader(64, 10, 5)
        lossFn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        evaluator = Evaluator(model, loader, lossFn, device, numClasses=5)
        result = evaluator.evaluate()

        assert "loss" in result
        assert "accuracy" in result
        assert "top5_accuracy" in result
        assert "confusion_matrix" in result
        assert "per_class_accuracy" in result
        assert "macro_f1" in result
        assert "micro_f1" in result
        assert result["num_samples"] == 64

    def test_evaluate_per_class(self):
        model = _DummyModel()
        loader = _makeLoader(64, 10, 5)
        lossFn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        evaluator = Evaluator(model, loader, lossFn, device, numClasses=5)
        result = evaluator.evaluatePerClass()

        assert len(result) == 5
        for clsIdx in range(5):
            assert "precision" in result[clsIdx]
            assert "recall" in result[clsIdx]
            assert "f1" in result[clsIdx]
            assert "support" in result[clsIdx]

    def test_evaluate_model_in_eval_mode(self):
        model = _DummyModel()
        loader = _makeLoader(32, 10, 5)
        lossFn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        evaluator = Evaluator(model, loader, lossFn, device, numClasses=5)
        evaluator.evaluate()

        # 评估后模型应回到 eval 模式
        assert not model.training


# ── 可视化 ────────────────────────────────────────────────────


class TestVisualize:
    def test_plot_confusion_matrix(self):
        cm = torch.tensor([[8, 2], [3, 7]])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cm.png")
            fig = plotConfusionMatrix(cm, classNames=["cat", "dog"], savePath=path)
            assert os.path.exists(path)
            assert fig is not None

    def test_plot_confusion_matrix_no_normalize(self):
        cm = torch.tensor([[8, 2], [3, 7]])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cm.png")
            plotConfusionMatrix(cm, normalize=False, savePath=path)
            assert os.path.exists(path)

    def test_plot_training_history(self):
        history = {
            "train_loss": [2.0, 1.5, 1.0, 0.8, 0.5],
            "val_loss": [1.8, 1.4, 1.2, 1.1, 1.0],
            "train_acc": [30.0, 50.0, 65.0, 75.0, 82.0],
            "val_acc": [28.0, 45.0, 58.0, 62.0, 65.0],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.png")
            fig = plotTrainingHistory(history, savePath=path)
            assert os.path.exists(path)
            assert fig is not None

    def test_plot_per_class_accuracy(self):
        perClass = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.8}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "per_class.png")
            fig = plotPerClassAccuracy(perClass, savePath=path)
            assert os.path.exists(path)
            assert fig is not None

    def test_plot_predictions(self):
        class _ImageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(3, 5)

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = _ImageModel()
        x = torch.randn(16, 3, 32, 32)
        y = torch.randint(0, 5, (16,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "preds.png")
            fig = plotPredictions(
                model,
                loader,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                numSamples=4,
                savePath=path,
                device="cpu",
            )
            assert os.path.exists(path)
            assert fig is not None

    def test_plot_save_to_directory(self):
        """保存到目录（自动命名）"""
        cm = torch.eye(3) * 5
        with tempfile.TemporaryDirectory() as tmp:
            fig = plotConfusionMatrix(cm, savePath=tmp)
            assert os.path.exists(os.path.join(tmp, "confusion_matrix.png"))

    def test_plot_with_class_names(self):
        cm = torch.tensor([[5, 1, 0], [0, 6, 0], [1, 0, 4]])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cm.png")
            plotConfusionMatrix(cm, classNames=["A", "B", "C"], savePath=path)
            assert os.path.exists(path)
