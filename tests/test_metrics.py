"""
Unit tests for src/eval/metrics.py — confusion matrix and classification report.
"""

import torch

from src.eval.metrics import (
    classificationReport,
    computeConfusionMatrix,
    formatReport,
)


class TestComputeConfusionMatrix:
    """Tests for the confusion matrix builder."""

    def test_all_correct(self):
        """When all predictions are correct, only diagonal entries are non-zero."""
        labels = torch.tensor([0, 1, 2, 3, 4])
        predictions = torch.tensor([0, 1, 2, 3, 4])
        cm = computeConfusionMatrix(labels, predictions, num_classes=5)
        assert cm.diagonal().tolist() == [1, 1, 1, 1, 1]
        assert cm.sum() == 5

    def test_all_wrong(self):
        """Every sample in class 0 is predicted as class 1."""
        labels = torch.tensor([0, 0, 0])
        predictions = torch.tensor([1, 1, 1])
        cm = computeConfusionMatrix(labels, predictions, num_classes=3)
        assert cm[0, 1] == 3  # true 0 → pred 1
        assert cm[0, 0] == 0
        assert cm.sum() == 3

    def test_mixed_predictions(self):
        """Standard mixed case."""
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        predictions = torch.tensor([0, 1, 0, 1, 0, 2])
        cm = computeConfusionMatrix(labels, predictions, num_classes=3)
        assert cm[0, 0] == 1  # true 0 → pred 0
        assert cm[0, 1] == 1  # true 0 → pred 1
        assert cm[1, 0] == 1  # true 1 → pred 0
        assert cm[1, 1] == 1  # true 1 → pred 1
        assert cm[2, 0] == 1  # true 2 → pred 0
        assert cm[2, 2] == 1  # true 2 → pred 2
        assert cm.sum() == 6

    def test_10_classes_default(self):
        """Default num_classes=10 with MNIST-like data."""
        labels = torch.arange(10)
        predictions = torch.arange(10)
        cm = computeConfusionMatrix(labels, predictions)
        assert cm.shape == (10, 10)
        assert cm.diagonal().sum() == 10

    def test_empty_input(self):
        """Empty tensors produce a zero matrix."""
        labels = torch.empty(0, dtype=torch.int64)
        predictions = torch.empty(0, dtype=torch.int64)
        cm = computeConfusionMatrix(labels, predictions, num_classes=3)
        assert cm.shape == (3, 3)
        assert cm.sum() == 0

    def test_large_single_class(self):
        """Many samples from a single class, all correct."""
        labels = torch.zeros(1000, dtype=torch.int64)
        predictions = torch.zeros(1000, dtype=torch.int64)
        cm = computeConfusionMatrix(labels, predictions, num_classes=10)
        assert cm[0, 0] == 1000
        assert cm.sum() == 1000

    def test_row_sum_equals_true_per_class(self):
        """Each row sum equals the number of true samples of that class."""
        labels = torch.tensor([0, 0, 1, 2, 2, 2])
        predictions = torch.tensor([0, 0, 0, 2, 1, 2])
        cm = computeConfusionMatrix(labels, predictions, num_classes=3)
        assert cm.sum(dim=1).tolist() == [2, 1, 3]


class TestClassificationReport:
    """Tests for the per-class precision / recall / F1 report."""

    def test_perfect_classification(self):
        """100% accuracy → precision=1.0, recall=1.0, F1=1.0 for every class."""
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        predictions = torch.tensor([0, 0, 1, 1, 2, 2])
        report = classificationReport(labels, predictions, num_classes=3)
        for entry in report["per_class"]:
            assert entry["precision"] == 1.0
            assert entry["recall"] == 1.0
            assert entry["f1"] == 1.0
        assert report["accuracy"] == 1.0
        assert report["macro_avg"]["f1"] == 1.0

    def test_one_class_completely_wrong(self):
        """Class 0 always predicted as 1 → precision=0, recall=0, F1=0 for class 0."""
        labels = torch.tensor([0, 0, 1, 1])
        predictions = torch.tensor([1, 1, 1, 1])
        report = classificationReport(labels, predictions, num_classes=2)
        assert report["per_class"][0]["precision"] == 0.0
        assert report["per_class"][0]["recall"] == 0.0
        assert report["per_class"][0]["f1"] == 0.0
        assert report["per_class"][0]["support"] == 2

    def test_precision_vs_recall(self):
        """Precision != recall in a typical imbalanced prediction case."""
        # 3 true-positive for class 0, 2 false-positive, 1 false-negative
        labels = torch.tensor([0, 0, 0, 0, 1, 1])
        predictions = torch.tensor([0, 0, 0, 1, 0, 0])
        report = classificationReport(labels, predictions, num_classes=2)
        # Class 0: TP=3, FP=2 (samples 4,5 predicted as 0), FN=1 (sample 3 predicted as 1)
        # precision = 3/5 = 0.6, recall = 3/4 = 0.75
        assert abs(report["per_class"][0]["precision"] - 0.6) < 0.01
        assert abs(report["per_class"][0]["recall"] - 0.75) < 0.01
        assert report["per_class"][0]["support"] == 4

    def test_macro_avg_is_simple_average(self):
        """Macro avg = mean of per-class metrics, not weighted by support."""
        labels = torch.tensor([0, 1, 2, 3])
        predictions = torch.tensor([0, 1, 2, 3])
        report = classificationReport(labels, predictions, num_classes=4)
        assert report["macro_avg"]["precision"] == 1.0
        assert report["macro_avg"]["recall"] == 1.0

    def test_accuracy_computation(self):
        """Overall accuracy = total_correct / total_samples."""
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        predictions = torch.tensor([0, 0, 2, 0, 1, 1])  # 4 correct out of 6
        report = classificationReport(labels, predictions, num_classes=3)
        assert abs(report["accuracy"] - 4 / 6) < 0.01

    def test_custom_class_names(self):
        """Custom class_names appear in the report."""
        labels = torch.tensor([0, 1])
        predictions = torch.tensor([0, 1])
        report = classificationReport(
            labels, predictions, class_names=["zero", "one"], num_classes=2
        )
        assert report["per_class"][0]["class"] == "zero"
        assert report["per_class"][1]["class"] == "one"

    def test_missing_class_in_predictions(self):
        """A class that never appears in predictions gets precision=0."""
        labels = torch.tensor([0, 0, 1, 1])
        predictions = torch.tensor([0, 0, 0, 0])  # class 1 never predicted
        report = classificationReport(labels, predictions, num_classes=2)
        # Class 1: TP=0, FP=0 → precision=0 (div-by-zero guard)
        assert report["per_class"][1]["precision"] == 0.0
        assert report["per_class"][1]["recall"] == 0.0


class TestFormatReport:
    """Tests for the pretty-printing helper."""

    def test_format_returns_string(self):
        """formatReport returns a non-empty multi-line string."""
        labels = torch.tensor([0, 1, 2])
        predictions = torch.tensor([0, 1, 2])
        report = classificationReport(labels, predictions, num_classes=3)
        formatted = formatReport(report)
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "Precision" in formatted
        assert "Recall" in formatted
        assert "Accuracy" in formatted
        assert "0" in formatted  # class name appears
