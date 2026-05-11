"""
Unit tests for src/inference/predictor.py — Predictor class.

Tests cover: checkpoint loading, single/batch prediction,
input type handling (PIL, numpy, Tensor, file path).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.predictor import Predictor
from src.model.cnn import MNISTCNN
from src.train.checkpoint import saveCheckpoint

# ================================================================
# Fixtures
# ================================================================


@pytest.fixture(scope="module")
def checkpoint_path():
    """Create a temporary checkpoint with a trained-looking model.

    We save an untrained MNISTCNN with random weights. This is enough
    to verify loading and inference pipeline works — the actual digit
    predictions will be random but the shapes and flow are correct.
    """
    model = MNISTCNN()
    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        savepath = tmp.name
    saveCheckpoint(
        model, optimizer, epoch=5, metrics={"val_acc": 0.95}, filepath=savepath
    )
    yield savepath
    # Cleanup
    Path(savepath).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def predictor(checkpoint_path: str) -> Predictor:
    """Create a Predictor from the temporary checkpoint."""
    return Predictor(checkpoint_path, device="cpu")


# ================================================================
# Test images
# ================================================================


def _make_pil_image() -> Image.Image:
    """Create a simple 28x28 grayscale PIL image with a white digit-like blob."""
    array = np.zeros((28, 28), dtype=np.uint8)
    # Draw a rough "7" shape: horizontal top + diagonal
    array[6:9, 6:22] = 255  # top bar
    array[9:22, 16:20] = 255  # vertical/diagonal stroke
    return Image.fromarray(array, mode="L")


def _make_numpy_uint8() -> np.ndarray:
    """Create a numpy uint8 array (28, 28)."""
    return np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)


def _make_numpy_float32() -> np.ndarray:
    """Create a numpy float32 array (28, 28) in [0, 1]."""
    return np.random.rand(28, 28).astype(np.float32)


def _make_tensor_normalized() -> torch.Tensor:
    """Create a normalized tensor (1, 28, 28) float32."""
    return torch.randn(1, 28, 28, dtype=torch.float32)


# ================================================================
# Tests
# ================================================================


class TestPredictorLoading:
    """Tests for checkpoint loading and metadata."""

    def test_loads_from_checkpoint(self, predictor: Predictor):
        """Predictor loads without error and stores metadata."""
        assert predictor.loaded_epoch == 5
        assert predictor.loaded_metrics == {"val_acc": 0.95}
        assert predictor.device == "cpu"

    def test_model_in_eval_mode(self, predictor: Predictor):
        """Model should be in eval mode after loading."""
        assert not predictor.model.training


class TestPredictSingle:
    """Tests for predict() with various input types."""

    def test_pil_image(self, predictor: Predictor):
        """Predict accepts a PIL grayscale Image."""
        image = _make_pil_image()
        result = predictor.predict(image, top_k=3)
        self._assert_result_structure(result, top_k=3)

    def test_pil_rgb_image(self, predictor: Predictor):
        """RGB PIL Image is auto-converted to grayscale."""
        rgb_array = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        image = Image.fromarray(rgb_array, mode="RGB")
        result = predictor.predict(image)
        self._assert_result_structure(result, top_k=5)

    def test_pil_any_size(self, predictor: Predictor):
        """Non-28x28 images are auto-resized."""
        image = Image.fromarray(
            np.random.randint(0, 256, size=(56, 56), dtype=np.uint8), mode="L"
        )
        result = predictor.predict(image)
        self._assert_result_structure(result, top_k=5)

    def test_numpy_uint8_2d(self, predictor: Predictor):
        """numpy (28, 28) uint8 works."""
        array = _make_numpy_uint8()
        result = predictor.predict(array)
        self._assert_result_structure(result, top_k=5)

    def test_numpy_float32_2d(self, predictor: Predictor):
        """numpy (28, 28) float32 [0,1] works."""
        array = _make_numpy_float32()
        result = predictor.predict(array)
        self._assert_result_structure(result, top_k=5)

    def test_numpy_3d_single_channel(self, predictor: Predictor):
        """numpy (28, 28, 1) is squeezed and works."""
        array = np.random.randint(0, 256, size=(28, 28, 1), dtype=np.uint8)
        result = predictor.predict(array)
        self._assert_result_structure(result, top_k=5)

    def test_numpy_3d_rgb(self, predictor: Predictor):
        """numpy (H, W, 3) is converted to grayscale."""
        array = np.random.randint(0, 256, size=(28, 28, 3), dtype=np.uint8)
        result = predictor.predict(array)
        self._assert_result_structure(result, top_k=5)

    def test_tensor_3d_normalized(self, predictor: Predictor):
        """Pre-normalized (1, 28, 28) float32 tensor passes through."""
        tensor = _make_tensor_normalized()
        result = predictor.predict(tensor)
        self._assert_result_structure(result, top_k=5)

    def test_tensor_2d(self, predictor: Predictor):
        """(28, 28) float32 tensor in [0,1] is auto-converted."""
        tensor = torch.rand(28, 28, dtype=torch.float32)
        result = predictor.predict(tensor)
        self._assert_result_structure(result, top_k=5)

    def test_file_path(self, predictor: Predictor):
        """File path to PNG works."""
        image = _make_pil_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            filepath = tmp.name
        try:
            result = predictor.predict(filepath)
            self._assert_result_structure(result, top_k=5)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_top_k_1(self, predictor: Predictor):
        """top_k=1 returns exactly one entry."""
        image = _make_pil_image()
        result = predictor.predict(image, top_k=1)
        assert len(result["top_k"]) == 1

    def test_top_k_10(self, predictor: Predictor):
        """top_k=10 returns all classes."""
        image = _make_pil_image()
        result = predictor.predict(image, top_k=10)
        assert len(result["top_k"]) == 10

    # ---- Result structure assertions ----

    @staticmethod
    def _assert_result_structure(result: dict, top_k: int):
        """Verify the predict() return dict has the correct keys and types."""
        assert "class_index" in result
        assert "class_name" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "top_k" in result

        assert isinstance(result["class_index"], int)
        assert 0 <= result["class_index"] <= 9
        assert isinstance(result["class_name"], str)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["probabilities"].shape == (10,)
        # Probabilities should sum to ~1.0
        assert abs(result["probabilities"].sum().item() - 1.0) < 1e-5

        assert len(result["top_k"]) == top_k
        # top_k should be sorted descending by confidence
        confidences = [entry["confidence"] for entry in result["top_k"]]
        assert confidences == sorted(confidences, reverse=True)

        for entry in result["top_k"]:
            assert "class_index" in entry
            assert "class_name" in entry
            assert "confidence" in entry


class TestPredictBatch:
    """Tests for predictBatch()."""

    def test_batch_pil_images(self, predictor: Predictor):
        """Batch of 3 PIL images."""
        images = [_make_pil_image() for _ in range(3)]
        results = predictor.predictBatch(images, top_k=3)
        assert len(results) == 3
        for result in results:
            TestPredictSingle._assert_result_structure(result, top_k=3)

    def test_batch_mixed_types(self, predictor: Predictor):
        """Batch with mixed PIL + numpy + tensor inputs."""
        images = [
            _make_pil_image(),
            _make_numpy_uint8(),
            _make_tensor_normalized(),
        ]
        results = predictor.predictBatch(images, top_k=3)
        assert len(results) == 3

    def test_empty_batch(self, predictor: Predictor):
        """Empty list returns empty list."""
        results = predictor.predictBatch([], top_k=5)
        assert results == []

    def test_batch_larger_than_default_batch_size(self, predictor: Predictor):
        """Batch larger than batch_size uses multiple mini-batches internally."""
        images = [_make_pil_image() for _ in range(10)]
        results = predictor.predictBatch(images, top_k=3, batch_size=4)
        assert len(results) == 10

    def test_batch_consistency_with_single(self, predictor: Predictor):
        """Same image through predict() and predictBatch() gives same result."""
        image = _make_pil_image()
        single_result = predictor.predict(image, top_k=5)
        batch_results = predictor.predictBatch([image], top_k=5)
        assert single_result["class_index"] == batch_results[0]["class_index"]
        assert abs(single_result["confidence"] - batch_results[0]["confidence"]) < 1e-5
