"""
src/inference/predictor.py

MNIST digit predictor — loads a trained checkpoint and runs inference.

Supports single-image and batch prediction from multiple input formats:
    - PIL Image (grayscale or RGB, any size — auto-resized to 28x28)
    - numpy array (uint8 or float32, shape (28,28) or (H,W))
    - torch Tensor (float32, shape (1,28,28) or (C,H,W))
    - file path (str or Path — PNG, JPG, etc.)

Returns top-k class indices, class names, and softmax confidence scores.

Usage:
    from src.inference.predictor import Predictor

    predictor = Predictor("checkpoints/best_model.pth", device="cpu")
    result = predictor.predict("my_digit.png")
    print(f"Predicted: {result['class_name']}, confidence: {result['confidence']:.4f}")

    # Top-3 predictions
    for entry in result["top_k"]:
        print(f"  {entry['class_name']}: {entry['confidence']:.4f}")

    # Batch inference
    results = predictor.predictBatch(["digit1.png", "digit2.png"])
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config.default_params import DataParams, DefaultParams
from config.paths import BEST_MODEL_PATH
from src.model.factory import createModel
from src.train.checkpoint import loadCheckpoint


class Predictor:
    """
    MNIST digit classifier for inference.

    Wraps a trained MNISTCNN model with the preprocessing transform
    pipeline used during training. Handles model loading, input
    normalization, and softmax post-processing.

    The model is loaded once at construction time and can then be
    called repeatedly via predict() / predictBatch().
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = DefaultParams.DEVICE,
        conv_channels: list[int] | None = None,
        hidden_size: int = 128,
        dropout: float = 0.5,
    ):
        """
        Load a trained model from a checkpoint.

        Args:
            checkpoint_path:
                Path to a .pth checkpoint file saved by saveCheckpoint().
                Must contain "model_state_dict" (and optionally "metrics"
                for informational display).

            device:
                Torch device for inference: "cpu" or "cuda".
                Inference on CPU is fast enough for MNIST (~0.5ms/image).

            conv_channels:
                Model architecture — must match the saved checkpoint.
                Default [32, 64] — two ConvBlocks.
                If the checkpoint was trained with a different architecture,
                pass the same values here.

            hidden_size:
                Hidden FC units. Must match the saved checkpoint.

            dropout:
                Dropout rate used during training. Irrelevant for inference
                (dropout is disabled in eval mode), but needed to construct
                the model with the correct architecture.
        """
        self.device = device

        # Build the model with the same architecture as training
        self.model = createModel(
            modelName="cnn",
            conv_channels=conv_channels,
            hidden_size=hidden_size,
            dropout=dropout,
            device=device,
        )

        # Load weights from checkpoint. We pass optimizer=None because we
        # only need the model weights for inference — no training will follow.
        epoch, metrics = loadCheckpoint(
            checkpoint_path,
            self.model,
            optimizer=None,
            device=device,
        )

        self.model.eval()
        self.loaded_epoch = epoch
        self.loaded_metrics = metrics

        # ---- Preprocessing transform ----
        # Must match the validation/test transform from training:
        #   1. Convert any input to PIL grayscale 28x28
        #      (handled separately in _preprocess because we accept
        #       multiple input types)
        #   2. ToTensor:  PIL (H,W) [0,255] → Tensor (1,H,W) [0,1]
        #   3. Normalize: subtract mean, divide by std
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=DataParams.MNIST_MEAN,
                    std=DataParams.MNIST_STD,
                ),
            ]
        )

    # ================================================================
    # Public API
    # ================================================================

    def predict(
        self,
        image: Image.Image | np.ndarray | torch.Tensor | str | Path,
        top_k: int = 5,
    ) -> dict:
        """
        Classify a single digit image.

        Accepts PIL Image, numpy array, torch Tensor, or a file path.
        Images are automatically converted to grayscale, resized to
        28x28, and normalized before being fed to the model.

        Args:
            image:
                Input image in one of these formats:
                - PIL.Image:  grayscale (mode "L") or RGB (mode "RGB"),
                              any size — auto-converted and resized
                - np.ndarray: shape (H,W) or (H,W,C), uint8 [0,255] or
                              float [0,1]. If 3-channel, converted to grayscale.
                - torch.Tensor: shape (1,28,28) float32, already normalized
                                (passed through as-is, no preprocessing)
                - str / Path:  file path to an image (PNG, JPG, etc.)

            top_k:
                Number of top predictions to return in the "top_k" list.
                Default 5 returns the 5 most likely digits.

        Returns:
            result: dict with keys:
                "class_index":  int    — most likely digit (0-9)
                "class_name":   str    — digit as string, e.g. "7"
                "confidence":   float  — softmax probability of the top class
                "probabilities": (10,) float32 Tensor — full probability
                                 distribution over all 10 classes
                "top_k":        list[dict] — top-k predictions, each with
                                {"class_index", "class_name", "confidence"},
                                sorted by confidence descending
        """
        # ---- Step 1: normalize input to a (1, 1, 28, 28) float32 Tensor ----
        image_tensor = self._preprocess(image)

        # ---- Step 2: forward pass ----
        with torch.no_grad():
            logits = self.model(image_tensor)  # (1, 10)
            probabilities = logits.softmax(dim=1).squeeze(0)  # (10,)

        # ---- Step 3: build result ----
        return self._buildResult(probabilities, top_k)

    def predictBatch(
        self,
        images: list[Image.Image | np.ndarray | torch.Tensor | str | Path],
        top_k: int = 5,
        batch_size: int = 64,
    ) -> list[dict]:
        """
        Classify a list of digit images.

        Processes images in mini-batches for GPU efficiency. If the
        total number of images is small (≤ batch_size), they are all
        processed in one forward pass.

        Args:
            images:
                List of images, each in one of the formats accepted
                by predict() (PIL, numpy, Tensor, or file path).

            top_k:
                Number of top predictions per image.

            batch_size:
                Mini-batch size for processing. Larger values use more
                GPU memory but are faster. 64 is fine for MNIST on any GPU.

        Returns:
            List of result dicts, one per input image. Same format as
            predict()'s return value.
        """
        results: list[dict] = []

        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]

            # Preprocess each image and stack into a batch tensor
            batch_tensors = [self._preprocess(image) for image in batch_images]
            batch = torch.cat(batch_tensors, dim=0)  # (B, 1, 28, 28)

            with torch.no_grad():
                logits = self.model(batch)  # (B, 10)
                probabilities = logits.softmax(dim=1)  # (B, 10)

            for index in range(batch.size(0)):
                result = self._buildResult(probabilities[index], top_k)
                results.append(result)

        return results

    # ================================================================
    # Internal helpers
    # ================================================================

    def _preprocess(
        self,
        image: Image.Image | np.ndarray | torch.Tensor | str | Path,
    ) -> torch.Tensor:
        """
        Convert any supported image type to a normalized (1, 1, 28, 28) tensor.

        The preprocessing pipeline:
            1. File path → PIL Image (via PIL.Image.open)
            2. numpy/Tensor → PIL Image (grayscale, 28x28)
            3. PIL Image → apply self.transform → (1, 28, 28) normalized tensor
            4. Add batch dimension → (1, 1, 28, 28)
        """
        # --- File path: load with PIL ---
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # --- numpy array → PIL Image ---
        if isinstance(image, np.ndarray):
            image = self._numpyToPIL(image)

        # --- torch Tensor → PIL Image (if not already the right shape) ---
        # A tensor of shape (1, 28, 28) in float32 is assumed to already
        # be normalized — we just add the batch dim and send it through.
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 1:
                # Already (1, H, W) normalized tensor
                return image.unsqueeze(0).to(self.device)
            # Convert tensor to PIL for proper preprocessing
            # Assume (C, H, W) or (H, W), values in [0, 1] float or [0, 255] uint8
            if image.dim() == 3:
                image = image.squeeze(0)  # (H, W)
            if image.max() <= 1.0 and image.dtype != torch.uint8:
                image = (image * 255).byte()
            image = Image.fromarray(image.cpu().numpy(), mode="L")

        # --- PIL Image: ensure grayscale + 28x28, then apply transform ---
        if image.mode != "L":
            image = image.convert("L")

        if image.size != (28, 28):
            # Use LANCZOS resampling — highest quality for downscaling.
            # LANCZOS uses a sinc-based kernel that preserves fine details
            # better than bilinear or bicubic, important for digit strokes.
            image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Apply the same transform used during training/test:
        #   ToTensor: (H,W) PIL [0,255] → (1,H,W) float [0,1]
        #   Normalize: (x - mean) / std
        tensor = self.transform(image)  # (1, 28, 28)

        # Add batch dimension: (1, 28, 28) → (1, 1, 28, 28)
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def _numpyToPIL(self, array: np.ndarray) -> Image.Image:
        """
        Convert a numpy array to a PIL grayscale Image.

        Handles:
            - (H, W)      grayscale, uint8 [0,255] or float [0,1]
            - (H, W, 1)   grayscale with channel dim
            - (H, W, 3)   RGB — converted to grayscale via luminosity
        """
        # Remove trailing channel dimensions of size 1: (H,W,1) → (H,W)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array.squeeze(-1)

        # RGB → grayscale using standard luminosity weights:
        #   gray = 0.299*R + 0.587*G + 0.114*B
        # These weights match human perception (green is brightest to us).
        if array.ndim == 3 and array.shape[-1] == 3:
            array = (
                0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
            )

        # Convert float [0, 1] to uint8 [0, 255]
        if array.dtype == np.float32 or array.dtype == np.float64:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)

        return Image.fromarray(array, mode="L")

    def _buildResult(self, probabilities: torch.Tensor, top_k: int) -> dict:
        """
        Build the result dict from a probability distribution.

        Args:
            probabilities: (10,) float32 Tensor — softmax probabilities
            top_k: Number of top predictions to include

        Returns:
            Result dict with "class_index", "class_name", "confidence",
            "probabilities", and "top_k".
        """
        # Get the top-k indices and values
        # torch.topk returns (values, indices), both sorted descending
        top_probabilities, top_indices = probabilities.topk(min(top_k, 10))

        top_index = top_indices[0].item()
        top_confidence = top_probabilities[0].item()

        top_k_list: list[dict] = []
        for rank in range(len(top_indices)):
            top_k_list.append(
                {
                    "class_index": top_indices[rank].item(),
                    "class_name": str(top_indices[rank].item()),
                    "confidence": top_probabilities[rank].item(),
                }
            )

        return {
            "class_index": top_index,
            "class_name": str(top_index),
            "confidence": top_confidence,
            "probabilities": probabilities.cpu(),
            "top_k": top_k_list,
        }

    # ================================================================
    # Introspection
    # ================================================================

    def __repr__(self) -> str:
        return (
            f"Predictor(device={self.device!r}, "
            f"checkpoint_epoch={self.loaded_epoch}, "
            f"checkpoint_metrics={self.loaded_metrics!r})"
        )


# ================================================================
# Convenience: quick prediction without creating a Predictor manually
# ================================================================


def predictImage(
    image: Image.Image | np.ndarray | torch.Tensor | str | Path,
    checkpoint_path: str | Path = BEST_MODEL_PATH,
    device: str = DefaultParams.DEVICE,
    top_k: int = 5,
) -> dict:
    """
    One-shot: load checkpoint + predict a single image.

    This is a convenience function for quick testing. For repeated
    inference, create a Predictor instance once and reuse it —
    that avoids reloading the model every call.

    Args:
        image: Input image (PIL, numpy, Tensor, or file path).
        checkpoint_path: Path to .pth checkpoint. Defaults to BEST_MODEL_PATH.
        device: "cpu" or "cuda".
        top_k: Number of top predictions to return.

    Returns:
        Result dict (same format as Predictor.predict()).
    """
    predictor = Predictor(checkpoint_path, device=device)
    return predictor.predict(image, top_k=top_k)
