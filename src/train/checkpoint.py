"""
src/train/checkpoint.py

Save and load model checkpoints.

A checkpoint bundles:
    - model state_dict (weights and biases)
    - optimizer state_dict (Adam momentum buffers)
    - epoch number (for resuming)
    - metrics dict (loss, accuracy, etc.)

Usage:
    from src.train.checkpoint import saveCheckpoint, loadCheckpoint

    # Save after each epoch
    saveCheckpoint(model, optimizer, epoch, metrics, "checkpoints/last_model.pth")

    # Resume training
    startEpoch, metrics = loadCheckpoint("checkpoints/last_model.pth", model, optimizer)

    # Inference only (no optimizer needed)
    _, metrics = loadCheckpoint("checkpoints/best_model.pth", model)
"""

from pathlib import Path

import torch
import torch.nn as nn

from config.default_params import DefaultParams


def saveCheckpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    filepath: str | Path,
) -> None:
    """
    Save a training checkpoint to disk.

    Saves everything needed to resume training from exactly the same state:
    model weights, optimizer buffers, current epoch, and tracked metrics.

    Args:
        model:
            The model whose weights to save. model.state_dict() captures
            all learnable parameters (conv kernels, BN gamma/beta, etc.)
            and persistent buffers (BN running_mean/var).

        optimizer:
            The optimizer to save. optimizer.state_dict() captures:
                - learning rate per param group
                - Adam moment buffers (exp_avg, exp_avg_sq)
                - any other optimizer internal state
            Without this, resuming training would start with fresh Adam state,
            which can cause a momentary spike in loss.

        epoch:
            The epoch number that JUST finished (1-indexed). On resume,
            this becomes the checkpoint to restart FROM — the next epoch
            should be epoch + 1.

        metrics:
            Dictionary of tracked values at this epoch, e.g.:
                {"train_loss": 0.12, "val_loss": 0.15, "val_acc": 0.95}
            Stored for reference; not used by the code, just for the user.

        filepath:
            Destination path. Parent directories are created if needed.
            Convention:
                checkpoints/last_model.pth   — most recent epoch
                checkpoints/best_model.pth   — best validation accuracy
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }

    torch.save(checkpoint, str(filepath))
    print(f"Checkpoint saved → {filepath}")
    print(f"  Epoch: {epoch}  Metrics: {metrics}")


def loadCheckpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = DefaultParams.DEVICE,
) -> tuple[int, dict[str, float]]:
    """
    Load a checkpoint from disk.

    Restores model weights, and optionally optimizer state.
    Returns the epoch and metrics that were stored in the checkpoint.

    Args:
        filepath:
            Path to the .pth file to load.

        model:
            The model to load weights INTO. Must have the same architecture
            as the model that was saved (same layer names and shapes).
            Weights are loaded via model.load_state_dict().

        optimizer:
            Optional optimizer to restore. Pass None when loading for
            inference only (no training will follow). Pass the actual
            optimizer object when resuming training.

        device:
            Device to map the checkpoint tensors to. Default "cpu" since
            checkpoints are typically saved from GPU but may be loaded on
            a different machine.
            Use map_location="cuda" if loading on GPU and the checkpoint
            was saved on GPU.

    Returns:
        (epoch, metrics):
            epoch:   int — the epoch the checkpoint was saved AT. The next
                     epoch to train should be epoch + 1.
            metrics: dict — the metric values tracked at that epoch.
    """
    filepath = Path(filepath)

    # map_location ensures the checkpoint loads correctly even if the
    # GPU topology differs between save and load machines.
    checkpoint = torch.load(str(filepath), map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model weights loaded from {filepath}")

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state loaded")

    epoch = checkpoint["epoch"]
    metrics = checkpoint.get("metrics", {})

    print(f"  Epoch: {epoch}  Metrics: {metrics}")
    return epoch, metrics
