"""
src/eval/visualize.py

Visualization utilities for training analysis and error inspection.

Functions:
    plotTrainingCurves   — loss + accuracy subplots over epochs
    plotConfusionMatrix  — confusion matrix heatmap
    plotErrorGrid        — grid of misclassified samples

All functions accept a save_path parameter. If None, the figure is
displayed interactively (plt.show()). Otherwise it's saved to disk.

Usage:
    from src.eval.visualize import plotTrainingCurves, plotConfusionMatrix, plotErrorGrid

    history = {"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...]}
    plotTrainingCurves(history, save_path="outputs/curves.png")

    confusion_matrix = result["confusion_matrix"]
    plotConfusionMatrix(confusion_matrix, save_path="outputs/confusion_matrix.png")

    plotErrorGrid(images, true_labels, pred_labels, save_path="outputs/errors.png")
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Use a non-interactive backend when saving to file (no GUI needed).
# This avoids errors on headless servers (no $DISPLAY).
# When the user wants to show the plot, we temporarily switch back.
matplotlib.use("Agg")

from config.default_params import DefaultParams
from config.paths import VISUALIZATIONS_DIR

# ================================================================
# Global style settings
# ================================================================

# Apply a clean default style to all plots. These are overridable
# by passing custom kwargs to individual functions.
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "image.cmap": "Blues",
    }
)


# ================================================================
# 1. Training curves
# ================================================================


def plotTrainingCurves(
    history: dict,
    save_path: str | Path | None = None,
    title: str = "Training Curves",
    figure_size: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot training and validation loss + accuracy curves side by side.

    Produces a 1-row × 2-column figure:
        Left:  loss curves (train + val) over epochs
        Right: accuracy curves (train + val) over epochs

    The two subplots share the same x-axis (epoch number) but have
    independent y-axes since loss and accuracy have different scales
    (loss is typically 0.0-2.0, accuracy is 0.0-1.0).

    Args:
        history:
            Dictionary with keys:
                "train_loss": list[float] — training loss per epoch
                "val_loss":   list[float] — validation loss per epoch
                "train_acc":  list[float] — training accuracy per epoch
                "val_acc":    list[float] — validation accuracy per epoch
            All four lists must have the same length (= number of epochs).

        save_path:
            File path to save the figure. If None, the figure is
            displayed with plt.show(). Parent directories are created
            automatically.

        title:
            Overall figure title (suptitle).

        figure_size:
            Figure dimensions as (width, height) in inches.

    Returns:
        matplotlib Figure object (for further customization if needed).
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, (loss_axes, accuracy_axes) = plt.subplots(1, 2, figsize=figure_size)
    figure.suptitle(title, fontsize=14, fontweight="bold")

    # ---- Left: Loss curves ----
    # Both train and val loss on the same axes. Train loss is shown with
    # higher opacity (solid feel) since it's the primary signal; val loss
    # is slightly transparent to distinguish it.
    loss_axes.plot(
        epochs,
        history["train_loss"],
        "b-",
        label="Train Loss",
        alpha=0.8,
        linewidth=1.5,
    )
    loss_axes.plot(
        epochs, history["val_loss"], "r-", label="Val Loss", alpha=0.7, linewidth=1.5
    )
    loss_axes.set_xlabel("Epoch")
    loss_axes.set_ylabel("Loss")
    loss_axes.set_title("Loss")
    loss_axes.legend(loc="upper right")
    loss_axes.grid(True, alpha=0.3)

    # Mark the best epoch (lowest val loss)
    best_loss_epoch = (
        np.argmin(history["val_loss"]) + 1
    )  # +1 because epochs are 1-indexed
    best_loss_value = history["val_loss"][best_loss_epoch - 1]
    loss_axes.axvline(
        x=best_loss_epoch, color="gray", linestyle="--", alpha=0.5, linewidth=0.8
    )
    loss_axes.annotate(
        f"Best: {best_loss_value:.4f}",
        xy=(best_loss_epoch, best_loss_value),
        fontsize=8,
        color="gray",
    )

    # ---- Right: Accuracy curves ----
    accuracy_axes.plot(
        epochs, history["train_acc"], "b-", label="Train Acc", alpha=0.8, linewidth=1.5
    )
    accuracy_axes.plot(
        epochs, history["val_acc"], "r-", label="Val Acc", alpha=0.7, linewidth=1.5
    )
    accuracy_axes.set_xlabel("Epoch")
    accuracy_axes.set_ylabel("Accuracy")
    accuracy_axes.set_title("Accuracy")
    accuracy_axes.legend(loc="lower right")
    accuracy_axes.grid(True, alpha=0.3)

    # Mark the best epoch (highest val accuracy)
    best_accuracy_epoch = np.argmax(history["val_acc"]) + 1
    best_accuracy_value = history["val_acc"][best_accuracy_epoch - 1]
    accuracy_axes.axvline(
        x=best_accuracy_epoch, color="gray", linestyle="--", alpha=0.5, linewidth=0.8
    )
    accuracy_axes.annotate(
        f"Best: {best_accuracy_value:.4f}",
        xy=(best_accuracy_epoch, best_accuracy_value),
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(figure)
    else:
        # Temporarily switch to interactive backend for display
        matplotlib.use("TkAgg", force=True)
        plt.show()

    return figure


# ================================================================
# 2. Confusion matrix heatmap
# ================================================================


def plotConfusionMatrix(
    confusion_matrix: torch.Tensor | np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figure_size: tuple[float, float] = (8, 7),
    colormap: str = "Blues",
) -> plt.Figure:
    """
    Plot a confusion matrix as an annotated heatmap.

    Each cell (i, j) shows how many (or what fraction of) true class i
    samples were predicted as class j.

    Diagonal cells (i, i): correctly classified.
    Off-diagonal cells (i, j), i ≠ j: confusions / errors.

    When normalize=True (default), each row sums to 1.0, so cell values
    show the recall for that (true, pred) pair. This makes it easier to
    spot which digit pairs the model confuses most.

    Args:
        confusion_matrix:
            Confusion matrix of shape (num_classes, num_classes).
            Can be a torch Tensor or numpy array.

        class_names:
            Labels for the axes. Defaults to ["0", "1", ..., "9"].

        save_path:
            File path to save. None to display interactively.

        normalize:
            If True, normalize each row by its sum (row-sum = 1.0).
            Cell values become fractions of the true class.
            If False, show raw counts.

        title:
            Figure title.

        figure_size:
            Figure dimensions (width, height) in inches.

        colormap:
            Matplotlib colormap name. "Blues" is clean and readable.
            "Reds" also works well. Avoid "jet" — it's perceptually
            non-uniform and misleading for statistical data.

    Returns:
        matplotlib Figure object.
    """
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.numpy()

    # Normalize: divide each row by its sum so cells are fractions [0, 1]
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero for classes with zero support
        row_sums = np.where(row_sums == 0, 1, row_sums)
        matrix_normalized = confusion_matrix.astype(np.float64) / row_sums
    else:
        matrix_normalized = confusion_matrix.astype(np.float64)

    num_classes = confusion_matrix.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    figure, axes = plt.subplots(figsize=figure_size)

    # imshow: display the matrix as a color grid
    heatmap = axes.imshow(
        matrix_normalized, interpolation="nearest", cmap=colormap, vmin=0, vmax=1
    )

    # Colorbar: shows the scale (0 = dark, 1 = light)
    colorbar = figure.colorbar(heatmap, ax=axes, shrink=0.78)
    colorbar.ax.set_ylabel(
        "Fraction of true class" if normalize else "Count",
        rotation=270,
        labelpad=15,
    )

    # Tick marks: center on each cell
    tick_positions = np.arange(num_classes)
    axes.set_xticks(tick_positions)
    axes.set_yticks(tick_positions)
    axes.set_xticklabels(class_names, fontsize=10)
    axes.set_yticklabels(class_names, fontsize=10)

    # Labels
    axes.set_xlabel("Predicted", fontsize=12)
    axes.set_ylabel("True", fontsize=12)
    axes.set_title(title, fontsize=13, fontweight="bold")

    # Annotate each cell with its value.
    # For normalized: show as decimal (e.g. "0.98")
    # For raw counts: show as integer (e.g. "980")
    threshold = matrix_normalized.max() / 2.0  # Switch text color at midpoint
    for row in range(num_classes):
        for column in range(num_classes):
            if normalize:
                text = f"{matrix_normalized[row, column]:.2f}"
            else:
                text = f"{int(confusion_matrix[row, column])}"
            axes.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white"
                if matrix_normalized[row, column] > threshold
                else "black",
            )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(figure)
    else:
        matplotlib.use("TkAgg", force=True)
        plt.show()

    return figure


# ================================================================
# 3. Error prediction grid
# ================================================================


def plotErrorGrid(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    class_names: list[str] | None = None,
    max_samples: int = 25,
    save_path: str | Path | None = None,
    title: str = "Misclassified Samples",
    figure_size: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Display a grid of misclassified samples.

    Each cell shows:
        - The MNIST image (grayscale, 28×28)
        - The true label and the (wrong) predicted label

    This is the most useful diagnostic for understanding WHAT the model
    gets wrong. Looking at error grids often reveals patterns:
        - "Oh, the model confuses 4 and 9" (both have loops)
        - "That 7 looks exactly like a 1"
        - "This 8 is so badly written that even I can't tell"

    Args:
        images:
            Tensor of shape (N, 1, 28, 28) — the misclassified images.
            Batch dimension N should already be filtered to errors only.
            If N > max_samples, a random subset of max_samples is shown.

        true_labels:
            (N,) int64 Tensor — true class for each image.

        predicted_labels:
            (N,) int64 Tensor — (wrong) predicted class for each image.

        class_names:
            Class names for the label text. Defaults to string digits.

        max_samples:
            Maximum number of error samples to show. Grid layout auto-fits:
            25 → 5×5, 16 → 4×4, 9 → 3×3, etc.

        save_path:
            File path to save. None to display interactively.

        title:
            Overall figure suptitle.

        figure_size:
            Figure size. If None, auto-computes based on grid dimensions.

    Returns:
        matplotlib Figure object.
    """
    num_samples = images.size(0)
    if num_samples == 0:
        figure, axes = plt.subplots(figsize=(6, 2))
        axes.text(
            0.5, 0.5, "No misclassified samples!", ha="center", va="center", fontsize=14
        )
        axes.axis("off")
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
            plt.close(figure)
        return figure

    # Take a random subset so each viewing shows different errors
    if num_samples > max_samples:
        indices = torch.randperm(num_samples)[:max_samples]
        images = images[indices]
        true_labels = true_labels[indices]
        predicted_labels = predicted_labels[indices]
        num_samples = max_samples

    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # Compute grid layout: try for a square-ish grid
    grid_columns = int(np.ceil(np.sqrt(num_samples)))
    grid_rows = int(np.ceil(num_samples / grid_columns))

    if figure_size is None:
        figure_size = (grid_columns * 1.8, grid_rows * 2.0)

    figure, axes = plt.subplots(grid_rows, grid_columns, figsize=figure_size)
    figure.suptitle(title, fontsize=14, fontweight="bold")

    # Flatten axes array for uniform indexing (handles both 1D and 2D cases)
    if grid_rows == 1 and grid_columns == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    # MNIST normalization values (same as transform.py)
    normalization_mean = 0.1307
    normalization_std = 0.3081

    for index in range(grid_rows * grid_columns):
        axes[index].axis("off")

        if index < num_samples:
            # De-normalize: reverse the Normalize(mean, std) transform
            # normalized = (image - mean) / std  →  image = normalized * std + mean
            image = images[index]  # (1, 28, 28)
            image = image * normalization_std + normalization_mean

            # Clamp to [0, 1] for display (handles any normalization artifacts)
            image = image.clamp(0, 1)

            # Convert (1, 28, 28) → (28, 28) for imshow
            image_array = image.squeeze(0).cpu().numpy()

            true_class = true_labels[index].item()
            predicted_class = predicted_labels[index].item()

            axes[index].imshow(image_array, cmap="gray_r", vmin=0, vmax=1)

            # Green text if correct, red if wrong. In an error grid,
            # it should always be wrong — but we keep the logic for robustness.
            text_color = "green" if true_class == predicted_class else "red"
            axes[index].set_title(
                f"True: {class_names[true_class]}  Pred: {class_names[predicted_class]}",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(figure)
    else:
        matplotlib.use("TkAgg", force=True)
        plt.show()

    return figure


# ================================================================
# Convenience: gather error samples from a model run
# ================================================================


@torch.no_grad()
def gatherErrorSamples(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = DefaultParams.DEVICE,
    max_errors: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the model and collect all misclassified samples.

    This function does one forward pass over the DataLoader and returns
    the images, true labels, and predicted labels for every incorrectly
    classified sample. The returned tensors can be fed directly to
    plotErrorGrid().

    Stops collecting once max_errors samples are gathered (checking each
    batch to avoid unnecessarily processing the entire dataset).

    Args:
        model: The trained nn.Module (eval mode applied internally).
        dataloader: DataLoader for the set to inspect.
        device: "cuda" or "cpu".
        max_errors: Maximum number of error samples to collect.

    Returns:
        (error_images, error_true_labels, error_predicted_labels):
            error_images:            (E, 1, 28, 28) float32 Tensor
            error_true_labels:       (E,) int64 Tensor
            error_predicted_labels:  (E,) int64 Tensor
            where E ≤ max_errors.
    """
    was_training = model.training
    model.eval()

    error_image_list: list[torch.Tensor] = []
    error_true_list: list[torch.Tensor] = []
    error_predicted_list: list[torch.Tensor] = []

    for images, labels in dataloader:
        if sum(tensor.size(0) for tensor in error_image_list) >= max_errors:
            break

        batch_images = images.to(device)
        batch_labels = labels.to(device)

        logits = model(batch_images)
        _, predictions = logits.max(dim=1)

        # Find which samples in this batch are misclassified
        wrong_mask = ~predictions.eq(batch_labels)  # (B,) bool Tensor

        if wrong_mask.any():
            wrong_images = batch_images[wrong_mask].cpu()
            wrong_true_labels = batch_labels[wrong_mask].cpu()
            wrong_predicted_labels = predictions[wrong_mask].cpu()

            error_image_list.append(wrong_images)
            error_true_list.append(wrong_true_labels)
            error_predicted_list.append(wrong_predicted_labels)

    model.train(was_training)

    if not error_image_list:
        return (
            torch.empty(0, 1, 28, 28),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )

    error_images = torch.cat(error_image_list)[:max_errors]
    error_true_labels = torch.cat(error_true_list)[:max_errors]
    error_predicted_labels = torch.cat(error_predicted_list)[:max_errors]

    return error_images, error_true_labels, error_predicted_labels


# ================================================================
# Batch visualization: run eval + generate all plots
# ================================================================


def generateEvaluationPlots(
    model: nn.Module,
    dataloader: DataLoader,
    history: dict,
    device: str = DefaultParams.DEVICE,
    output_dir: str | Path | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Path]:
    """
    Run evaluation and generate all standard plots in one call.

    Produces three files:
        1. training_curves.png    — loss + accuracy curves
        2. confusion_matrix.png   — confusion matrix heatmap
        3. error_grid.png         — misclassified sample grid

    Args:
        model: Trained nn.Module.
        dataloader: Test/eval DataLoader for confusion matrix and errors.
        history: Training history dict (see plotTrainingCurves).
        device: "cuda" or "cpu".
        output_dir: Directory for output files. Defaults to
                    config.paths.VISUALIZATIONS_DIR.
        class_names: Class names for plots. Defaults to digit strings.

    Returns:
        Dict mapping plot names to their saved file paths:
            {"curves": Path, "confusion": Path, "errors": Path}
    """
    if output_dir is None:
        output_dir = VISUALIZATIONS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = [str(i) for i in range(10)]

    saved: dict[str, Path] = {}

    # 1. Training curves (no model inference needed — uses history directly)
    curves_path = output_dir / "training_curves.png"
    plotTrainingCurves(history, save_path=curves_path)
    saved["curves"] = curves_path
    print(f"Training curves saved → {curves_path}")

    # 2. Confusion matrix — requires one forward pass
    from src.eval.metrics import evaluateModel

    result = evaluateModel(
        model, dataloader, criterion=None, device=device, class_names=class_names
    )
    confusion_matrix = result["confusion_matrix"]

    confusion_matrix_path = output_dir / "confusion_matrix.png"
    plotConfusionMatrix(
        confusion_matrix, class_names=class_names, save_path=confusion_matrix_path
    )
    saved["confusion"] = confusion_matrix_path
    print(f"Confusion matrix saved → {confusion_matrix_path}")

    # 3. Error grid — requires gathering misclassified samples
    error_images, error_true_labels, error_predicted_labels = gatherErrorSamples(
        model, dataloader, device=device
    )

    if error_images.size(0) > 0:
        errors_path = output_dir / "error_grid.png"
        plotErrorGrid(
            error_images,
            error_true_labels,
            error_predicted_labels,
            class_names=class_names,
            save_path=errors_path,
        )
        saved["errors"] = errors_path
        print(f"Error grid saved → {errors_path}")
    else:
        print("No misclassified samples found — skipping error grid")

    return saved
