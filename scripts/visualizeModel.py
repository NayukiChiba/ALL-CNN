"""
Visualization script for MNIST-CNN.

Generates images in the visualizations/ directory:
  1. mnist_samples.png        — Grid of one sample per digit (0-9)
  2. feature_maps_conv1.png    — 32 feature maps after ConvBlock #1
  3. feature_maps_conv2.png    — 64 feature maps after ConvBlock #2
  4. architecture_diagram.png  — Full CNN architecture with data flow
  5. layer_transforms.png      — Single image transformed at each stage
  6. prediction_demo.png       — Random test predictions (correct/incorrect)

Usage:
    uv run python -m scripts.visualizeModel
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyBboxPatch
from torchvision import transforms

import config
from config.default_params import DataParams

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)

OUTPUT_DIR = config.paths.VISUALIZATIONS_DIR


def loadModel(device):
    """Load trained model from checkpoint."""
    from src.model.factory import createModel
    from src.train.checkpoint import loadCheckpoint

    model = createModel(device=device)
    checkpointPath = config.paths.BEST_MODEL_PATH
    if not os.path.exists(checkpointPath):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpointPath}. Train the model first."
        )
    loadCheckpoint(str(checkpointPath), model, optimizer=None)
    model.eval()
    return model


def getOneSamplePerClass():
    """Get one MNIST sample image per digit class (0-9)."""
    from torchvision import datasets

    dataset = datasets.MNIST(
        root=str(config.paths.DATASETS_DIR),
        train=False,
        download=True,
    )

    samples = {}
    for img, label in dataset:
        if label not in samples:
            samples[label] = img
        if len(samples) == 10:
            break
    return [samples[i] for i in range(10)]


def plotMnistSamples():
    """Plot a grid of MNIST digit samples, one per class."""
    print("[1/5] Generating mnist_samples.png ...")
    samples = getOneSamplePerClass()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(samples[idx], cmap="gray")
        ax.set_title(f"Digit {idx}", fontsize=14, fontweight="bold")
        ax.axis("off")

    fig.suptitle("MNIST Handwritten Digit Samples", fontsize=16, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "mnist_samples.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


def plotFeatureMaps(model, device):
    """
    Plot intermediate feature maps using forward hooks.
    Captures activation after ReLU in each ConvBlock.
    """
    from torchvision import datasets

    print("[2/5] Generating feature_maps_conv1.png and feature_maps_conv2.png ...")

    dataset = datasets.MNIST(
        root=str(config.paths.DATASETS_DIR),
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((DataParams.MNIST_MEAN,), (DataParams.MNIST_STD,)),
            ]
        ),
    )
    img, label = dataset[0]
    img = img.unsqueeze(0).to(device)  # (1, 1, 28, 28)

    activations = {}

    def makeHook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()

        return hook

    hooks = []
    for i, block in enumerate(model.conv_blocks):
        hooks.append(block.conv.register_forward_hook(makeHook(f"conv{i + 1}")))
        hooks.append(block.relu.register_forward_hook(makeHook(f"relu{i + 1}")))

    with torch.no_grad():
        model(img)

    for h in hooks:
        h.remove()

    # ---- Original image + ConvBlock #1 activations (32 channels, 28x28) ----
    origImg = img.cpu().squeeze().numpy()
    origImgDenorm = origImg * DataParams.MNIST_STD + DataParams.MNIST_MEAN

    fig, axes = plt.subplots(4, 9, figsize=(14, 7))
    # Top-left: original image
    axes[0, 0].imshow(origImgDenorm, cmap="gray")
    axes[0, 0].set_title("Input (28x28)", fontsize=8)
    axes[0, 0].axis("off")

    conv1Acts = activations.get("relu1", None)
    if conv1Acts is not None:
        conv1Acts = conv1Acts.squeeze(0)  # (32, 28, 28)
        for c in range(32):
            row, col = (c + 1) // 9, (c + 1) % 9
            axes[row, col].imshow(conv1Acts[c].numpy(), cmap="viridis")
            axes[row, col].set_title(f"ch {c}", fontsize=7)
            axes[row, col].axis("off")

    fig.suptitle(
        f"ConvBlock #1 Output — 32 Feature Maps (Digit {label}, 28x28)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "feature_maps_conv1.png")
    fig.savefig(path1)
    plt.close(fig)
    print(f"    Saved: {path1}")

    # ---- ConvBlock #2 activations (64 channels, 14x14) ----
    conv2Acts = activations.get("relu2", None)
    if conv2Acts is not None:
        conv2Acts = conv2Acts.squeeze(0)  # (64, 14, 14)
        fig, axes = plt.subplots(8, 8, figsize=(14, 14))
        for c in range(64):
            row, col = c // 8, c % 8
            axes[row, col].imshow(conv2Acts[c].numpy(), cmap="viridis")
            axes[row, col].set_title(f"ch {c}", fontsize=6)
            axes[row, col].axis("off")

        fig.suptitle(
            f"ConvBlock #2 Output — 64 Feature Maps (Digit {label}, 14x14)",
            fontsize=14,
            y=1.01,
        )
        fig.tight_layout()
        path2 = os.path.join(OUTPUT_DIR, "feature_maps_conv2.png")
        fig.savefig(path2)
        plt.close(fig)
        print(f"    Saved: {path2}")


def plotArchitectureDiagram():
    """Plot the full CNN architecture diagram with data flow arrows."""
    print("[3/5] Generating architecture_diagram.png ...")

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis("off")

    COLOR_CONV = "#4ECDC4"
    COLOR_POOL = "#FF6B6B"
    COLOR_FC = "#FFE66D"
    COLOR_OUT = "#95E1D3"
    COLOR_ARROW = "#2C3E50"
    COLOR_BG = "#F7F9FC"

    ax.set_facecolor(COLOR_BG)

    layers = [
        {
            "x": 0.5,
            "y": 3.5,
            "w": 1.8,
            "h": 1.5,
            "label": "Input\n1x28x28",
            "color": "#E8ECEF",
        },
        {
            "x": 3.0,
            "y": 3.5,
            "w": 2.0,
            "h": 1.5,
            "label": "Conv1\n3x3, 1->32\n+BN+ReLU",
            "color": COLOR_CONV,
        },
        {
            "x": 5.5,
            "y": 3.5,
            "w": 1.5,
            "h": 1.5,
            "label": "MaxPool1\n2x2",
            "color": COLOR_POOL,
        },
        {
            "x": 7.5,
            "y": 3.5,
            "w": 2.0,
            "h": 1.5,
            "label": "Conv2\n3x3, 32->64\n+BN+ReLU",
            "color": COLOR_CONV,
        },
        {
            "x": 10.0,
            "y": 3.5,
            "w": 1.5,
            "h": 1.5,
            "label": "MaxPool2\n2x2",
            "color": COLOR_POOL,
        },
        {
            "x": 12.0,
            "y": 3.5,
            "w": 1.8,
            "h": 1.5,
            "label": "Flatten\n64x7x7\n-> 3136",
            "color": "#DFE6E9",
        },
        {
            "x": 14.0,
            "y": 3.5,
            "w": 2.0,
            "h": 1.5,
            "label": "FC+BN+ReLU\n3136->128\nDropout 0.5",
            "color": COLOR_FC,
        },
        {
            "x": 16.5,
            "y": 3.5,
            "w": 1.2,
            "h": 1.5,
            "label": "Linear\n128->10",
            "color": COLOR_OUT,
        },
    ]

    shapes = {
        "Input\n1x28x28": "(1, 28, 28)",
        "Conv1\n3x3, 1->32\n+BN+ReLU": "(32, 28, 28)",
        "MaxPool1\n2x2": "(32, 14, 14)",
        "Conv2\n3x3, 32->64\n+BN+ReLU": "(64, 14, 14)",
        "MaxPool2\n2x2": "(64, 7, 7)",
        "Flatten\n64x7x7\n-> 3136": "(3136,)",
        "FC+BN+ReLU\n3136->128\nDropout 0.5": "(128,)",
        "Linear\n128->10": "logits (10,)",
    }

    for i, layer in enumerate(layers):
        fancyBox = FancyBboxPatch(
            (layer["x"], layer["y"]),
            layer["w"],
            layer["h"],
            boxstyle="round,pad=0.1",
            facecolor=layer["color"],
            edgecolor="#2C3E50",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.add_patch(fancyBox)

        ax.text(
            layer["x"] + layer["w"] / 2,
            layer["y"] + layer["h"] / 2 + 0.15,
            layer["label"],
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#2C3E50",
        )

        shapeLabel = shapes[layer["label"]]
        ax.text(
            layer["x"] + layer["w"] / 2,
            layer["y"] + 0.15,
            shapeLabel,
            ha="center",
            va="center",
            fontsize=7,
            color="#636E72",
        )

        if i < len(layers) - 1:
            arrowX = layer["x"] + layer["w"]
            nextX = layers[i + 1]["x"]
            ax.annotate(
                "",
                xy=(nextX, 4.25),
                xytext=(arrowX, 4.25),
                arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=2.0),
            )

    # Parameter count footer
    paramText = (
        "Total Params: 422,090    |    "
        "Conv1: 320    Conv2: 18,496    "
        "FC: 401,536    Output: 1,290"
    )
    ax.text(
        9,
        2.2,
        paramText,
        ha="center",
        fontsize=9,
        color="#636E72",
        fontfamily="monospace",
    )

    # Per-layer param breakdown
    breakdownText = (
        "Conv layers: < 5% of total params    |    FC layers: > 95% of total params"
    )
    ax.text(
        9,
        1.8,
        breakdownText,
        ha="center",
        fontsize=8,
        color="#B2BEC3",
    )

    # Title
    ax.text(
        9,
        7.2,
        "MNIST-CNN Architecture",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color="#2C3E50",
    )
    ax.text(
        9,
        6.6,
        "Data Flow: Raw Pixels -> Low-level Features (edges) -> High-level Features (shapes) -> Class Scores",
        ha="center",
        fontsize=11,
        color="#636E72",
    )

    # Legend
    legendY = 1.2
    for label, color, xPos in [
        ("Conv+BN+ReLU", COLOR_CONV, 2.5),
        ("MaxPool (downsample)", COLOR_POOL, 6.5),
        ("Fully Connected", COLOR_FC, 10.5),
        ("Output Layer", COLOR_OUT, 13.5),
    ]:
        rect = FancyBboxPatch(
            (xPos - 0.7, legendY),
            1.4,
            0.4,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#2C3E50",
            linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(xPos, legendY - 0.15, label, ha="center", fontsize=8, color="#2C3E50")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "architecture_diagram.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


def plotLayerTransforms(model, device):
    """
    Plot a single digit image transformed at each stage:
    Input -> Conv1 -> ReLU1 -> MaxPool1 -> Conv2 -> ReLU2 -> MaxPool2 -> FC Hidden
    """
    from torchvision import datasets

    print("[4/5] Generating layer_transforms.png ...")

    dataset = datasets.MNIST(
        root=str(config.paths.DATASETS_DIR),
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((DataParams.MNIST_MEAN,), (DataParams.MNIST_STD,)),
            ]
        ),
    )
    img, label = dataset[0]
    imgBatch = img.unsqueeze(0).to(device)

    activations = {}

    def makeHook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()

        return hook

    hooks = [
        model.conv_blocks[0].conv.register_forward_hook(makeHook("conv1_out")),
        model.conv_blocks[0].relu.register_forward_hook(makeHook("relu1_out")),
        model.conv_blocks[0].pool.register_forward_hook(makeHook("pool1_out")),
        model.conv_blocks[1].conv.register_forward_hook(makeHook("conv2_out")),
        model.conv_blocks[1].relu.register_forward_hook(makeHook("relu2_out")),
        model.conv_blocks[1].pool.register_forward_hook(makeHook("pool2_out")),
        model.fc_block.relu.register_forward_hook(makeHook("fc_hidden")),
    ]

    with torch.no_grad():
        logits = model(imgBatch)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    for h in hooks:
        h.remove()

    origImgDenorm = img.squeeze().numpy() * DataParams.MNIST_STD + DataParams.MNIST_MEAN

    stages = [
        ("Input\n(28x28 grayscale)", origImgDenorm, "gray"),
        (
            "Conv1 Out\n(32ch x 28x28)\n[channel 0 shown]",
            activations["conv1_out"].squeeze(0)[0].numpy(),
            "viridis",
        ),
        (
            "ReLU1 Out\n(32ch x 28x28)\n[channel 0 shown]",
            activations["relu1_out"].squeeze(0)[0].numpy(),
            "viridis",
        ),
        (
            "MaxPool1 Out\n(32ch x 14x14)\n[channel 0 shown]",
            activations["pool1_out"].squeeze(0)[0].numpy(),
            "viridis",
        ),
        (
            "Conv2 Out\n(64ch x 14x14)\n[channel 0 shown]",
            activations["conv2_out"].squeeze(0)[0].numpy(),
            "viridis",
        ),
        (
            "ReLU2 Out\n(64ch x 14x14)\n[channel 0 shown]",
            activations["relu2_out"].squeeze(0)[0].numpy(),
            "viridis",
        ),
        (
            "MaxPool2 Out\n(64ch x 7x7)\n[channel 0 shown]",
            activations["pool2_out"].squeeze(0)[0].numpy(),
            "viridis",
        ),
        (None, None, None),  # placeholder for FC bar chart
    ]

    fig, axes = plt.subplots(1, 8, figsize=(20, 4.5))

    for idx, (title, data, cmap) in enumerate(stages):
        ax = axes[idx]
        if idx < 7 and data is not None:
            ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=7.5)
            ax.axis("off")
        elif idx == 7:
            fcActs = activations["fc_hidden"].squeeze(0).numpy()
            colors = ["#FF6B6B" if v > 0 else "#DFE6E9" for v in fcActs[:50]]
            ax.bar(range(50), fcActs[:50], color=colors, width=0.8)
            ax.set_ylim(0, max(fcActs[:50]) * 1.2)
            ax.set_xlabel("Neuron index", fontsize=7)
            ax.set_ylabel("Activation", fontsize=7)
            ax.set_title(
                f"FC Hidden (128D)\n-> Logits -> Class {pred}",
                fontsize=7.5,
            )
            ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Layer-by-Layer Transform — Digit {label} -> Predicted {pred} (confidence {confidence:.2%})",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "layer_transforms.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


def plotPredictionDemo(model, device):
    """Plot a batch of test predictions with correct/incorrect color coding."""
    from torchvision import datasets

    print("[5/5] Generating prediction_demo.png ...")

    dataset = datasets.MNIST(
        root=str(config.paths.DATASETS_DIR),
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((DataParams.MNIST_MEAN,), (DataParams.MNIST_STD,)),
            ]
        ),
    )

    indices = torch.randperm(len(dataset))[:20]
    images = []
    trueLabels = []
    for idx in indices:
        img, lbl = dataset[idx]
        images.append(img)
        trueLabels.append(lbl)

    batch = torch.stack(images).to(device)
    with torch.no_grad():
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu()
        probs = torch.softmax(logits, dim=1).max(dim=1).values.cpu()

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        imgDenorm = (
            images[i].squeeze().numpy() * DataParams.MNIST_STD + DataParams.MNIST_MEAN
        )
        ax.imshow(imgDenorm, cmap="gray")
        correct = preds[i].item() == trueLabels[i]
        color = "#27AE60" if correct else "#E74C3C"
        symbol = "+" if correct else "-"
        ax.set_title(
            f"[{symbol}] True: {trueLabels[i]}  Pred: {preds[i].item()}\n"
            f"Confidence: {probs[i].item():.2%}",
            fontsize=9,
            color=color,
            fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle(
        "Model Predictions on Test Set (random sample)",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "prediction_demo.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="MNIST-CNN Visualization")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: checkpoints/best_model.pth)",
    )
    parser.add_argument("--device", type=str, default=None, help="Compute device")
    args = parser.parse_args()

    device = args.device or config.default_params.DefaultParams.DEVICE
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model = loadModel(device)
    print("Model loaded.\n")

    plotMnistSamples()
    plotFeatureMaps(model, device)
    plotArchitectureDiagram()
    plotLayerTransforms(model, device)
    plotPredictionDemo(model, device)

    print(f"\nAll images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
