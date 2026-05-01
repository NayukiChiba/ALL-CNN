"""
src/data/process.py

Read eaw MNIST, apply transforms pipelines, save preprocessed .pt files

Outputs:
    datasets/MNIST/processed/
        - train.pt: training set (images and labels)
        - test.pt: test set (images and labels)

Usage:
    uv run python -m src.data.process
    from src.data.process import loadProcessed
    images, labels = loadProcessed("train")

"""

from pathlib import Path
from typing import Literal, Tuple

import torch
from torchvision.datasets import MNIST
from tqdm import tqdm

from config.paths import DATASETS_DIR
from src.data.transform import getTestTransform

PROCESSED_DIR = DATASETS_DIR / "MNIST" / "processed"


def processSplit(train: bool, savePath: str | Path) -> None:
    """
    Process one split:
        read raw MNIST -> ToTensor -> Normalize -> save .pt file
    Args:
        train(bool): whether to process the training set (True) or test set (False)
        savePath(str or Path): where to save the processed .pt file

    """
    savePath = Path(savePath)
    name = "train" if train else "test"
    print(f"Processing {name} split...")

    # load raw MNIST dataset with appropriate transform
    dataset = MNIST(root=str(DATASETS_DIR), train=train, transform=getTestTransform())

    images, labels = [], []

    # iterate through the dataset, apply transforms, and collect images and labels
    for i in tqdm(range(len(dataset)), desc=f"Processing {name}"):
        img, label = dataset[i]
        images.append(img)
        labels.append(label)

    # stack images and labels into tensors and save as .pt file
    images = torch.stack(images)  # shape: (N, C, H, W)
    labels = torch.tensor(labels)  # shape: (N,)

    savePath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "images": images,
            "labels": labels,
        },
        str(savePath),
    )

    print(f" Saved -> {savePath} | images: {images.shape} labels: {labels.shape}")


def processAll() -> None:
    """Process both train and test splits"""
    print("=" * 60)
    print("Raw -> Processed")
    processSplit(train=True, savePath=PROCESSED_DIR / "train.pt")
    processSplit(train=False, savePath=PROCESSED_DIR / "test.pt")
    print("Done")


def loadProcessed(split: Literal["train", "test"]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load processed .pt file for the specified split.

    Args:
        split(str): "train" or "test"

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (images, labels)
    """
    path = PROCESSED_DIR / f"{split}.pt"
    data = torch.load(path)
    return data["images"], data["labels"]


if __name__ == "__main__":
    processAll()
