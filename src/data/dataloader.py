"""
src/data/dataloader.py

Build train / val / test DataLoaders for MNIST

Data split:
    Raw MNIST: 60k train, 10k test
    Our split:
        - train: 60k * (1 - val_ratio)
        - val: 60k * val_ratio
        - test: 10k

Usage:
    from src.data.dataloader import buildDataLoaders
    trainLoader, valLoader, testLoader = buildDataLoaders(
        batchSize=64, numWorkers=4, valSplit=0.1
    )
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

from config.default_params import DataParams, DefaultParams
from config.paths import DATASETS_DIR
from src.data.transform import getTestTransform, getTrainTransform


def _splitIndices(datasetSize: int, valSplit: float) -> Tuple[list[int], list[int]]:
    """
    Generate shuffled train / val indices with fixed seed for reproducibility.

    Args:
        datasetSize (int): Total number of samples in the full training set (60,000).
        valSplit (float): Fraction reserved for validation (e.g., 0.1 → 6,000).

    Returns:
        (trainIndices, valIndices) — disjoint lists of int indices.
    """
    indices = torch.randperm(
        datasetSize, generator=torch.Generator().manual_seed(DefaultParams.SEED)
    ).tolist()
    valSize = int(datasetSize * valSplit)
    valIndices = indices[:valSize]
    trainIndices = indices[valSize:]
    return trainIndices, valIndices


def buildDataLoaders(
    batchSize: int = DataParams.BATCH_SIZE,
    numWorkers: int = DataParams.NUM_WORKERS,
    valSplit: float = DataParams.VAL_SPLIT,
    pinMemory: bool = DataParams.PIN_MEMORY,
    augment: bool = True,
    dataDir: Path = DATASETS_DIR,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders for MNIST.

    Two independent Dataset instances for train and val:
        - trainDataset: full 630k with getTrainTransform (augment=True)
        - valDataset: subset of trainDataset with getTrainTransform (augment=False)
    They share the same underlying raw files but have different transformes.
    Indices split them into disjoint train/val subsets.

    Args:
        batchSize (int): Batch size for DataLoaders.
        numWorkers (int): Number of subprocesses for data loading.
        valSplit (float): Fraction of training data to use for validation.
        pinMemory (bool): Whether to pin memory in DataLoaders.
        augment (bool): Whether to apply data augmentation in the training transform.
        dataDir (Path): Directory where MNIST is stored or will be downloaded.


    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """

    root = str(dataDir) if dataDir else str(DATASETS_DIR)

    # Create two Dataset instances with different transforms
    # Both wrap the same 60k data, but transform differ

    trainFull = MNIST(
        root=root,
        train=True,
        download=False,
        transform=getTrainTransform(augment=augment),
    )
    valFull = MNIST(
        root=root,
        train=True,
        download=False,
        transform=getTrainTransform(augment=False),
    )
    testDataset = MNIST(
        root=root, train=False, download=False, transform=getTestTransform()
    )

    # split indices for train/val, same random split applied to both
    trainIdx, valIdx = _splitIndices(len(trainFull), valSplit)

    trainDataset = Subset(trainFull, trainIdx)  # 54k samples with augmentation
    valDataset = Subset(valFull, valIdx)  # 6k samples without augmentation

    # Build DataLoaders
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=pinMemory,
    )
    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=pinMemory,
    )
    testLoader = DataLoader(
        testDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=pinMemory,
    )

    return trainLoader, valLoader, testLoader


if __name__ == "__main__":
    trainLoader, valLoader, testLoader = buildDataLoaders()
    print(f"train: {len(trainLoader.dataset):,} samples, {len(trainLoader):,} batches")
    print(f"val:   {len(valLoader.dataset):,} samples, {len(valLoader):,} batches")
    print(f"test:  {len(testLoader.dataset):,} samples, {len(testLoader):,} batches")

    # Quick sanity check — grab one batch
    images, labels = next(iter(trainLoader))
    print(f"\nBatch shape: {images.shape}  # (B, C, H, W)")
    print(f"Label shape: {labels.shape}")
    print(f"Label values: {labels[:10].tolist()}")
