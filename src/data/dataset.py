"""
src/data/dataset.py

MNIST Dataset wrapper with auto-download

Usage:
    dataset = MNISTDataset(train=True, transform=myTransform)
    image, label = dataset[0]

"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from config.paths import DATASETS_DIR


class MNISTDataset(Dataset):
    """
    Thin wrapper around torchvision.datasets.MNIST with auto-download to datasets/

    """

    def __init__(
        self,
        train: bool = True,  # Whether to load the training set (True) or test set (False)
        transform: Optional[
            Callable
        ] = None,  # Optional transform to apply to the images
        download: bool = True,  # Whether to download the dataset if not found
        root: Optional[
            Path
        ] = None,  # Optional root directory to store the dataset (defaults to DATASETS_DIR)
    ) -> None:
        """
        Args:
            train(bool): Whether to load the training set (True) or test set (False)
            transform(Optional[Callable]): torchvision transform to apply to the images (e.g. ToTensor, Normalize)
            download(bool): Whether to download the dataset if not found
            root(Optional[Path]): Optional root directory to store the dataset (defaults to DATASETS_DIR)


        """
        super().__init__()
        self.train = train
        self.root = root or DATASETS_DIR
        self.dataset = MNIST(
            root=str(self.root),
            train=self.train,
            transform=transform,
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        return image, label


if __name__ == "__main__":
    print("Downloading MNIST dataset...")
    trainDs = MNISTDataset(train=True, download=True)
    testDs = MNISTDataset(train=False, download=True)
    print(f"Train set: {len(trainDs)} images")
    print(f"Test set:  {len(testDs)} images")
    print(f"Files saved to: {DATASETS_DIR}")
