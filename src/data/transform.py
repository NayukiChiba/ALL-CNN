"""
src/data/transform.py

Preprocessing and augmentation transforms for MNIST

Pipelines:
    getTrainTransform: random rotation, random affine, ToTensor, Normalize
    getTestTransform: ToTensor, Normalize

Usage:
    from src.data.transform import getTrainTransform
    dataset = MNISTDataset(train=True, transform=getTrainTransform())

"""

from torchvision import transforms

from config.default_params import DataParams


def getTrainTransform(augment: bool = True) -> transforms.Compose:
    """
    Training pipeline: optional augmentation -> ToTensor -> Normalize

    Args:
        augment(bool):
            - Positive and Negative Random rotation
            - Positive and Negative Random affine

    Returns:
        transforms.Compose: Training pipeline
    """
    pipeline = []
    if augment:
        pipeline.append(
            transforms.RandomAffine(
                degrees=DataParams.ROTATION_DEGREES,
                translate=(DataParams.TRANSLATION_RATIO, DataParams.TRANSLATION_RATIO),
            )
        )

    # PIL.Image -> Tensor and normalize to [0, 1]
    pipeline.append(transforms.ToTensor())
    pipeline.append(
        transforms.Normalize(mean=DataParams.MNIST_MEAN, std=DataParams.MNIST_STD)
    )
    return transforms.Compose(pipeline)


def getTestTransform() -> transforms.Compose:
    """
    Validation / Test pipeline: ToTensor -> Normalize only
    No augmentation — evaluation must be deterministic.

    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=DataParams.MNIST_MEAN, std=DataParams.MNIST_STD),
        ]
    )


if __name__ == "__main__":
    from src.data.dataset import MNISTDataset

    # Load one raw image (no transform) — returns PIL.Image
    rawDs = MNISTDataset(train=True, transform=None)
    pilImg, label = rawDs[0]

    print("=== Raw image (PIL.Image, no transform) ===")
    print(f"Type:  {type(pilImg)}")
    print(f"Size:  {pilImg.size}          # (W, H)")
    print(f"Mode:  {pilImg.mode}          # L = grayscale")
    print(
        f"Pixel: [{pilImg.getpixel((0, 0))}, ..., {pilImg.getpixel((27, 27))}]  # uint8 [0-255]"
    )
    print(f"Label: {label}")

    # Apply train transform (with augmentation) — returns normalized Tensor
    print("\n=== After train transform (augment=True) ===")
    dsAug = MNISTDataset(train=True, transform=getTrainTransform(augment=True))
    tensorAug, _ = dsAug[0]
    print(f"Type:   {type(tensorAug)}")
    print(f"Shape:  {tensorAug.shape}    # (C, H, W)")
    print(f"dtype:  {tensorAug.dtype}")
    print(f"Min:    {tensorAug.min().item():.4f}")
    print(f"Max:    {tensorAug.max().item():.4f}")
    print(f"Mean:   {tensorAug.mean().item():.4f}")

    # Apply test transform (no augmentation) — returns normalized Tensor
    print("\n=== After test transform ===")
    dsTest = MNISTDataset(train=True, transform=getTestTransform())
    tensorTest, _ = dsTest[0]
    print(f"Type:   {type(tensorTest)}")
    print(f"Shape:  {tensorTest.shape}")
    print(f"Min:    {tensorTest.min().item():.4f}")
    print(f"Max:    {tensorTest.max().item():.4f}")
    print(f"Mean:   {tensorTest.mean().item():.4f}")
