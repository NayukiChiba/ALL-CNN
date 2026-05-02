"""
scripts/infer.py

Single-image or batch inference for a trained MNIST-CNN checkpoint.

Orchestrates: Load Predictor → Preprocess → Forward pass → Print top-K.

Called by main.py after argument parsing. No CLI parsing here.

Usage (via main.py):
    python main.py infer --image my_digit.png --checkpoint checkpoints/best_model.pth
    python main.py infer --image-dir ./digits/ --checkpoint checkpoints/best_model.pth
"""

import argparse
from pathlib import Path

from src.inference.predictor import Predictor


def run(args: argparse.Namespace) -> None:
    """
    Run inference on a single image or a directory of images.

    Args:
        args: argparse.Namespace with infer settings. Expected fields:
              checkpoint, device, image, image_dir, top_k,
              conv_channels, fc_hidden_size, dropout_rate.
    """
    device = args.device

    if not args.image and not args.image_dir:
        print("Error: one of --image or --image-dir is required")
        return

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print("=" * 60)
    print("MNIST-CNN Inference")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device}")
    print("=" * 60)

    # ---- Load predictor ----
    predictor = Predictor(
        checkpoint_path=checkpoint_path,
        device=device,
        conv_channels=args.conv_channels,
        hidden_size=args.fc_hidden_size,
        dropout=args.dropout_rate,
    )
    print(f"Model loaded (trained for {predictor.loaded_epoch} epochs)\n")

    # ---- Single image ----
    if args.image:
        _predictSingle(predictor, args.image, args.top_k)

    # ---- Batch directory ----
    if args.image_dir:
        _predictBatch(predictor, args.image_dir, args.top_k)

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


def _predictSingle(predictor: Predictor, image_path: str | Path, top_k: int) -> None:
    """Predict and display results for a single image."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    print(f"Input: {image_path}")
    result = predictor.predict(image_path, top_k=top_k)

    print("-" * 40)
    print(f"Predicted: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence'] * 100:.1f}%)")
    print("-" * 40)
    print(f"Top-{top_k} predictions:")
    for rank, entry in enumerate(result["top_k"], start=1):
        bar = "█" * int(entry["confidence"] * 40)
        print(f"  {rank}.  {entry['class_name']}  {entry['confidence']:.4f}  {bar}")
    print("-" * 40)

    print("\nFull probability distribution:")
    probabilities = result["probabilities"]
    for digit in range(10):
        print(f"  {digit}: {probabilities[digit].item():.4f}")


def _predictBatch(predictor: Predictor, image_dir: str | Path, top_k: int) -> None:
    """Predict and display results for all images in a directory."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        print(f"Directory not found: {image_dir}")
        return

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions]
    )

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images in {image_dir}\n")
    print("-" * 60)

    results = predictor.predictBatch(image_paths, top_k=top_k)

    correct_count = 0
    total_count = 0

    for image_path, result in zip(image_paths, results):
        print(
            f"{image_path.name:<24s} → {result['class_name']}  "
            f"({result['confidence']:.4f})"
        )
        # If filename starts with a digit, treat it as ground truth
        filename = image_path.stem
        if filename and filename[0].isdigit():
            if filename[0] == result["class_name"]:
                correct_count += 1
            total_count += 1

    print("-" * 60)

    if total_count > 0:
        accuracy = correct_count / total_count
        print(
            f"\nAccuracy (by filename prefix): {correct_count}/{total_count} = {accuracy:.4f}"
        )
    else:
        print(f"\nProcessed {len(image_paths)} images")
