"""
scripts/eval.py

Evaluation pipeline for a trained MNIST-CNN checkpoint.

Orchestrates: DataLoader → Load checkpoint → Compute metrics →
Print report → Generate visualizations.

Called by main.py after argument parsing. No CLI parsing here.

Usage (via main.py):
    python main.py eval --checkpoint checkpoints/best_model.pth --device cuda
    python main.py eval --checkpoint checkpoints/best_model.pth --no-visualize
"""

import argparse
import json
from pathlib import Path

from config.paths import VISUALIZATIONS_DIR
from src.data.dataloader import buildDataLoaders
from src.eval.metrics import evaluateModel, formatReport
from src.eval.visualize import (
    gatherErrorSamples,
    plotConfusionMatrix,
    plotErrorGrid,
    plotTrainingCurves,
)
from src.model.factory import createModel
from src.train.checkpoint import loadCheckpoint


def run(args: argparse.Namespace) -> None:
    """
    Run the full evaluation pipeline.

    Args:
        args: argparse.Namespace with eval settings. Expected fields:
              checkpoint, device, batch_size, num_workers, val_split,
              data_dir, output_dir, no_visualize.
    """
    device = args.device
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print("=" * 60)
    print("MNIST-CNN Evaluation")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device}")
    print("=" * 60)

    # ---- 1. Test DataLoader ----
    _train_loader, _val_loader, test_loader = buildDataLoaders(
        batchSize=args.batch_size,
        numWorkers=args.num_workers,
        valSplit=args.val_split,
        augment=False,
        dataDir=Path(args.data_dir),
    )
    print(f"Test samples: {len(test_loader.dataset):,} | batches: {len(test_loader)}")

    # ---- 2. Load model from checkpoint ----
    model = createModel(device=device)
    epoch, metrics_from_checkpoint = loadCheckpoint(
        checkpoint_path, model, optimizer=None, device=device
    )

    print("\nCheckpoint info:")
    print(f"  Saved at epoch: {epoch}")
    if metrics_from_checkpoint:
        for key, value in metrics_from_checkpoint.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # ---- 3. Evaluate ----
    print("\nRunning evaluation on test set...")
    result = evaluateModel(model, test_loader, criterion=None, device=device)

    # ---- 4. Print results ----
    print("\n" + formatReport(result["report"]))
    print(f"\nConfusion Matrix:\n{result['confusion_matrix']}")

    # Save evaluation results as JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "eval_results.json"

    serializable = {
        "checkpoint": str(checkpoint_path),
        "accuracy": result["accuracy"],
        "num_samples": result["num_samples"],
        "per_class": result["report"]["per_class"],
        "macro_avg": result["report"]["macro_avg"],
        "confusion_matrix": result["confusion_matrix"].tolist(),
    }
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {results_file}")

    # ---- 5. Visualizations ----
    if args.no_visualize:
        print("\nVisualization skipped (--no-visualize)")
    else:
        print("\nGenerating visualizations...")
        visualizations_dir = VISUALIZATIONS_DIR
        visualizations_dir.mkdir(parents=True, exist_ok=True)

        # Training curves (if history file exists)
        history_path = Path(args.log_dir) / "training_history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            curves_path = visualizations_dir / "training_curves.png"
            plotTrainingCurves(history, save_path=curves_path)
            print(f"Training curves → {curves_path}")
        else:
            print(f"No training history found at {history_path} — skipping curves")

        # Confusion matrix heatmap
        cm_path = visualizations_dir / "confusion_matrix.png"
        plotConfusionMatrix(result["confusion_matrix"], save_path=cm_path)
        print(f"Confusion matrix → {cm_path}")

        # Error grid
        error_images, error_true_labels, error_predicted_labels = gatherErrorSamples(
            model, test_loader, device=device
        )
        if error_images.size(0) > 0:
            error_path = visualizations_dir / "error_grid.png"
            plotErrorGrid(
                error_images,
                error_true_labels,
                error_predicted_labels,
                save_path=error_path,
            )
            print(f"Error grid → {error_path}")
        else:
            print("No misclassified samples — skipping error grid")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"  Test accuracy: {result['accuracy']:.4f}")
    print("=" * 60)
