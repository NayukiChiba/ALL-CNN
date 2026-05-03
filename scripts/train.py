"""
scripts/train.py

Full training pipeline for MNIST-CNN.

Orchestrates: DataLoaders → Model → Optimizer + Scheduler + Loss →
Training loop → Checkpoint saving → Logging.

Called by main.py after argument parsing. No CLI parsing here.

Usage (via main.py):
    python main.py train --epochs 20 --lr 0.001 --device cuda
    python main.py train --resume checkpoints/last_model.pth
"""

import argparse
from pathlib import Path

from src.data.dataloader import buildDataLoaders
from src.model.factory import createModel
from src.train.checkpoint import loadCheckpoint, saveCheckpoint
from src.train.engine import trainEpoch, validateEpoch
from src.train.logger import TrainLogger
from src.train.loss import createLossFunction
from src.train.optimizer import createOptimizer, createScheduler


def run(args: argparse.Namespace) -> None:
    """
    Run the full training pipeline.

    Args:
        args: argparse.Namespace with all training settings. Expected fields:
              device, epochs, batch_size, num_workers, val_split, no_augment,
              lr, weight_decay, lr_factor, lr_patience, lr_min,
              conv_channels, fc_hidden_size, dropout_rate,
              data_dir, checkpoint_dir, log_dir, output_dir,
              resume, seed.
    """
    device = args.device

    print("=" * 60)
    print("MNIST-CNN Training")
    print(f"  Device:        {device}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Data dir:      {args.data_dir}")
    print("=" * 60)

    # ---- 1. DataLoaders ----
    train_loader, val_loader, _test_loader = buildDataLoaders(
        batchSize=args.batch_size,
        numWorkers=args.num_workers,
        valSplit=args.val_split,
        augment=not args.no_augment,
        dataDir=Path(args.data_dir),
    )
    print(
        f"Train samples: {len(train_loader.dataset):,} | batches: {len(train_loader)}"
    )
    print(f"Val samples:   {len(val_loader.dataset):,} | batches: {len(val_loader)}")

    # ---- 2. Model ----
    model = createModel(
        conv_channels=args.conv_channels,
        hidden_size=args.fc_hidden_size,
        dropout=args.dropout_rate,
        device=device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model params: {total_params:,} total | "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable"
    )

    # ---- 3. Optimizer + Scheduler + Loss ----
    criterion = createLossFunction()
    optimizer = createOptimizer(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = createScheduler(
        optimizer,
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )

    # ---- 4. Resume from checkpoint (optional) ----
    start_epoch = 0
    best_val_accuracy = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    if args.resume:
        resume_epoch, resume_metrics = loadCheckpoint(
            args.resume, model, optimizer, device=device
        )
        start_epoch = resume_epoch
        best_val_accuracy = resume_metrics.get("val_acc", 0.0)
        print(
            f"Resumed from epoch {start_epoch}, best val acc: {best_val_accuracy:.4f}"
        )

    # ---- 5. Logger ----
    logger = TrainLogger(
        log_dir=Path(args.log_dir),
        tensorboard_dir=Path(args.output_dir) / "tensorboard",
        resume=args.resume is not None,
    )
    logger.logConfig(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "conv_channels": args.conv_channels,
            "fc_hidden_size": args.fc_hidden_size,
            "dropout": args.dropout_rate,
            "lr_factor": args.lr_factor,
            "lr_patience": args.lr_patience,
            "augment": not args.no_augment,
            "device": device,
            "seed": args.seed,
        }
    )

    # Log model graph to TensorBoard
    sample_images, _ = next(iter(train_loader))
    logger.logGraph(model, sample_images[:1].to(device))

    # ---- 6. Training loop ----
    print("\nStarting training...\n")

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_loss, train_acc = trainEpoch(
            model, train_loader, criterion, optimizer, epoch, device
        )
        val_loss, val_acc = validateEpoch(model, val_loader, criterion, epoch, device)

        # Scheduler monitors val_loss — call after each epoch
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Log
        logger.logEpoch(epoch, train_loss, train_acc, val_loss, val_acc, lr=current_lr)

        # Track history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Always save last checkpoint
        saveCheckpoint(
            model,
            optimizer,
            epoch,
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            Path(args.checkpoint_dir) / "last_model.pth",
        )

        # Save best checkpoint (by val accuracy)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            logger.logBest("val accuracy", val_acc, epoch)
            saveCheckpoint(
                model,
                optimizer,
                epoch,
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                Path(args.checkpoint_dir) / "best_model.pth",
            )

    # ---- 7. Save training history as JSON for later visualization ----
    import json

    history_path = Path(args.log_dir) / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Training history saved → {history_path}")

    # ---- 8. Summary ----
    best_epoch = history["val_acc"].index(best_val_accuracy) + start_epoch + 1
    logger.logSummary(best_epoch, best_val_accuracy)
    logger.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best val accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")
    print(f"  Best model:        {Path(args.checkpoint_dir) / 'best_model.pth'}")
    print(f"  Last model:        {Path(args.checkpoint_dir) / 'last_model.pth'}")
    print("=" * 60)
