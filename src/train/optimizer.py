"""
src/train/optimizer.py

Adam optimizer and ReduceLROnPlateau scheduler for MNIST-CNN.

Usage:
    from src.train.optimizer import createOptimizer, createScheduler
    optimizer = createOptimizer(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = createScheduler(optimizer, factor=0.5, patience=3, min_lr=1e-6)

    # in training loop:
    scheduler.step(val_loss)   # call after each epoch
"""

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def createOptimizer(
    parameters,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
) -> optim.Adam:
    """
    Create Adam optimizer.

    Adam (Adaptive Moment Estimation) combines momentum (first moment)
    and RMSprop-style scaling (second moment) to adapt per-parameter
    learning rates. It is the default choice for most CNN training.

    Args:
        parameters: model.parameters() or a filtered param group.
        lr:          Initial learning rate. Default 0.001 (common Adam default).
        weight_decay: L2 regularization coefficient. Default 1e-4.
                      Applied as decoupled weight decay (AdamW-style in
                      recent PyTorch — use `fused=False` for strict AdamW,
                      or just use the standard `weight_decay` kwarg which
                      PyTorch Adam now supports natively).

    Returns:
        Adam optimizer instance.
    """
    optimizer = optim.Adam(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


def createScheduler(
    optimizer: optim.Optimizer,
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 1e-6,
) -> lr_scheduler.ReduceLROnPlateau:
    """
    Create ReduceLROnPlateau learning rate scheduler.

    How it works:
      - After each epoch, call scheduler.step(val_loss).
      - If val_loss has not improved for `patience` consecutive epochs,
        lr = lr * factor (e.g., 0.001 → 0.0005 → 0.00025 → ...).
      - Stops reducing when lr reaches `min_lr`.

    Why use it:
      - When training plateaus, a smaller lr can settle into a better minimum.
      - Avoids manual lr tuning mid-training.

    Args:
        optimizer: Adam optimizer to schedule.
        factor:    Multiplicative factor for lr reduction. Default 0.5.
        patience:  Number of epochs with no improvement before reducing lr.
        min_lr:    Lower bound on learning rate. No reduction below this.

    Returns:
        ReduceLROnPlateau scheduler instance.
    """
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # monitor val_loss — lower is better
        factor=factor,  # lr = lr * factor on plateau
        patience=patience,  # wait this many stagnant epochs
        min_lr=min_lr,  # floor
    )
    return scheduler
