"""
src/train/engine.py

Single-epoch training and validation loops with tqdm progress bars.

Usage:
    from src.train.engine import trainEpoch, validateEpoch

    trainLoss, trainAcc = trainEpoch(model, trainLoader, criterion, optimizer, epoch, device)
    valLoss,   valAcc   = validateEpoch(model, valLoader, criterion, epoch, device)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.default_params import DefaultParams


def trainEpoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = DefaultParams.DEVICE,
) -> tuple[float, float]:
    """
    Run a single epoch of training over the full training DataLoader.

    For each batch the function performs the standard training loop:
      forward → loss → backward → optimizer step → zero grad

    Args:
        model:
            The neural network (nn.Module). Must already be on the target
            device. This function sets model.train() internally, which
            enables Dropout, BatchNorm running stats updates, and gradient
            tracking.

        dataloader:
            Training DataLoader. Should have shuffle=True so the model
            sees data in a different order each epoch — this prevents it
            from memorising the sequence and improves generalisation.
            Each iteration yields (images, labels) where:
                images: (batch_size, C, H, W) float32 Tensor
                labels: (batch_size,) int64 Tensor, values 0-9

        criterion:
            Loss function (e.g., nn.CrossEntropyLoss). Takes (logits, labels)
            and returns a scalar loss Tensor. The scalar is detached via
            .item() for logging; the graph is kept alive for .backward().

        optimizer:
            torch.optim.Optimizer instance (e.g., Adam). Manages parameter
            updates. Each batch:
                1. optimizer.zero_grad() — clear previous gradients
                2. loss.backward()       — compute gradients via autograd
                3. optimizer.step()      — update all parameters

        epoch:
            Current epoch number (1-indexed). Only used in the tqdm
            progress bar label — has no effect on the computation.

        device:
            Torch device string: "cuda" or "cpu". Each batch's images and
            labels are moved to this device before being fed to the model.
            The model should already be on this device (done by factory).

    Returns:
        (avg_loss, accuracy):
            avg_loss:  float — average CrossEntropy loss over all samples
            accuracy:  float — fraction of correct predictions in [0.0, 1.0]
    """
    # training mode enables:
    #   - Dropout (randomly zeroes neurons)
    #   - BatchNorm running mean/var updates
    #   - Gradient tracking (requires_grad=True for all params)
    model.train()

    running_loss = 0.0  # accumulated loss * batch_size (for correct averaging)
    correct = 0  # number of correctly predicted samples
    total = 0  # total samples seen so far

    # tqdm wraps the DataLoader, showing a progress bar that updates every batch.
    # desc:   left-side label on the bar
    # leave=True: keep the bar visible after the loop finishes
    progress = tqdm(dataloader, desc=f"Train Epoch {epoch:>3d}", leave=True)

    for images, labels in progress:
        # Move data to the same device as the model.
        # If already on the correct device, .to() is a cheap no-op.
        images = images.to(device)  # (B, 1, 28, 28)
        labels = labels.to(device)  # (B,)

        # ---- Forward pass ----
        # model(images) calls model.forward(images).
        # Output: (B, 10) raw logits — one score per class per sample.
        logits = model(images)

        # criterion(logits, labels) computes the scalar loss.
        # For CrossEntropyLoss: loss = -log(softmax(logits)[true_class]).
        # Result is a 0-dim Tensor; loss.item() gives a Python float.
        loss = criterion(logits, labels)

        # ---- Backward pass ----
        # zero_grad() must be called BEFORE backward, otherwise gradients
        # from the previous batch would accumulate on top of the new ones.
        optimizer.zero_grad()

        # loss.backward() runs autograd: traverses the computation graph
        # backwards and computes ∂loss/∂param for every parameter.
        loss.backward()

        # optimizer.step() applies the update rule.
        # For Adam: param = param - lr * m_hat / (sqrt(v_hat) + eps).
        optimizer.step()

        # ---- Accumulate statistics ----
        batch_size = images.size(0)  # number of samples in this batch

        # loss.item() is the *average* loss over the batch (CrossEntropyLoss
        # default reduction='mean'). Multiply by batch_size to get the sum,
        # so the final average = total_sum / total_samples is correct even
        # when the last batch has a different size.
        running_loss += loss.item() * batch_size

        # logits.max(dim=1) returns (values, indices).
        # indices: (B,) — predicted class for each sample (0-9).
        _, predicted = logits.max(dim=1)

        # predicted.eq(labels) → BoolTensor of shape (B,).
        # .sum().item() counts how many True values → number of correct predictions.
        correct += predicted.eq(labels).sum().item()
        total += batch_size

        # Update the right side of the tqdm bar with live statistics
        progress.set_postfix(
            {
                "loss": f"{running_loss / total:.4f}",
                "acc": f"{correct / total:.4f}",
            }
        )

    # End of epoch: compute final averages over the entire training set
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validateEpoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    device: str = DefaultParams.DEVICE,
) -> tuple[float, float]:
    """
    Run a single epoch of validation over the full validation DataLoader.

    Unlike trainEpoch, this function does NOT compute gradients or update
    model parameters. It is read-only evaluation.

    Args:
        model:
            The neural network in eval mode. model.eval() disables Dropout
            (all neurons active, no zeroing) and freezes BatchNorm statistics
            (uses running averages, not batch statistics).

        dataloader:
            Validation DataLoader. Should have shuffle=False — we only
            evaluate, no need to randomise the order.

        criterion:
            Loss function — same one used in training. Consistent loss
            semantics allow direct comparison between train and val curves.

        epoch:
            Current epoch number, only used for the tqdm label.

        device:
            Torch device string: "cuda" or "cpu".

    Returns:
        (avg_loss, accuracy):
            avg_loss:  float — average validation loss over all samples.
            accuracy:  float — fraction of correct predictions in [0.0, 1.0].
    """
    # eval mode disables:
    #   - Dropout (all connections active, no random zeroing)
    #   - BatchNorm updates (running mean/var frozen)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc=f"Val   Epoch {epoch:>3d}", leave=True)

    # torch.no_grad() disables autograd graph construction for everything
    # inside this block. Benefits:
    #   - No intermediate activations stored → significant memory savings.
    #   - No gradient computation → faster forward pass.
    # Required for validation because we don't call backward().
    with torch.no_grad():
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass only — no backward, no optimizer
            logits = model(images)
            loss = criterion(logits, labels)

            # Statistics — identical logic to trainEpoch
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = logits.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += batch_size

            progress.set_postfix(
                {
                    "loss": f"{running_loss / total:.4f}",
                    "acc": f"{correct / total:.4f}",
                }
            )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
