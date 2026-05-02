"""
src/train/loss.py

Loss function wrapper for MNIST classification.

Usage:
    from src.train.loss import createLoss
    criterion = createLossFunction()
    loss = criterion(logits, labels)
"""

import torch.nn as nn


def createLossFunction() -> nn.CrossEntropyLoss:
    """
    Create the loss function for multi-class classification.

    CrossEntropyLoss combines log_softmax + NLLLoss in one operation.
    It expects raw logits (not softmax probabilities) and integer labels.

    Returns:
        nn.CrossEntropyLoss with default reduction='mean'.
    """
    criterion = nn.CrossEntropyLoss()
    return criterion
