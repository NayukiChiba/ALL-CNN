"""
src/model/cnn.py

MNIST CNN model assembled from ConvBlock and LinearBlock.

Architecture:
    Input (B, 1, 28, 28)
      → ConvBlock(in_channels=1,  out_channels=32, pool=True)  → (B, 32, 14, 14)
      → ConvBlock(in_channels=32, out_channels=64, pool=True)  → (B, 64, 7, 7)
      → Flatten                                                → (B, 3136)
      → LinearBlock(in_features=3136, out_features=128)        → (B, 128)
      → Linear(in_features=128, out_features=10)               → (B, 10)

Usage:
    from src.model.cnn import MNISTCNN
    model = MNISTCNN()
    logits = model(images)
"""

import torch
import torch.nn as nn

from src.model.layers import ConvBlock, LinearBlock


class MNISTCNN(nn.Module):
    """
    A simple CNN for MNIST digit classification

    Convolutional parts: extracts spatial features, compresses
    (1. 28, 28) -> (32. 14, 14) -> (64. 7, 7) through two ConvBlocks with max pooling

    Classifier part: flattens then runs through one LinearBlock with dropout
    and a final Linear layer to output logits for 10 classes


    """

    def __init__(
        self,
        conv_channels: list[int] = None,  # If None, defaults to [32, 64]
        fc_hidden_size: int = 128,
        dropout: float = 0.5,
    ):
        """
        Args:
            conv_channels:
                List of output channels for each ConvBlock.
                Default [32, 64] → two blocks: 1→32→64.
                Adding more entries (e.g., [32, 64, 128]) deepens the network.
            fc_hidden_size:
                Number of units in the hidden FC layer (after flatten).
            dropout:
                Dropout probability for the LinearBlock. None = disabled.



        """
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]

        # Convolutional blocks
        in_channels = 1  # MNIST images have 1 channel
        self.conv_blocks = nn.ModuleList()
        for out_channels in conv_channels:
            self.conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    pool=True,
                )
            )
            in_channels = out_channels  # For next block
        self.flatten = nn.Flatten()

        # Flattened dimension after L ConvBlocks: last_out_channels * (28 / 2^L)^2
        pooled_size = 28 // (2 ** len(conv_channels))  # Each block halves spatial dims
        flattened_size = conv_channels[-1] * (pooled_size**2)

        # Classifier head
        self.fc_block = LinearBlock(
            in_features=flattened_size, out_features=fc_hidden_size, dropout=dropout
        )
        self.classifier = nn.Linear(in_features=fc_hidden_size, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN

        Args:
            x: Input tensor of shape (B, 1, 28, 28)

        Returns:
            Logits tensor of shape (B, 10)
        """
        for block in self.conv_blocks:
            x = block(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    model = MNISTCNN()
    print(model)
    print("=" * 50)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print("=" * 50)

    # Dummy forward pass
    x = torch.randn(4, 1, 28, 28)  # Batch of 4 images
    with torch.no_grad():
        logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"predicted class indices: {logits.argmax(dim=1)}")
