"""
src/model/layers.py

Reusable building blocks for CNN models. Each block packages a standard
layer sequence so the main model can be written as a clean stack of blocks
instead of dozens of individual nn.Module lines.

Classes:
    ConvBlock   — Conv2d → BatchNorm2d → ReLU → [MaxPool2d]
    LinearBlock — Linear → BatchNorm1d → ReLU → [Dropout]

Usage:
    from src.model.layers import ConvBlock, LinearBlock
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv2d → BatchNorm2d → ReLU → optional MaxPool2d.

    Shape transformation (with kernel_size=3, pool=True):
        Input:  (B, C_in, H,     W)
        Conv:   (B, C_out, H,     W)    # "same" padding preserves H,W
        BN:     (B, C_out, H,     W)    # normalized per-channel
        ReLU:   (B, C_out, H,     W)    # non-linearity, zeros out negatives
        MaxPool:(B, C_out, H//2, W//2)  # spatial downsampling

    Why this exact sequence:
        - Conv BEFORE BN: BN normalizes the convolution output, which makes
          training more stable by preventing activation drift.
        - BN BEFORE ReLU: normalized values →0 mean, unit variance→ land in
          ReLU's non-saturating region, reducing dead neurons.
        - ReLU BEFORE Pool: apply non-linearity first so the max operation
          selects the most activated features, not the most activated raw sums.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
    ) -> None:
        """
        Args:
            in_channels:
                Number of input channels from the previous layer. For MNIST:
                - First ConvBlock:  in_channels=1  (grayscale image)
                - Second ConvBlock: in_channels=32 (output of first block)
                - Deeper ConvBlock: in_channels=64, 128, ...

            out_channels:
                Number of output channels — i.e. how many different feature
                maps this layer learns. Each channel captures one type of
                pattern (edges, textures, curves, ...).
                Common heuristics: double channels each block (32→64→128).

            kernel_size:
                Size of the square convolution kernel. Default 3 means a
                3×3 sliding window.
                - 3×3: the standard choice. Two stacked 3×3 convs have the
                  same receptive field as one 5×5, but with fewer parameters
                  and one extra non-linearity. This is the VGG finding (2014).
                - 5×5: used when you want a larger receptive field per layer.
                - Odd values allow symmetric 'same' padding (kernel_size // 2).

            pool:
                If True (default), append a 2×2 MaxPool2d after ReLU.
                - MaxPool with kernel_size=2, stride=2 halves H and W.
                - This reduces spatial resolution → fewer computations in
                  deeper layers.
                - It also provides local translation invariance: shifting the
                  input by 1-2 pixels does not change the pooled output.
                - Set pool=False when you want to preserve spatial size
                  (e.g., the last conv layer before flattening, or if you
                  want to stack multiple convs before pooling).
        """
        super().__init__()

        # ---- Conv2d ----
        # kernel_size // 2 gives symmetric 'same' padding for odd kernels:
        #   k=3 → pad 1   k=5 → pad 2   k=7 → pad 3
        # Without padding, a 3×3 conv shrinks H,W by 2 pixels each.
        # With same padding, H,W are preserved — only MaxPool changes them.
        # stride defaults to 1, dilation defaults to 1.
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # "same" padding
        )

        # ---- BatchNorm2d ----
        # For each channel, normalizes the activations over the batch:
        #   x_norm = (x - mean_batch) / sqrt(var_batch + eps) * gamma + beta
        # gamma and beta are learned parameters (scale and shift).
        # Benefits:
        #   - Stabilizes training (activations stay in a controlled range).
        #   - Allows higher learning rates without divergence.
        #   - Acts as a mild regularizer (the batch statistics are noisy).
        # 2d variant operates per-channel over (B, C, H, W) tensors.
        self.bn = nn.BatchNorm2d(out_channels)

        # ---- ReLU ----
        # f(x) = max(0, x)
        # Simplest effective non-linearity:
        #   - No upper bound → gradients don't saturate for positive values.
        #   - Computationally cheap (a comparison, not a function evaluation).
        #   - Sparse activation: ~50% of neurons output 0, which helps
        #     the network form disentangled representations.
        # inplace=True: modifies the tensor in memory instead of allocating
        # a new one. Saves GPU memory. Safe because the pre-ReLU value is
        # never needed after this point.
        self.relu = nn.ReLU(inplace=True)

        # ---- MaxPool2d (optional) ----
        # kernel_size=2, stride defaults to kernel_size → non-overlapping 2×2 windows.
        # For each 2×2 patch, takes the maximum value.
        # This does NOT have learnable parameters — it's a fixed operation.
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape:
                (B, out_channels, H,   W) if pool=False
                (B, out_channels, H//2, W//2) if pool=True (default)
        """
        x = self.conv(x)  # (B, C_out, H, W) — spatial features extracted
        x = self.bn(x)  # (B, C_out, H, W) — per-channel normalization
        x = self.relu(x)  # (B, C_out, H, W) — non-linearity
        if self.pool is not None:
            x = self.pool(x)  # (B, C_out, H//2, W//2) — spatial downsampling
        return x


class LinearBlock(nn.Module):
    """
    Fully-connected block: Linear → BatchNorm1d → ReLU → optional Dropout.

    Shape transformation (with out_features=128, dropout=0.5):
        Input:    (B, in_features)
        Linear:   (B, out_features)
        BN:       (B, out_features)    # normalized per-feature
        ReLU:     (B, out_features)    # non-linearity
        Dropout:  (B, out_features)    # randomly zeros some activations (train only)

    Typical usage in a CNN: after Flatten() converts the conv feature maps
    into a vector, stack 1-2 LinearBlocks as the "classifier head" before
    the final Linear(..., num_classes) output layer.

    Why BN1d BEFORE ReLU, same reasoning as ConvBlock — normalized values
    land in ReLU's active region.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float | None = 0.5,
    ) -> None:
        """
        Args:
            in_features:
                Dimension of the input vector. For MNIST with our architecture:
                - First LinearBlock:  in_features = 64*7*7 = 3136
                  (after ConvBlock×2 and Flatten).
                - Second LinearBlock: in_features = 128 (output of first block).

            out_features:
                Dimension of the output vector — the "hidden size" of the
                classifier head.
                - Larger → more capacity, more risk of overfitting.
                - Smaller → bottleneck, information loss.
                - 128 is a reasonable default for MNIST (simple task).

            dropout:
                Probability of randomly zeroing a neuron during training.
                - 0.5: each neuron has a 50% chance of being dropped per forward
                  pass. This forces the network to not rely on any single neuron,
                  acting as ensemble regularization.
                - None or 0.0: no dropout — used when you want all connections
                  active (e.g., the last Linear layer before output).
                - During eval() mode, dropout is automatically disabled.
        """
        super().__init__()

        # ---- Linear ----
        # y = xW^T + b
        # W shape: (out_features, in_features), b shape: (out_features,)
        # in_features determines the weight matrix column count.
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

        # ---- BatchNorm1d ----
        # Normalizes over the batch dimension for 2D inputs (B, F).
        # 1d variant operates per-feature: each of the out_features dimensions
        # gets its own mean/std estimate.
        # Stabilizes training in the FC layers just as BN2d does in conv layers.
        self.bn = nn.BatchNorm1d(out_features)

        # ---- ReLU ----
        # Same as ConvBlock — introduces non-linearity between FC layers.
        # Without this, stacking two Linear layers is equivalent to a single
        # Linear layer (composition of affine maps is affine).
        self.relu = nn.ReLU(inplace=True)

        # ---- Dropout (optional) ----
        # During training, each element is independently set to 0 with
        # probability 'dropout', and surviving elements are scaled by
        # 1/(1-dropout) to keep the expected sum constant.
        # During eval(), Dropout is a no-op.
        if dropout and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block.

        Args:
            x: Input tensor of shape (B, in_features).

        Returns:
            Output tensor of shape (B, out_features).
        """
        x = self.linear(x)  # (B, out_features) — affine transform
        x = self.bn(x)  # (B, out_features) — per-feature normalization
        x = self.relu(x)  # (B, out_features) — non-linearity
        if self.dropout is not None:
            x = self.dropout(x)  # (B, out_features) — randomly zeroed (train only)
        return x


if __name__ == "__main__":
    # Create the two blocks with MNIST-typical parameters
    conv = ConvBlock(in_channels=1, out_channels=32, kernel_size=3, pool=True)
    linear = LinearBlock(in_features=3136, out_features=128, dropout=0.5)

    print("=" * 60)
    print("ConvBlock(in_channels=1, out_channels=32, kernel_size=3, pool=True)")
    print(conv)
    print()

    # Show each sub-module
    for name, module in conv.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:8s} — params: {params:>6,d}  {module}")

    print()
    print("=" * 60)
    print("LinearBlock(in_features=3136, out_features=128, dropout=0.5)")
    print(linear)
    print()

    for name, module in linear.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:8s} — params: {params:>6,d}  {module}")

    # Total parameters
    convParams = sum(p.numel() for p in conv.parameters())
    linearParams = sum(p.numel() for p in linear.parameters())
    print()
    print(f"  ConvBlock total params:  {convParams:>8,d}")
    print(f"  LinearBlock total params: {linearParams:>8,d}")
