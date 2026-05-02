"""
Unit tests for src/model/layers.py — ConvBlock and LinearBlock.
"""

import pytest
import torch

from src.model.layers import ConvBlock, LinearBlock

# ================================================================
# ConvBlock
# ================================================================


class TestConvBlock:
    """Tests for the ConvBlock building block."""

    @pytest.mark.parametrize("pool", [True, False])
    def test_output_shape_with_pool(self, pool: bool):
        """Spatial dims are halved when pool=True, preserved when pool=False."""
        block = ConvBlock(in_channels=1, out_channels=32, kernel_size=3, pool=pool)
        x = torch.randn(4, 1, 28, 28)  # (B, C, H, W)
        out = block(x)

        expected_hw = 14 if pool else 28
        assert out.shape == (4, 32, expected_hw, expected_hw)

    def test_different_in_channels(self):
        """Output channel count matches the constructor argument."""
        block = ConvBlock(in_channels=3, out_channels=16, kernel_size=3, pool=True)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 16, 16, 16)

    def test_kernel_size_5_preserves_spatial(self):
        """k=5 with same padding (pad=2) also preserves H,W before pooling."""
        block = ConvBlock(in_channels=1, out_channels=8, kernel_size=5, pool=False)
        x = torch.randn(2, 1, 28, 28)
        out = block(x)
        assert out.shape == (2, 8, 28, 28)

    def test_kernel_size_7(self):
        """k=7 with same padding (pad=3)."""
        block = ConvBlock(in_channels=1, out_channels=4, kernel_size=7, pool=True)
        x = torch.randn(1, 1, 14, 14)
        out = block(x)
        assert out.shape == (1, 4, 7, 7)

    def test_parameter_count(self):
        """Conv2d: (in*out*k*k) weights + out biases. BN: out*2 (gamma+beta)."""
        block = ConvBlock(in_channels=3, out_channels=16, kernel_size=3, pool=True)

        # Conv2d: 3*16*3*3 = 432 weights + 16 biases = 448
        # BatchNorm2d: 16 + 16 = 32 (weight + bias)
        # ReLU: 0
        # MaxPool2d: 0
        total = sum(p.numel() for p in block.parameters())
        expected = (3 * 16 * 3 * 3) + 16 + 16 + 16  # conv_w + conv_b + bn_w + bn_b
        assert total == expected

    def test_train_eval_mode(self):
        """Forward pass works in both train and eval mode."""
        block = ConvBlock(1, 8, pool=True)
        x = torch.randn(2, 1, 28, 28)

        block.train()
        out_train = block(x)
        assert out_train.shape == (2, 8, 14, 14)

        block.eval()
        out_eval = block(x)
        assert out_eval.shape == (2, 8, 14, 14)

    def test_empty_pool_returns_none(self):
        """When pool=False, self.pool is None."""
        block = ConvBlock(1, 8, pool=False)
        assert block.pool is None

    def test_pool_is_module(self):
        """When pool=True, self.pool is a MaxPool2d module."""
        block = ConvBlock(1, 8, pool=True)
        assert isinstance(block.pool, torch.nn.MaxPool2d)


# ================================================================
# LinearBlock
# ================================================================


class TestLinearBlock:
    """Tests for the LinearBlock building block."""

    def test_output_shape(self):
        """Output features match the constructor argument."""
        block = LinearBlock(in_features=128, out_features=64, dropout=0.5)
        x = torch.randn(8, 128)
        out = block(x)
        assert out.shape == (8, 64)

    def test_no_dropout(self):
        """dropout=None should work and not drop any neurons in eval mode."""
        block = LinearBlock(in_features=256, out_features=128, dropout=None)
        block.eval()
        x = torch.randn(4, 256)
        out = block(x)
        assert out.shape == (4, 128)
        assert block.dropout is None

    def test_dropout_zero_disables(self):
        """dropout=0 should result in no Dropout layer."""
        block = LinearBlock(in_features=64, out_features=32, dropout=0.0)
        assert block.dropout is None

    def test_dropout_train_vs_eval(self):
        """Dropout zeros some activations in train mode, none in eval mode."""
        block = LinearBlock(in_features=100, out_features=50, dropout=0.5)
        x = torch.randn(2000, 100)  # many samples for statistical stability

        block.train()
        out_train = block(x)

        block.eval()
        out_eval = block(x)

        # In train mode with p=0.5, some values should be exactly 0 (dropped).
        # In eval mode, no dropout → no zeroing by dropout.
        # Note: ReLU also produces zeros, so we can't just count zeros.
        # Instead, verify the two outputs are different (dropout makes train stochastic).
        assert not torch.allclose(out_train, out_eval), (
            "train and eval outputs should differ due to dropout"
        )

    def test_parameter_count(self):
        """Linear: in*out w + out b. BN: out*2. ReLU: 0. Dropout: 0."""
        block = LinearBlock(in_features=128, out_features=64, dropout=0.5)
        total = sum(p.numel() for p in block.parameters())
        expected = (128 * 64) + 64 + 64 + 64  # linear_w + linear_b + bn_w + bn_b
        assert total == expected

    def test_different_sizes(self):
        """Works with various in/out feature sizes."""
        block = LinearBlock(in_features=3136, out_features=128, dropout=0.5)
        x = torch.randn(2, 3136)
        out = block(x)
        assert out.shape == (2, 128)
