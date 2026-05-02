"""
Unit tests for src/model/cnn.py — MNISTCNN model.
"""

import torch

from src.model.cnn import MNISTCNN


class TestMNISTCNN:
    """Tests for the MNISTCNN model architecture and forward pass."""

    def test_default_architecture_output_shape(self):
        """Default model: (B,1,28,28) → (B,10)."""
        model = MNISTCNN()
        x = torch.randn(8, 1, 28, 28)
        logits = model(x)
        assert logits.shape == (8, 10)

    def test_batch_size_1(self):
        """Single sample inference works (eval mode — BN needs >1 sample in train)."""
        model = MNISTCNN()
        model.eval()
        x = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 10)

    def test_batch_size_64(self):
        """Larger batch works."""
        model = MNISTCNN()
        x = torch.randn(64, 1, 28, 28)
        logits = model(x)
        assert logits.shape == (64, 10)

    def test_deeper_architecture(self):
        """Three ConvBlocks: 1→32→64→128."""
        model = MNISTCNN(conv_channels=[32, 64, 128])
        x = torch.randn(4, 1, 28, 28)
        logits = model(x)
        # 3 pooling layers: 28 → 14 → 7 → 3
        # flattened: 128 * 3 * 3 = 1152
        assert logits.shape == (4, 10)

    def test_custom_hidden_size(self):
        """Hidden FC size of 256."""
        model = MNISTCNN(fc_hidden_size=256)
        x = torch.randn(4, 1, 28, 28)
        logits = model(x)
        assert logits.shape == (4, 10)
        # Check the LinearBlock output size
        sample = torch.randn(2, 64 * 7 * 7)  # flattened after conv layers
        out = model.fc_block(sample)
        assert out.shape == (2, 256)

    def test_zero_dropout(self):
        """dropout=0 is valid (no dropout applied)."""
        model = MNISTCNN(dropout=0.0)
        model.eval()
        x = torch.randn(4, 1, 28, 28)
        logits = model(x)
        assert logits.shape == (4, 10)

    def test_eval_mode_disables_dropout(self):
        """Two forward passes in eval mode should give identical results."""
        model = MNISTCNN(dropout=0.5)
        model.eval()
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_train_mode_gives_different_results(self):
        """In train mode, dropout causes different outputs each forward pass."""
        model = MNISTCNN(dropout=0.5)
        model.train()
        x = torch.randn(4, 1, 28, 28)
        out1 = model(x)
        out2 = model(x)
        # With dropout=0.5, outputs should differ
        assert not torch.allclose(out1, out2)

    def test_parameter_count(self):
        """Verify default model has the expected number of parameters."""
        model = MNISTCNN()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # All params should be trainable in this model
        assert total == trainable
        # Sanity: MNIST CNN should have between 100k and 1M params
        assert 100_000 < total < 1_000_000

    def test_conv_channels_list(self):
        """conv_channels is stored as a list, converted from default None."""
        model_default = MNISTCNN()
        assert len(list(model_default.conv_blocks)) == 2  # [32, 64] default

        model_custom = MNISTCNN(conv_channels=[16, 32, 64, 128])
        assert len(list(model_custom.conv_blocks)) == 4

    def test_flatten_output_size(self):
        """Verify the flatten math for different conv channel counts."""
        # Default: 2 conv blocks → 28/(2^2) = 7, 64*7*7 = 3136
        model = MNISTCNN(conv_channels=[32, 64])
        x = torch.randn(2, 1, 28, 28)
        # Pass through conv blocks only
        for block in model.conv_blocks:
            x = block(x)
        assert x.shape == (2, 64, 7, 7)
        flat = model.flatten(x)
        assert flat.shape == (2, 3136)

    def test_logits_not_softmaxed(self):
        """Forward pass returns raw logits (not probabilities)."""
        model = MNISTCNN()
        model.eval()
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            logits = model(x)
        # Raw logits can be negative and don't sum to 1
        assert logits.min() < 0 or logits.max() > 1
        assert not torch.allclose(logits.sum(dim=1), torch.ones(4))

    def test_gradient_flow(self):
        """Gradients propagate through the full model."""
        model = MNISTCNN()
        model.train()
        x = torch.randn(4, 1, 28, 28, requires_grad=False)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Every parameter should have a gradient
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
