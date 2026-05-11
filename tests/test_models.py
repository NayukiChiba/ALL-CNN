"""
cnnlib 8 种模型架构单元测试

覆盖: LeNet, AlexNet, VGG11/13/16/19, NiN, GoogLeNet
"""

import pytest
import torch

from cnnlib.models.factory import create_model
from cnnlib.registry.models import get_model_info, list_models

# ── 模型清单 ────────────────────────────────────────────────

EXPECTED = [
    ("lenet", 32, 1),
    ("alexnet", 224, 3),
    ("vgg11", 224, 3),
    ("vgg13", 224, 3),
    ("vgg16", 224, 3),
    ("vgg19", 224, 3),
    ("nin", 32, 3),
    ("googlenet", 224, 3),
]

NAMES = [m[0] for m in EXPECTED]


def _deviceOf(model):
    return next(model.parameters()).device


def _dummy(model, batch=4):
    return torch.randn(
        batch,
        model.in_channels,
        model.input_size,
        model.input_size,
        device=_deviceOf(model),
    )


# ── 注册表 ──────────────────────────────────────────────────


class TestRegistry:
    """注册表完整性"""

    def test_all_eight_registered(self):
        registered = list_models()
        for name, _, _ in EXPECTED:
            assert name in registered, f"模型未注册: {name}"
        assert len(registered) >= 8

    @pytest.mark.parametrize("name,size,ch", EXPECTED)
    def test_meta_info(self, name, size, ch):
        info = get_model_info(name)
        assert info["input_size"] == size
        assert info["channels"] == ch


# ── 前向传播 ────────────────────────────────────────────────


class TestForward:
    """前向传播输出形状"""

    @pytest.mark.parametrize("name", NAMES)
    def test_default_shape(self, name):
        model = create_model(name, num_classes=10)
        x = _dummy(model)
        logits = model(x)
        assert logits.shape == (4, 10)

    @pytest.mark.parametrize("name", NAMES)
    def test_batch_size_1(self, name):
        model = create_model(name, num_classes=10)
        model.eval()
        x = _dummy(model, batch=1)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 10)

    @pytest.mark.parametrize("name", NAMES)
    def test_num_classes_100(self, name):
        model = create_model(name, num_classes=100)
        x = _dummy(model)
        logits = model(x)
        assert logits.shape == (4, 100)

    @pytest.mark.parametrize("name", NAMES)
    def test_logits_not_softmaxed(self, name):
        model = create_model(name, num_classes=10)
        model.eval()
        x = _dummy(model)
        with torch.no_grad():
            logits = model(x)
        # 原始 logits 不需要和为 1
        assert not torch.allclose(
            logits.sum(dim=1), torch.ones(4, device=logits.device)
        )


# ── 训练 / 梯度 ─────────────────────────────────────────────


class TestGradient:
    """梯度流验证"""

    @pytest.mark.parametrize("name", NAMES)
    def test_gradient_flow(self, name):
        model = create_model(name, num_classes=10)
        model.train()
        x = _dummy(model)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        for pname, param in model.named_parameters():
            assert param.grad is not None, f"{name}/{pname} 无梯度"


# ── eval / 确定性 ───────────────────────────────────────────


class TestEval:
    """eval 模式下确定性"""

    @pytest.mark.parametrize("name", NAMES)
    def test_eval_deterministic(self, name):
        model = create_model(name, num_classes=10)
        model.eval()
        x = _dummy(model)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    @pytest.mark.parametrize("name", NAMES)
    def test_train_nondeterministic(self, name):
        """train 模式下 Dropout 导致两次前传输出不同"""
        model = create_model(name, num_classes=10)
        model.train()
        x = _dummy(model)
        out1 = model(x)
        out2 = model(x)
        # LeNet 和 NiN 无 Dropout，跳过
        if name in ("lenet", "nin"):
            return
        assert not torch.allclose(out1, out2), f"{name} train 模式两次输出应不同"


# ── summary / param_count ───────────────────────────────────


class TestSummary:
    """BaseModel 工具方法"""

    @pytest.mark.parametrize("name", NAMES)
    def test_param_count_positive(self, name):
        model = create_model(name, num_classes=10)
        assert model.param_count() > 0

    @pytest.mark.parametrize("name", NAMES)
    def test_summary_contains_name(self, name):
        model = create_model(name, num_classes=10)
        s = model.summary()
        assert "param" in s
        assert "input" in s


# ── 工厂 ────────────────────────────────────────────────────


class TestFactory:
    """create_model 各参数组合"""

    def test_create_lenet_custom_num_classes(self):
        m = create_model("lenet", num_classes=5)
        x = _dummy(m)
        assert m(x).shape == (4, 5)

    def test_create_alexnet_dropout(self):
        m1 = create_model("alexnet", num_classes=10, dropout=0.2)
        m2 = create_model("alexnet", num_classes=10, dropout=0.8)
        assert m1(x := _dummy(m1)).shape == m2(_dummy(m2)).shape

    @pytest.mark.parametrize("variant", ["11", "13", "16", "19"])
    def test_vgg_variants(self, variant):
        name = f"vgg{variant}"
        m = create_model(name, num_classes=10)
        x = _dummy(m)
        assert m(x).shape == (4, 10)

    def test_nin_num_classes(self):
        m = create_model("nin", num_classes=200)
        x = _dummy(m)
        assert m(x).shape == (4, 200)

    def test_googlenet_num_classes(self):
        m = create_model("googlenet", num_classes=50)
        x = _dummy(m)
        assert m(x).shape == (4, 50)

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            create_model("nonexistent", num_classes=10)


# ── VGG 特有：各变体权重层数 ────────────────────────────────


class TestVGG:
    """VGG 变体参数量随深度递增"""

    @pytest.mark.parametrize("variant", ["11", "13", "16", "19"])
    def test_vgg_stage_count(self, variant):
        m = create_model(f"vgg{variant}", num_classes=10)
        stages = [mod for mod in m.features if isinstance(mod, torch.nn.Sequential)]
        assert len(stages) == 5

    def test_vgg_params_increasing(self):
        """VGG11 < VGG13 < VGG16 < VGG19"""
        prev = 0
        for v in ["11", "13", "16", "19"]:
            m = create_model(f"vgg{v}", num_classes=10)
            cur = m.param_count()
            assert cur > prev, f"VGG{v} param <= VGG 前一个"
            prev = cur
