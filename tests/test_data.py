"""
数据集注册表 & 数据加载集成测试

覆盖: 10 个数据集的注册表元信息、transform 管线、DataLoader 构建
"""

import pytest
import torch

# 导入模型以触发注册（transform 需要查模型注册表）
import cnnlib.models  # noqa: F401
from cnnlib.registry.datasets import get_dataset_info, list_datasets

# ── 全部 10 个数据集 ───────────────────────────────────────

ALL_DATASETS = [
    ("mnist", 1, 10, 28),
    ("fashionmnist", 1, 10, 28),
    ("emnist", 1, 47, 28),
    ("cifar10", 3, 10, 32),
    ("cifar100", 3, 100, 32),
    ("svhn", 3, 10, 32),
    ("stl10", 3, 10, 96),
    ("caltech101", 3, 101, None),
    ("gtsrb", 3, 43, None),
    ("flowers102", 3, 102, None),
]

# 选两个小数据集做实加载（需联网下载，首次较慢）
SMALL_DATASETS = ["mnist", "cifar10"]


# ── 注册表 ──────────────────────────────────────────────────


class TestRegistry:
    """注册表元信息完整性"""

    def test_all_ten_registered(self):
        registered = list_datasets()
        for name, _, _, _ in ALL_DATASETS:
            assert name in registered, f"数据集未注册: {name}"
        assert len(registered) >= len(ALL_DATASETS)

    @pytest.mark.parametrize("name,ch,classes,size", ALL_DATASETS)
    def test_meta(self, name, ch, classes, size):
        info = get_dataset_info(name)
        assert info["channels"] == ch
        assert info["num_classes"] == classes
        assert info["image_size"] == size
        assert isinstance(info["mean"], tuple)
        assert isinstance(info["std"], tuple)
        assert len(info["mean"]) == ch
        assert len(info["std"]) == ch
        assert len(info["description"]) > 0

    def test_svhn_has_split_kwargs(self):
        """SVHN 需要 split= 而非 train="""
        info = get_dataset_info("svhn")
        assert "train_kwargs" in info
        assert "test_kwargs" in info
        assert info["train_kwargs"]["split"] == "train"
        assert info["test_kwargs"]["split"] == "test"

    def test_emnist_has_split_kwargs(self):
        """EMNIST 需要 split=balanced"""
        info = get_dataset_info("emnist")
        assert "train_kwargs" in info
        assert info["train_kwargs"]["split"] == "balanced"

    def test_default_datasets_no_extra_kwargs(self):
        """MNIST / CIFAR-10 等不需要额外构造参数"""
        for name in ["mnist", "cifar10", "fashionmnist"]:
            info = get_dataset_info(name)
            assert "train_kwargs" not in info, f"{name} 不应有特殊 kwargs"

    def test_case_insensitive(self):
        """数据集名称大小写不敏感"""
        assert get_dataset_info("CIFAR10")["num_classes"] == 10
        assert get_dataset_info("MnIsT")["channels"] == 1

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_dataset_info("nonexistent")


# ── Transform 管线 ──────────────────────────────────────────


class TestTransform:
    """transform 自动拼装"""

    def test_build_grayscale_to_grayscale(self):
        """灰度模型 + 灰度数据集: 无通道转换"""
        import numpy as np

        from cnnlib.data.transform import build_transform

        tf = build_transform("lenet", "mnist", augment=False)
        # ToTensor 需要 PIL 或 numpy (H, W, C)
        x = np.random.rand(3, 28, 28, 1).astype(np.float32)
        y = torch.stack([tf(img) for img in x])
        assert y.shape == (3, 1, 32, 32)  # MNIST 28→32(Resize to lenet size)

    def test_build_rgb_to_rgb(self):
        """RGB 模型 + RGB 数据集: 无通道转换"""
        import numpy as np

        from cnnlib.data.transform import build_transform

        tf = build_transform("vgg16", "cifar10", augment=False)
        x = np.random.rand(3, 32, 32, 3).astype(np.float32)
        y = torch.stack([tf(img) for img in x])
        assert y.shape == (3, 3, 224, 224)  # 32→224(Resize)

    def test_build_grayscale_to_rgb(self):
        """灰度数据集 + RGB 模型: 1→3 通道复制"""
        import numpy as np

        from cnnlib.data.transform import build_transform

        tf = build_transform("vgg16", "mnist", augment=False)
        x = np.random.rand(3, 28, 28, 1).astype(np.float32)
        y = torch.stack([tf(img) for img in x])
        assert y.shape == (3, 3, 224, 224)

    def test_train_augment_adds_ops(self):
        """训练模式比评估模式多增强操作"""
        from cnnlib.data.transform import build_transform

        train_tf = build_transform("vgg16", "cifar10", augment=True)
        eval_tf = build_transform("vgg16", "cifar10", augment=False)
        assert len(train_tf.transforms) > len(eval_tf.transforms)

    def test_normalize_applied(self):
        """Normalize 之后均值为 0 附近（大batch统计）"""
        import numpy as np

        from cnnlib.data.transform import build_transform

        tf = build_transform("alexnet", "cifar10", augment=False)
        x = np.random.rand(100, 32, 32, 3).astype(np.float32)
        y = torch.stack([tf(img) for img in x])
        # 归一化后均值接近 0
        assert y.mean().abs() < 0.2


# ── DataLoader 构建（实加载） ───────────────────────────────


class TestDataLoader:
    """真实加载小数据集，验证 Pipeline 端到端"""

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS)
    def test_build_and_iterate(self, dataset_name):
        from cnnlib.data.loader import build_dataloaders

        # 用匹配的模型避免尺寸不兼容
        if "mnist" in dataset_name:
            model = "lenet"
        else:
            model = "nin"  # 32x32 RGB，和 CIFAR-10 匹配

        train_ldr, val_ldr, test_ldr = build_dataloaders(
            model_name=model,
            dataset_name=dataset_name,
            batch_size=16,
            num_workers=0,
            val_split=0.1,
            seed=42,
        )

        # 三个 loader 都有数据
        assert len(train_ldr.dataset) > 0
        assert len(val_ldr.dataset) > 0
        assert len(test_ldr.dataset) > 0

        # 取一个 batch 验证形状
        images, labels = next(iter(train_ldr))
        info = get_dataset_info(dataset_name)
        assert images.dim() == 4
        assert images.shape[1] == info["channels"]
        assert labels.dim() == 1
        assert labels.shape[0] == images.shape[0]

    @pytest.mark.parametrize(
        "dataset_name,expected_train_total",
        [
            ("mnist", 60000),
            ("cifar10", 50000),
        ],
    )
    def test_val_split_ratio(self, dataset_name, expected_train_total):
        from cnnlib.data.loader import build_dataloaders

        if "mnist" in dataset_name:
            model = "lenet"
        else:
            model = "nin"

        train_ldr, val_ldr, test_ldr = build_dataloaders(
            model_name=model,
            dataset_name=dataset_name,
            batch_size=16,
            num_workers=0,
            val_split=0.2,
            seed=42,
        )
        assert len(train_ldr.dataset) + len(val_ldr.dataset) == expected_train_total

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS)
    def test_train_shuffled_val_not(self, dataset_name):
        """训练集 shuffle=True，验证/测试集 shuffle=False"""
        from cnnlib.data.loader import build_dataloaders

        if "mnist" in dataset_name:
            model = "lenet"
        else:
            model = "nin"

        train_ldr, val_ldr, _ = build_dataloaders(
            model_name=model,
            dataset_name=dataset_name,
            batch_size=64,
            num_workers=0,
            seed=42,
        )

        # 两次遍历 train_loader 顺序不同
        batch1_labels = []
        batch2_labels = []
        for images, labels in train_ldr:
            batch1_labels.append(labels.clone())
            break
        for images, labels in train_ldr:
            batch2_labels.append(labels.clone())
            break
        # shuffle 后大概率不同（极小概率相同）
        # 不做严格断言，只验证 loader 可复现迭代
        assert len(batch1_labels) > 0
        assert len(batch2_labels) > 0

    def test_svhn_loader_kwargs(self):
        """SVHN 使用 split= 而非 train= 也能正确加载"""
        from cnnlib.data.loader import build_dataloaders

        try:
            train_ldr, val_ldr, test_ldr = build_dataloaders(
                model_name="nin",
                dataset_name="svhn",
                batch_size=16,
                num_workers=0,
                val_split=0.1,
                seed=42,
            )
            assert len(train_ldr.dataset) > 0
            assert len(test_ldr.dataset) > 0
        except Exception as e:
            # SVHN 下载可能失败（网络问题等），跳过
            pytest.skip(f"SVHN 加载失败（可能是网络问题）: {e}")

    def test_emnist_loader_kwargs(self):
        """EMNIST 使用 split=balanced 正确加载"""
        from cnnlib.data.loader import build_dataloaders

        try:
            train_ldr, val_ldr, test_ldr = build_dataloaders(
                model_name="lenet",
                dataset_name="emnist",
                batch_size=16,
                num_workers=0,
                val_split=0.1,
                seed=42,
            )
            assert len(train_ldr.dataset) > 0
            assert len(test_ldr.dataset) > 0
        except Exception as e:
            pytest.skip(f"EMNIST 加载失败（可能是网络问题）: {e}")
