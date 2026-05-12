"""
训练模块单元测试

覆盖: 损失函数工厂、优化器工厂、调度器工厂、早停、checkpoint、日志、引擎、Trainer
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader, TensorDataset

import cnnlib.models  # noqa: F401
from cnnlib.training import (
    EarlyStopping,
    Trainer,
    TrainingLogger,
    createLoss,
    createOptimizer,
    createScheduler,
    loadCheckpoint,
    saveCheckpoint,
    trainOneEpoch,
    validate,
)

# ── 简易线性模型，不依赖注册表 ─────────────────────────────────


class _TinyModel(nn.Module):
    def __init__(self, inDim=10, numClasses=3):
        super().__init__()
        self.fc = nn.Linear(inDim, numClasses)

    def forward(self, x):
        return self.fc(x)


def _makeDummyLoader(samples=64, inDim=10, numClasses=3, batchSize=16):
    x = torch.randn(samples, inDim)
    y = torch.randint(0, numClasses, (samples,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batchSize, shuffle=True)


# ── 损失函数 ──────────────────────────────────────────────────


class TestLoss:
    def test_cross_entropy_default(self):
        lossFn = createLoss("cross_entropy")
        assert isinstance(lossFn, nn.CrossEntropyLoss)

    def test_cross_entropy_label_smoothing(self):
        lossFn = createLoss("cross_entropy", label_smoothing=0.1)
        assert lossFn.label_smoothing == 0.1

    def test_mse(self):
        lossFn = createLoss("mse")
        assert isinstance(lossFn, nn.MSELoss)

    def test_case_insensitive(self):
        assert isinstance(createLoss("CROSS_ENTROPY"), nn.CrossEntropyLoss)
        assert isinstance(createLoss("Cross_Entropy"), nn.CrossEntropyLoss)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="未知损失函数"):
            createLoss("nonexistent")


# ── 优化器 ────────────────────────────────────────────────────


class TestOptimizer:
    def test_adam_default(self):
        model = _TinyModel()
        opt = createOptimizer(model, "adam", lr=0.001)
        assert isinstance(opt, Adam)
        assert opt.param_groups[0]["lr"] == 0.001

    def test_adamw(self):
        model = _TinyModel()
        opt = createOptimizer(model, "adamw", lr=0.01, weight_decay=1e-4)
        assert isinstance(opt, AdamW)
        assert opt.param_groups[0]["weight_decay"] == 1e-4

    def test_sgd_momentum(self):
        model = _TinyModel()
        opt = createOptimizer(model, "sgd", lr=0.01, momentum=0.9)
        assert isinstance(opt, SGD)
        assert opt.param_groups[0]["momentum"] == 0.9

    def test_rmsprop(self):
        model = _TinyModel()
        opt = createOptimizer(model, "rmsprop", lr=0.001)
        assert isinstance(opt, RMSprop)

    def test_all_params_covered(self):
        """优化器覆盖模型全部参数"""
        model = _TinyModel()
        opt = createOptimizer(model, "adam")
        paramIds = {id(p) for p in model.parameters()}
        optParamIds = {id(p) for group in opt.param_groups for p in group["params"]}
        assert paramIds == optParamIds

    def test_case_insensitive(self):
        model = _TinyModel()
        assert isinstance(createOptimizer(model, "ADAM"), Adam)

    def test_unknown_raises(self):
        model = _TinyModel()
        with pytest.raises(ValueError, match="未知优化器"):
            createOptimizer(model, "nonexistent")


# ── 调度器 ────────────────────────────────────────────────────


class TestScheduler:
    def test_plateau(self):
        model = _TinyModel()
        opt = createOptimizer(model, "adam")
        sched = createScheduler(opt, "plateau", factor=0.5, patience=3)
        assert isinstance(sched, ReduceLROnPlateau)
        assert sched.factor == 0.5
        assert sched.patience == 3

    def test_step(self):
        model = _TinyModel()
        opt = createOptimizer(model, "adam")
        sched = createScheduler(opt, "step", step_size=10, gamma=0.1)
        assert isinstance(sched, StepLR)
        assert sched.step_size == 10

    def test_cosine(self):
        model = _TinyModel()
        opt = createOptimizer(model, "adam")
        sched = createScheduler(opt, "cosine", T_max=50)
        assert isinstance(sched, CosineAnnealingLR)
        assert sched.T_max == 50

    def test_cosine_warm(self):
        model = _TinyModel()
        opt = createOptimizer(model, "adam")
        sched = createScheduler(opt, "cosine_warm", T_0=10, T_mult=2)
        assert isinstance(sched, CosineAnnealingWarmRestarts)

    def test_case_insensitive(self):
        model = _TinyModel()
        opt = createOptimizer(model, "adam")
        assert isinstance(createScheduler(opt, "PLATEAU"), ReduceLROnPlateau)

    def test_unknown_raises(self):
        model = _TinyModel()
        opt = createOptimizer(model, "adam")
        with pytest.raises(ValueError, match="未知调度器"):
            createScheduler(opt, "nonexistent")


# ── 早停 ──────────────────────────────────────────────────────


class TestEarlyStopping:
    def test_init_defaults(self):
        es = EarlyStopping()
        assert es.patience == 10
        assert es.minDelta == 0.001
        assert es.mode == "max"
        assert es.bestMetric is None
        assert es.counter == 0

    def test_mode_max_improves(self):
        es = EarlyStopping(patience=3, mode="max")
        assert not es.step(0.8)
        assert es.bestMetric == 0.8
        assert not es.step(0.85)
        assert es.bestMetric == 0.85

    def test_mode_max_stops(self):
        es = EarlyStopping(patience=3, minDelta=0.01, mode="max")
        es.step(0.8)
        assert not es.step(0.81)  # improve=0.01, not > minDelta → counter=1
        assert not es.step(0.79)  # worse → counter=2
        assert es.step(0.79)  # no improvement → counter=3 → stop
        assert es.shouldStop

    def test_mode_min_improves(self):
        es = EarlyStopping(patience=3, mode="min")
        assert not es.step(0.5)
        assert not es.step(0.4)
        assert es.bestMetric == 0.4

    def test_mode_min_stops(self):
        es = EarlyStopping(patience=2, minDelta=0.01, mode="min")
        es.step(0.5)
        assert not es.step(0.5)
        assert es.step(0.51)
        assert es.shouldStop

    def test_reset(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.8)
        es.step(0.7)
        es.step(0.7)
        assert es.shouldStop
        es.reset()
        assert not es.shouldStop
        assert es.bestMetric is None
        assert es.counter == 0

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode 必须是"):
            EarlyStopping(mode="invalid")

    def test_min_delta_zero(self):
        """minDelta=0 时只要持平不改善即计次"""
        es = EarlyStopping(patience=2, minDelta=0.0, mode="max")
        es.step(0.8)
        assert not es.step(0.8)  # 持平也不算改善
        assert es.counter == 1


# ── Checkpoint ────────────────────────────────────────────────


class TestCheckpoint:
    def test_save_and_load_roundtrip(self):
        model = _TinyModel()
        opt = createOptimizer(model, "sgd", lr=0.01)
        sched = createScheduler(opt, "step", step_size=10)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pth")
            metrics = {"loss": 0.5, "accuracy": 80.0}
            saveCheckpoint(
                path,
                model,
                opt,
                epoch=5,
                bestMetric=0.9,
                metrics=metrics,
                scheduler=sched,
            )

            model2 = _TinyModel()
            opt2 = createOptimizer(model2, "sgd", lr=0.01)
            sched2 = createScheduler(opt2, "step", step_size=10)

            ckpt = loadCheckpoint(path, model2, opt2, sched2, device="cpu")
            assert ckpt["epoch"] == 5
            assert ckpt["best_metric"] == 0.9
            assert ckpt["metrics"] == metrics

            # 权重一致
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.equal(p1, p2)

    def test_load_missing_file(self):
        model = _TinyModel()
        with pytest.raises(FileNotFoundError):
            loadCheckpoint("nonexistent.pth", model)

    def test_load_without_optimizer_scheduler(self):
        model = _TinyModel()
        opt = createOptimizer(model, "sgd", lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pth")
            saveCheckpoint(path, model, opt, epoch=1, bestMetric=0.8, metrics={})

            model2 = _TinyModel()
            ckpt = loadCheckpoint(path, model2, device="cpu")
            assert ckpt["epoch"] == 1

    def test_creates_parent_dirs(self):
        model = _TinyModel()
        opt = createOptimizer(model, "sgd", lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "nested", "test.pth")
            saveCheckpoint(path, model, opt, epoch=1, bestMetric=0.8, metrics={})
            assert os.path.exists(path)


# ── 日志 ──────────────────────────────────────────────────────


class TestLogger:
    def test_creates_log_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrainingLogger(tmp, "testModel", "testData")
            logger.logMessage("hello world")
            logger.close()
            logFile = os.path.join(tmp, "training.log")
            assert os.path.exists(logFile)
            with open(logFile, encoding="utf-8") as f:
                content = f.read()
            assert "hello world" in content

    def test_creates_tensorboard_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrainingLogger(tmp, "testModel", "testData")
            logger.logMetrics({"loss": 0.5, "acc": 80.0}, step=1, prefix="train")
            logger.close()
            tbDir = os.path.join(tmp, "..", "tensorboard")
            assert os.path.isdir(tbDir)

    def test_log_metrics_without_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrainingLogger(tmp, "testModel", "testData")
            logger.logMetrics({"lr": 0.001}, step=1)
            logger.close()

    def test_log_multiple_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrainingLogger(tmp, "testModel", "testData")
            logger.logMessage("debug msg", level="debug")
            logger.logMessage("info msg", level="info")
            logger.logMessage("warning msg", level="warning")
            logger.logMessage("error msg", level="error")
            logger.close()
            logFile = os.path.join(tmp, "training.log")
            with open(logFile, encoding="utf-8") as f:
                content = f.read()
            assert "info msg" in content
            assert "warning msg" in content
            assert "error msg" in content


# ── 引擎 ──────────────────────────────────────────────────────


class TestEngine:
    def test_train_one_epoch_returns_metrics(self):
        model = _TinyModel()
        loader = _makeDummyLoader(64)
        lossFn = nn.CrossEntropyLoss()
        opt = createOptimizer(model, "sgd", lr=0.01)
        device = torch.device("cpu")

        metrics = trainOneEpoch(model, loader, lossFn, opt, device, epoch=1)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0
        assert 0 <= metrics["accuracy"] <= 100

    def test_train_reduces_loss(self):
        """训练后损失应下降"""
        model = _TinyModel()
        loader = _makeDummyLoader(128)
        lossFn = nn.CrossEntropyLoss()
        opt = createOptimizer(model, "sgd", lr=0.1)
        device = torch.device("cpu")

        m1 = trainOneEpoch(model, loader, lossFn, opt, device, epoch=1)
        m2 = trainOneEpoch(model, loader, lossFn, opt, device, epoch=2)
        # 在简单数据上 SGD 应显著降低 loss
        assert m2["loss"] < m1["loss"], (
            f"loss 未下降: {m1['loss']:.4f} → {m2['loss']:.4f}"
        )

    def test_validate_returns_metrics(self):
        model = _TinyModel()
        loader = _makeDummyLoader(64)
        lossFn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        metrics = validate(model, loader, lossFn, device)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_validate_deterministic(self):
        model = _TinyModel()
        model.eval()
        loader = _makeDummyLoader(128)
        lossFn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        m1 = validate(model, loader, lossFn, device)
        m2 = validate(model, loader, lossFn, device)
        assert m1["loss"] == pytest.approx(m2["loss"], rel=1e-6)
        assert m1["accuracy"] == m2["accuracy"]

    def test_grad_clip(self):
        model = _TinyModel()
        loader = _makeDummyLoader(32)
        lossFn = nn.CrossEntropyLoss()
        opt = createOptimizer(model, "sgd", lr=0.01)
        device = torch.device("cpu")

        # gradClip > 0 应无报错
        metrics = trainOneEpoch(
            model, loader, lossFn, opt, device, epoch=1, gradClip=1.0
        )
        assert "loss" in metrics


# ── Trainer 集成 ──────────────────────────────────────────────


class TestTrainer:
    def test_minimal_training_run(self):
        """最小化训练: 2 轮, 小模型, 合成数据"""
        model = _TinyModel()
        trainLoader = _makeDummyLoader(128)
        valLoader = _makeDummyLoader(64)
        testLoader = _makeDummyLoader(32)
        opt = createOptimizer(model, "sgd", lr=0.05)
        sched = createScheduler(opt, "step", step_size=10)
        lossFn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmp:
            ckptDir = os.path.join(tmp, "checkpoints")
            logDir = os.path.join(tmp, "logs")
            logger = TrainingLogger(logDir, "tiny", "dummy")
            es = EarlyStopping(patience=10)

            trainer = Trainer(
                model=model,
                trainLoader=trainLoader,
                valLoader=valLoader,
                testLoader=testLoader,
                optimizer=opt,
                scheduler=sched,
                lossFn=lossFn,
                device=device,
                epochs=2,
                checkpointDir=ckptDir,
                logger=logger,
                earlyStopping=es,
                gradClip=0.0,
            )

            result = trainer.train()

            assert "best_metric" in result
            assert "history" in result
            assert "test_metrics" in result
            assert len(result["history"]["train_loss"]) == 2
            assert len(result["history"]["val_loss"]) == 2
            assert result["test_metrics"]["accuracy"] >= 0

            # checkpoint 已保存
            assert os.path.exists(os.path.join(ckptDir, "best_model.pth"))
            assert os.path.exists(os.path.join(ckptDir, "last_model.pth"))

    def test_early_stopping_triggers(self):
        """早停在 patience 耗尽时触发"""
        model = _TinyModel()
        loader = _makeDummyLoader(128)
        opt = createOptimizer(model, "sgd", lr=0.0)  # lr=0 不会改善
        sched = createScheduler(opt, "step", step_size=10)
        lossFn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmp:
            ckptDir = os.path.join(tmp, "checkpoints")
            logDir = os.path.join(tmp, "logs")
            logger = TrainingLogger(logDir, "tiny", "dummy")
            es = EarlyStopping(patience=2, minDelta=0.0, mode="max")

            trainer = Trainer(
                model=model,
                trainLoader=loader,
                valLoader=loader,
                optimizer=opt,
                scheduler=sched,
                lossFn=lossFn,
                device=device,
                epochs=20,
                checkpointDir=ckptDir,
                logger=logger,
                earlyStopping=es,
            )

            result = trainer.train()
            # lr=0 所以模型不会改善，早停应在 patience 耗尽后触发
            # 实际完成轮数 ≤ patience + 1(首轮不计) + 1
            assert trainer.currentEpoch <= 4

    def test_resume_from_checkpoint(self):
        """从 checkpoint 恢复训练"""
        model = _TinyModel()
        trainLoader = _makeDummyLoader(64)
        valLoader = _makeDummyLoader(64)
        opt = createOptimizer(model, "sgd", lr=0.05)
        sched = createScheduler(opt, "step", step_size=10)
        lossFn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmp:
            ckptDir = os.path.join(tmp, "checkpoints")
            logDir = os.path.join(tmp, "logs")
            logger = TrainingLogger(logDir, "tiny", "dummy")
            es = EarlyStopping(patience=10)

            trainer = Trainer(
                model=model,
                trainLoader=trainLoader,
                valLoader=valLoader,
                optimizer=opt,
                scheduler=sched,
                lossFn=lossFn,
                device=device,
                epochs=2,
                checkpointDir=ckptDir,
                logger=logger,
                earlyStopping=es,
            )
            result1 = trainer.train()

            # 用 best_model.pth 恢复继续训练
            resumePath = os.path.join(ckptDir, "best_model.pth")
            model2 = _TinyModel()
            opt2 = createOptimizer(model2, "sgd", lr=0.05)
            sched2 = createScheduler(opt2, "step", step_size=10)
            logger2 = TrainingLogger(logDir, "tiny", "dummy2")

            trainer2 = Trainer(
                model=model2,
                trainLoader=trainLoader,
                valLoader=valLoader,
                optimizer=opt2,
                scheduler=sched2,
                lossFn=lossFn,
                device=device,
                epochs=4,
                checkpointDir=ckptDir,
                logger=logger2,
                earlyStopping=EarlyStopping(patience=10),
                resumeFrom=resumePath,
            )
            result2 = trainer2.train()
            # 恢复训练从 epoch 2 开始，至少有一个 epoch 的记录
            assert result2["best_epoch"] >= 2


# ── LeNet + Trainer 端到端 ─────────────────────────────────────


class TestEndToEnd:
    """使用真实注册模型 + DataLoader 做集成验证"""

    def test_lenet_mnist_small(self):
        """LeNet + MNIST(实加载), 2 epochs，验证管线无报错"""
        from cnnlib.data.loader import build_dataloaders
        from cnnlib.models.factory import create_model_for_dataset

        model = create_model_for_dataset("lenet", "mnist", device="cpu")
        trainLoader, valLoader, testLoader = build_dataloaders(
            model_name="lenet",
            dataset_name="mnist",
            batch_size=32,
            val_split=0.1,
            num_workers=0,
            seed=42,
        )

        opt = createOptimizer(model, "adam", lr=0.001, weight_decay=1e-4)
        sched = createScheduler(opt, "plateau", factor=0.5, patience=2)
        lossFn = createLoss("cross_entropy")
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmp:
            ckptDir = os.path.join(tmp, "checkpoints")
            logDir = os.path.join(tmp, "logs")
            logger = TrainingLogger(logDir, "lenet", "mnist")
            es = EarlyStopping(patience=5)

            trainer = Trainer(
                model=model,
                trainLoader=trainLoader,
                valLoader=valLoader,
                optimizer=opt,
                scheduler=sched,
                lossFn=lossFn,
                device=device,
                epochs=2,
                checkpointDir=ckptDir,
                logger=logger,
                earlyStopping=es,
            )

            result = trainer.train()

            assert "history" in result
            assert len(result["history"]["train_loss"]) == 2
            assert os.path.exists(os.path.join(ckptDir, "best_model.pth"))
            assert os.path.exists(os.path.join(ckptDir, "last_model.pth"))
