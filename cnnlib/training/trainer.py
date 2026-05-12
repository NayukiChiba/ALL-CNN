"""
训练器

编排完整的训练流程:训练循环 + 验证 + 调度 + 早停 + checkpoint + 日志.

用法:
    from cnnlib.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        trainLoader=trainLoader,
        valLoader=valLoader,
        optimizer=optimizer,
        scheduler=scheduler,
        lossFn=lossFn,
        device=device,
        epochs=50,
        checkpointDir=checkpointDir,
        logger=logger,
        earlyStopping=EarlyStopping(patience=10),
    )
    result = trainer.train()
"""

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from cnnlib.training.checkpoint import loadCheckpoint, saveCheckpoint
from cnnlib.training.earlyStopping import EarlyStopping
from cnnlib.training.engine import trainOneEpoch, validate
from cnnlib.training.logger import TrainingLogger


class Trainer:
    """
    训练编排器

    职责:
        - 逐轮调用 engine 完成训练/验证
        - 调度学习率(plateau 需传验证 loss)
        - 早停判断
        - 保存最佳/最新 checkpoint
        - 日志记录
        - 训练结束后在测试集上评估
    """

    def __init__(
        self,
        model: nn.Module,
        trainLoader: DataLoader,
        valLoader: DataLoader,
        optimizer: Optimizer,
        scheduler,
        lossFn: nn.Module,
        device: torch.device,
        epochs: int,
        checkpointDir: str | Path,
        testLoader: DataLoader | None = None,
        logger: TrainingLogger | None = None,
        earlyStopping: EarlyStopping | None = None,
        gradClip: float = 0.0,
        resumeFrom: str | None = None,
    ):
        """
        Args:
            model:          模型实例
            trainLoader:    训练集 DataLoader
            valLoader:      验证集 DataLoader
            optimizer:      优化器
            scheduler:      学习率调度器
            lossFn:         损失函数
            device:         计算设备
            epochs:         最大训练轮数
            checkpointDir:  checkpoint 保存目录
            testLoader:     测试集 DataLoader(可选,训练结束后评估)
            logger:         日志器
            earlyStopping:  早停控制器
            gradClip:       梯度裁剪阈值(0=不裁剪)
            resumeFrom:     checkpoint 路径(恢复训练)
        """
        self.model = model
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.testLoader = testLoader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lossFn = lossFn
        self.device = device
        self.epochs = epochs
        self.checkpointDir = Path(checkpointDir)
        self.logger = logger
        self.earlyStopping = earlyStopping
        self.gradClip = gradClip
        self.resumeFrom = resumeFrom

        self.bestModelPath = self.checkpointDir / "best_model.pth"
        self.lastModelPath = self.checkpointDir / "last_model.pth"

        # 训练状态
        self.currentEpoch: int = 0
        self.bestMetric: float = 0.0
        self.history: Dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train(self) -> Dict:
        """
        运行完整训练流程

        Returns:
            {"best_metric": 最佳验证指标, "best_epoch": 最佳轮次, "history": {...}}
        """
        try:
            return self._runTraining()
        finally:
            if self.logger:
                self.logger.close()

    def _runTraining(self) -> Dict:
        startEpoch = 1

        # 恢复训练
        if self.resumeFrom is not None:
            checkpoint = loadCheckpoint(
                self.resumeFrom,
                self.model,
                self.optimizer,
                self.scheduler,
                device=str(self.device),
            )
            startEpoch = checkpoint["epoch"] + 1
            self.bestMetric = checkpoint["best_metric"]
            if self.logger:
                self.logger.logMessage(
                    f"从 checkpoint 恢复: {self.resumeFrom} (epoch {checkpoint['epoch']}, "
                    f"best_metric={checkpoint['best_metric']:.4f})"
                )

        if self.logger:
            paramCount = getattr(self.model, "param_count", lambda: "?")()
            self.logger.logMessage(
                f"模型: {self.model.__class__.__name__}, "
                f"参数: {paramCount}, "
                f"设备: {self.device}"
            )
            self.logger.logMessage(
                f"训练集: {len(self.trainLoader.dataset):,}, "
                f"验证集: {len(self.valLoader.dataset):,}"
            )
            if self.testLoader:
                self.logger.logMessage(f"测试集: {len(self.testLoader.dataset):,}")
            self.logger.logMessage(
                f"Epochs: {self.epochs}, LR: {self.optimizer.param_groups[0]['lr']}"
            )
            self.logger.logMessage("=" * 60)

        bestEpoch = startEpoch

        for epoch in range(startEpoch, self.epochs + 1):
            self.currentEpoch = epoch

            # 训练
            trainMetrics = trainOneEpoch(
                self.model,
                self.trainLoader,
                self.lossFn,
                self.optimizer,
                self.device,
                epoch,
                self.logger,
                self.gradClip,
            )

            # 验证
            valMetrics = validate(
                self.model, self.valLoader, self.lossFn, self.device, desc="Val"
            )

            # 调度器 step
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(valMetrics["loss"])
            else:
                self.scheduler.step()

            currentLr = self.optimizer.param_groups[0]["lr"]

            # 记录历史
            self.history["train_loss"].append(trainMetrics["loss"])
            self.history["train_acc"].append(trainMetrics["accuracy"])
            self.history["val_loss"].append(valMetrics["loss"])
            self.history["val_acc"].append(valMetrics["accuracy"])

            # 日志
            if self.logger:
                self.logger.logMetrics(
                    {
                        "loss": trainMetrics["loss"],
                        "accuracy": trainMetrics["accuracy"],
                    },
                    epoch,
                    prefix="train",
                )
                self.logger.logMetrics(
                    {"loss": valMetrics["loss"], "accuracy": valMetrics["accuracy"]},
                    epoch,
                    prefix="val",
                )
                self.logger.logMetrics({"lr": currentLr}, epoch)
                self.logger.logMessage(
                    f"Epoch {epoch:3d}/{self.epochs} | "
                    f"train loss={trainMetrics['loss']:.4f} acc={trainMetrics['accuracy']:.2f}% | "
                    f"val loss={valMetrics['loss']:.4f} acc={valMetrics['accuracy']:.2f}% | "
                    f"lr={currentLr:.2e}"
                )

            # 保存最佳模型
            isBest = valMetrics["accuracy"] > self.bestMetric
            if isBest:
                self.bestMetric = valMetrics["accuracy"]
                bestEpoch = epoch
                saveCheckpoint(
                    self.bestModelPath,
                    self.model,
                    self.optimizer,
                    epoch,
                    self.bestMetric,
                    valMetrics,
                    self.scheduler,
                )
                if self.logger:
                    self.logger.logMessage(
                        f"  >> 保存最佳模型 (acc={self.bestMetric:.2f}%)"
                    )

            # 保存最新模型
            saveCheckpoint(
                self.lastModelPath,
                self.model,
                self.optimizer,
                epoch,
                self.bestMetric,
                valMetrics,
                self.scheduler,
            )

            # 早停
            if self.earlyStopping is not None:
                if self.earlyStopping.step(valMetrics["accuracy"]):
                    if self.logger:
                        self.logger.logMessage(
                            f"早停触发于 epoch {epoch},最佳 val_acc={self.bestMetric:.2f}%"
                        )
                    break

        # 训练结束
        if self.logger:
            self.logger.logMessage("=" * 60)
            self.logger.logMessage(
                f"训练完成 | 最佳 val_acc={self.bestMetric:.2f}% (epoch {bestEpoch})"
            )

        # 加载最佳模型
        if self.bestModelPath.exists():
            loadCheckpoint(self.bestModelPath, self.model, device=str(self.device))
            if self.logger:
                self.logger.logMessage("已加载最佳模型权重")

        # 测试集评估
        testMetrics = None
        if self.testLoader is not None:
            testMetrics = validate(
                self.model, self.testLoader, self.lossFn, self.device, desc="Test"
            )
            if self.logger:
                self.logger.logMessage(
                    f"测试集 | loss={testMetrics['loss']:.4f} acc={testMetrics['accuracy']:.2f}%"
                )

        return {
            "best_metric": self.bestMetric,
            "best_epoch": bestEpoch,
            "history": self.history,
            "test_metrics": testMetrics,
        }
