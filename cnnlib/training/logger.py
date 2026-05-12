"""
训练日志

同时输出到控制台、文件、TensorBoard.统一管理日志格式和写入.

用法:
    from cnnlib.training.logger import TrainingLogger

    logger = TrainingLogger(logDir, "vgg16", "cifar10")
    logger.logMetrics({"train_loss": 0.5, "train_acc": 85.0}, step=1, prefix="train")
    logger.logMessage("训练开始")
    logger.close()
"""

import logging
from pathlib import Path
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    训练日志管理器

    三路输出:
        1. 控制台 — 简洁的即时反馈
        2. 文件   — training.log 完整记录
        3. TensorBoard — 标量曲线可视化
    """

    def __init__(self, logDir: str | Path, modelName: str, datasetName: str):
        """
        Args:
            logDir:      日志目录
            modelName:   模型名称
            datasetName: 数据集名称
        """
        logDir = Path(logDir)
        logDir.mkdir(parents=True, exist_ok=True)

        # 文本日志
        loggerName = f"{modelName}_{datasetName}"
        self.logger = logging.getLogger(loggerName)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        logFile = logDir / "training.log"
        fileHandler = logging.FileHandler(str(logFile), encoding="utf-8")
        fileHandler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        )
        self.logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(consoleHandler)

        # TensorBoard
        tbDir = logDir.parent / "tensorboard"
        tbDir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tbDir))

    def logMetrics(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        """
        记录标量指标到 TensorBoard

        Args:
            metrics: 指标字典 {"loss": 0.5, "accuracy": 85.0}
            step:    全局步数(epoch)
            prefix:  标签前缀,如 "train" / "val"
        """
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(tag, value, step)

    def logMessage(self, message: str, level: str = "info") -> None:
        """
        记录文本消息

        Args:
            message: 消息内容
            level:   日志级别: debug / info / warning / error
        """
        getattr(self.logger, level)(message)

    def addGraph(self, model, inputTensor) -> None:
        """记录模型计算图到 TensorBoard"""
        try:
            self.writer.add_graph(model, inputTensor)
        except Exception:
            pass  # 部分模型可能不兼容,静默跳过

    def close(self) -> None:
        """关闭日志,释放资源"""
        self.writer.close()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
