"""
CLI 参数解析

所有命令行参数在此定义。main.py 只负责 dispatch。

用法:
    from cnnlib.cli import getSettings, buildParser

    args = getSettings()

CLI 规范:
    python main.py [全局参数] <子命令> [子命令参数]
    python main.py --model vgg16 --dataset cifar10 train --epochs 50
"""

import argparse

from config.data import DataParams
from config.defaults import DefaultParams
from config.paths import CHECKPOINTS_DIR, DATASETS_DIR, LOGS_DIR, OUTPUTS_DIR
from config.training import TrainingParams

# ═══════════════════════════════════════════════════════════════
# 共享参数组
# ═══════════════════════════════════════════════════════════════


def _addGlobalArgs(parser: argparse.ArgumentParser) -> None:
    """全局参数:设备、随机种子、模型选择、数据集选择"""
    parser.add_argument(
        "--device",
        type=str,
        default=DefaultParams.DEVICE,
        choices=["cpu", "cuda"],
        help="计算设备",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DefaultParams.SEED,
        help="随机种子",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lenet",
        help='模型名称(lenet/alexnet/vgg11~19/nin/googlenet), "all" 表示全部',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help='数据集名称(mnist/cifar10/cifar100/svhn/stl10...), "all" 表示全部',
    )


def _addDataArgs(parser: argparse.ArgumentParser) -> None:
    """数据:batch size, workers, val split, 增强"""
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DataParams.BATCH_SIZE,
        help="批次大小",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DataParams.NUM_WORKERS,
        help="DataLoader 子进程数",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=DataParams.VAL_SPLIT,
        help="验证集比例",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="禁用数据增强",
    )


def _addTrainingArgs(parser: argparse.ArgumentParser) -> None:
    """训练:epochs, optimizer, lr, weight decay, scheduler, early stopping"""
    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainingParams.EPOCHS,
        help="训练轮数",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=TrainingParams.OPTIMIZER,
        choices=["adam", "adamw", "sgd", "rmsprop"],
        help="优化器",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=TrainingParams.LEARNING_RATE,
        help="学习率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=TrainingParams.WEIGHT_DECAY,
        help="权重衰减(L2)",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=TrainingParams.LR_FACTOR,
        help="LR 衰减因子",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=TrainingParams.LR_PATIENCE,
        help="LR 调度器耐心值",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=TrainingParams.LR_MIN,
        help="最小 LR",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=TrainingParams.GRAD_CLIP,
        help="梯度裁剪阈值(0=不裁剪)",
    )


def _addPathArgs(parser: argparse.ArgumentParser) -> None:
    """路径:datasets, checkpoints, logs, outputs"""
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATASETS_DIR),
        help="数据集目录",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(CHECKPOINTS_DIR),
        help="checkpoint 目录",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(LOGS_DIR),
        help="日志目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUTS_DIR),
        help="输出目录",
    )


# ═══════════════════════════════════════════════════════════════
# 主解析器
# ═══════════════════════════════════════════════════════════════


def buildParser() -> argparse.ArgumentParser:
    """构建完整的参数解析器"""

    parser = argparse.ArgumentParser(
        description="CNN 图像分类训练/评估/推理工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 全局参数(必须在子命令之前)
    _addGlobalArgs(parser)

    subparsers = parser.add_subparsers(dest="command", help="可用子命令")

    # ── train ────────────────────────────────────────────
    trainParser = subparsers.add_parser("train", help="训练模型")
    _addDataArgs(trainParser)
    _addTrainingArgs(trainParser)
    _addPathArgs(trainParser)
    trainParser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 恢复训练",
    )

    # ── eval ─────────────────────────────────────────────
    evalParser = subparsers.add_parser("eval", help="评估模型")
    _addDataArgs(evalParser)
    _addPathArgs(evalParser)
    evalParser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint 路径(必填)",
    )
    evalParser.add_argument(
        "--no-visualize",
        action="store_true",
        help="跳过可视化",
    )

    # ── infer ────────────────────────────────────────────
    inferParser = subparsers.add_parser("infer", help="单张/批量推理")
    _addPathArgs(inferParser)
    inferParser.add_argument(
        "--image",
        type=str,
        default=None,
        help="输入图片路径",
    )
    inferParser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="批量推理图片目录",
    )
    inferParser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint 路径(必填)",
    )
    inferParser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="返回 top-K 预测",
    )

    # ── benchmark ─────────────────────────────────────────
    benchmarkParser = subparsers.add_parser("benchmark", help="基准测试")
    _addDataArgs(benchmarkParser)
    _addTrainingArgs(benchmarkParser)
    _addPathArgs(benchmarkParser)

    return parser


def getSettings(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 CLI 参数并返回 Namespace"""
    parser = buildParser()
    return parser.parse_args(argv)
