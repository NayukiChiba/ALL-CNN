"""
config/settings.py

Argument definitions and parsing. All CLI arguments are defined here;
main.py only handles dispatch.

Usage:
    from config.settings import getSettings
    args = getSettings()

CLI convention:
    python main.py [global args] <subcommand> [subcommand args]
    python main.py --device cuda --seed 42 train --epochs 50
    python main.py --config my_config.yaml eval --checkpoint best.pth
"""

import argparse

from config.default_params import DataParams, DefaultParams, ModelParams, TrainingParams
from config.paths import CHECKPOINTS_DIR, DATASETS_DIR, LOGS_DIR, OUTPUTS_DIR

# ===================== Shared argument groups =====================
# These add arguments to whichever parser they receive (train / eval / infer).
# They only contain subcommand-specific args; global args are on the parent parser.


def _addDataArgs(parser: argparse.ArgumentParser) -> None:
    """Data: batch size, workers, val split, augmentation"""
    parser.add_argument(
        "--batch-size", type=int, default=DataParams.BATCH_SIZE, help="batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DataParams.NUM_WORKERS,
        help="number of DataLoader workers",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=DataParams.VAL_SPLIT,
        help="validation split ratio",
    )
    parser.add_argument(
        "--no-augment", action="store_true", help="disable data augmentation"
    )


def _addModelArgs(parser: argparse.ArgumentParser) -> None:
    """Model: conv channels, FC size, dropout"""
    parser.add_argument(
        "--conv-channels",
        type=int,
        nargs="+",
        default=ModelParams.CONV_CHANNELS,
        help="conv layer output channels",
    )
    parser.add_argument(
        "--fc-hidden-size",
        type=int,
        default=ModelParams.FC_HIDDEN_SIZE,
        help="FC hidden layer size",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=ModelParams.DROPOUT_RATE,
        help="dropout rate",
    )


def _addTrainingArgs(parser: argparse.ArgumentParser) -> None:
    """Training: epochs, lr, weight decay, scheduler"""
    parser.add_argument(
        "--epochs", type=int, default=TrainingParams.EPOCHS, help="number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=TrainingParams.LEARNING_RATE, help="learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=TrainingParams.WEIGHT_DECAY,
        help="weight decay (L2)",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=TrainingParams.LR_FACTOR,
        help="LR reduction factor",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=TrainingParams.LR_PATIENCE,
        help="LR scheduler patience",
    )
    parser.add_argument(
        "--lr-min", type=float, default=TrainingParams.LR_MIN, help="minimum LR"
    )


def _addPathArgs(parser: argparse.ArgumentParser) -> None:
    """Paths: datasets, checkpoints, logs, outputs"""
    parser.add_argument(
        "--data-dir", type=str, default=str(DATASETS_DIR), help="dataset directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(CHECKPOINTS_DIR),
        help="checkpoint directory",
    )
    parser.add_argument(
        "--log-dir", type=str, default=str(LOGS_DIR), help="log directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUTS_DIR), help="output directory"
    )


# ===================== Main parser with subcommands =====================


def buildParser() -> argparse.ArgumentParser:
    """Build the argument parser with global args and train/eval/infer subcommands"""

    parser = argparse.ArgumentParser(
        description="MNIST-CNN: CNN for handwritten digit recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- global arguments ----
    # Defined on the parent parser, so they must appear BEFORE the subcommand:
    #   python main.py --device cuda --seed 42 train --epochs 50
    parser.add_argument(
        "--device",
        type=str,
        default=DefaultParams.DEVICE,
        choices=["cpu", "cuda"],
        help="compute device",
    )
    parser.add_argument(
        "--seed", type=int, default=DefaultParams.SEED, help="random seed"
    )

    subparsers = parser.add_subparsers(dest="command", help="available subcommands")

    # ---- train ----
    trainParser = subparsers.add_parser("train", help="train the model")
    _addDataArgs(trainParser)
    _addModelArgs(trainParser)
    _addTrainingArgs(trainParser)
    _addPathArgs(trainParser)
    trainParser.add_argument(
        "--resume", type=str, default=None, help="resume from checkpoint path"
    )

    # ---- eval ----
    evalParser = subparsers.add_parser("eval", help="evaluate the model")
    _addDataArgs(evalParser)
    _addPathArgs(evalParser)
    evalParser.add_argument(
        "--checkpoint", type=str, required=True, help="checkpoint path for evaluation"
    )
    evalParser.add_argument(
        "--no-visualize",
        action="store_true",
        help="skip generating visualization plots",
    )

    # ---- infer ----
    inferParser = subparsers.add_parser("infer", help="run inference on an image")
    _addModelArgs(inferParser)
    _addPathArgs(inferParser)
    inferParser.add_argument("--image", type=str, default=None, help="input image path")
    inferParser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="directory of images for batch inference",
    )
    inferParser.add_argument(
        "--checkpoint", type=str, required=True, help="checkpoint path for inference"
    )
    inferParser.add_argument(
        "--top-k", type=int, default=3, help="return top-K predictions"
    )

    return parser


def getSettings(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and return settings namespace"""
    parser = buildParser()
    args = parser.parse_args(argv)
    return args
