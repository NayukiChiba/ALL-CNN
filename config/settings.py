"""
config/settings.py

merge the default parameters and the args parameters in command line arguments.

function:
- define argparse parser based on the default_params.py
- the default values are automatically aligned
- the parameters passed in through the command line override the default values
- Path parameters can override the default directory form the command line.

usage instructions:
    python main.py                           # 使用全部默认值
    python main.py --epochs 50 --lr 0.0005   # 覆盖训练参数
    python main.py --mode eval               # 切换模式

use it in code:
    from config.settings import getSettings
    args = getSettings()
    batchSize = args.batch_size
"""

import argparse

from config.default_params import (
    DataParams,
    DefaultParams,
    ModelParams,
    TrainingParams,
)
from config.paths import (
    CHECKPOINTS_DIR,
    DATASETS_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
)


def buildParser() -> argparse.ArgumentParser:
    """
    build the argparse parser based on the default parameters and paths.


    """
    parser = argparse.ArgumentParser(
        description="MNIST-CNN: A simple CNN for MNIST classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # running mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "inference"],
        help="the running mode of the program",
    )

    # general parameters
    general = parser.add_argument_group("General Parameters")
    general.add_argument(
        "--seed",
        type=int,
        default=DefaultParams.SEED,
        help="the random seed for reproducibility",
    )
    general.add_argument(
        "--device",
        type=str,
        default=DefaultParams.DEVICE,
        help="the device to run the model on",
    )

    # data parameters
    data = parser.add_argument_group("Data Parameters")
    data.add_argument(
        "--batch_size",
        type=int,
        default=DataParams.BATCH_SIZE,
        help="the batch size for training and evaluation",
    )
    data.add_argument(
        "--num_workers",
        type=int,
        default=DataParams.NUM_WORKERS,
        help="the number of workers for data loading",
    )
    data.add_argument(
        "--val_split",
        type=float,
        default=DataParams.VAL_SPLIT,
        help="the validation split ratio",
    )
    data.add_argument(
        "--no-augment",
        action="store_true",
        help="禁用数据增强（默认开启）",
    )

    # model parameters
    model = parser.add_argument_group("Model Parameters")
    model.add_argument(
        "--conv-channels",
        type=int,
        nargs="+",
        default=ModelParams.CONV_CHANNELS,
        help="the number of channels for each convolutional layer",
    )
    model.add_argument(
        "--fc-hidden-sizes",
        type=int,
        default=ModelParams.FC_HIDDEN_SIZE,
        help="the number of hidden units for each fully connected layer",
    )

    model.add_argument(
        "--dropout-rate",
        type=float,
        default=ModelParams.DROPOUT_RATE,
        help="the dropout rate for regularization",
    )

    # training parameters
    training = parser.add_argument_group("Training Parameters")
    training.add_argument(
        "--epochs",
        type=int,
        default=TrainingParams.EPOCHS,
        help="the number of epochs to train the model",
    )
    training.add_argument(
        "--lr",
        type=float,
        default=TrainingParams.LEARNING_RATE,
        help="the learning rate for the optimizer",
    )

    training.add_argument(
        "--weight-decay",
        type=float,
        default=TrainingParams.WEIGHT_DECAY,
        help="the weight decay for regularization",
    )
    training.add_argument(
        "--lr-factor",
        type=float,
        default=TrainingParams.LR_FACTOR,
        help="learning rate reduction factor",
    )
    training.add_argument(
        "--lr-patience",
        type=int,
        default=TrainingParams.LR_PATIENCE,
        help="number of epochs to wait for improvement",
    )
    training.add_argument(
        "--lr-min",
        type=float,
        default=TrainingParams.LR_MIN,
        help="minimum learning rate",
    )

    # paths parameters
    pathGroup = parser.add_argument_group("Path Parameters")
    pathGroup.add_argument(
        "--datasets_dir",
        type=str,
        default=str(DATASETS_DIR),
        help="the directory for datasets",
    )
    pathGroup.add_argument(
        "--checkpoints_dir",
        type=str,
        default=str(CHECKPOINTS_DIR),
        help="the directory for model checkpoints",
    )
    pathGroup.add_argument(
        "--outputs_dir",
        type=str,
        default=str(OUTPUTS_DIR),
        help="the directory for outputs",
    )
    pathGroup.add_argument(
        "--logs_dir", type=str, default=str(LOGS_DIR), help="the directory for logs"
    )

    return parser


def getSettings(argv=None) -> argparse.Namespace:
    """
    get the settings from the command line arguments.
    Args:
        argv: the command line arguments, if None, use sys.argv
    Returns:
        argparse.Namespace: the settings from the command line arguments
    """
    parser = buildParser()
    args = parser.parse_args(argv)
    # --no-augment is a boolean flag, if it is set, then args.augment is False, otherwise it is True
    args.augment = not args.no_augment
    return args


if __name__ == "__main__":
    settings = getSettings()
    for key, value in vars(settings).items():
        print(f"{key}: {value}")
