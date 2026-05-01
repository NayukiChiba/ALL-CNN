"""
config/default_params.py
the default Hyperparameters for training and testing the model.

"""

import torch


class DefaultParams:
    # random seed
    SEED: int = 42

    # device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class DataParams:
    # batch size
    BATCH_SIZE: int = 64

    # number of workers for data loading
    NUM_WORKERS: int = 4

    # gpu accerelation for data loading
    PIN_MEMORY: bool = True

    # validation split ratio
    VAL_SPLIT: float = 0.1

    # the information of the MNIST dataset
    # the mean and std of the MNIST dataset for normalization
    MNIST_MEAN: float = 0.1307
    MNIST_STD: float = 0.3081

    # data augmentation
    ENABLE_DATA_AUGMENTATION: bool = True

    # randomly transformation
    ROTATION_DEGREES: int = 10
    TRANSLATION_RATIO: float = 0.1


class ModelParams:
    # the number of convolutional layers output channels
    CONV_CHANNELS: list = [32, 64]

    # the size of convolutional kernels
    CONV_KERNEL_SIZE: int = 3

    # mlp hidden layer size
    FC_HIDDEN_SIZE: int = 128

    # dropout rate
    DROPOUT_RATE: float = 0.5


class TrainingParams:
    # number of epochs
    EPOCHS: int = 20

    # learning rate of adam optimizer
    LEARNING_RATE: float = 0.001

    # weight decay for regularization
    WEIGHT_DECAY: float = 1e-4

    # ReduceLROnPlateau scheduler parameters
    LR_FACTOR: float = 0.5

    # if the validation loss does not improve for this many epochs, reduce the learning rate
    LR_PATIENCE: int = 3

    # the minimum learning rate after reduction
    LR_MIN: float = 1e-6


if __name__ == "__main__":
    print(f"DefaultParams: {DefaultParams.DEVICE}")
    print(f"DataParams: {DataParams.BATCH_SIZE}")
    print(f"ModelParams: {ModelParams.CONV_CHANNELS}")
    print(f"TrainingParams: {TrainingParams.EPOCHS}")
