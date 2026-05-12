"""
训练超参数默认值：优化器、学习率调度、正则化、早停

用法:
    from config.training import TrainingParams
    from config import TrainingParams
"""


class TrainingParams:
    # 训练规模
    EPOCHS: int = 20

    # 优化器
    OPTIMIZER: str = "adam"
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-4

    # 学习率调度（ReduceLROnPlateau）
    SCHEDULER: str = "plateau"
    LR_FACTOR: float = 0.5
    LR_PATIENCE: int = 3
    LR_MIN: float = 1e-6

    # 损失函数
    LOSS: str = "cross_entropy"
    LABEL_SMOOTHING: float = 0.0

    # 早停
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_MIN_DELTA: float = 0.001

    # 梯度裁剪（0 表示不裁剪）
    GRAD_CLIP: float = 0.0
