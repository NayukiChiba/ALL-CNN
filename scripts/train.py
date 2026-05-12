"""
训练脚本

编排从 CLI 参数到训练完成的完整流程:
    1. 创建模型
    2. 构建 DataLoader
    3. 配置损失/优化器/调度器/早停/日志
    4. 启动训练

用法:
    python main.py --model vgg16 --dataset cifar10 train --epochs 50 --lr 0.01
"""

import argparse

import torch

from cnnlib.data.loader import build_dataloaders
from cnnlib.models.factory import create_model_for_dataset
from cnnlib.training import (
    EarlyStopping,
    Trainer,
    TrainingLogger,
    createLoss,
    createOptimizer,
    createScheduler,
)
from config.paths import ensureDir, getBestModelPath, getCheckpointDir, getLogDir


def run(args: argparse.Namespace) -> None:
    """训练入口"""

    device = torch.device(args.device)

    # 模型
    model = create_model_for_dataset(args.model, args.dataset, device=str(device))

    # 数据
    enableAug = not getattr(args, "no_augment", False)
    trainLoader, valLoader, testLoader = build_dataloaders(
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        seed=args.seed,
    )
    # 如果禁用增强，覆盖 train transform
    if not enableAug:
        from cnnlib.data.transform import build_transform

        trainLoader.dataset.dataset.transform = build_transform(
            args.model, args.dataset, augment=False
        )

    # 损失 / 优化器 / 调度器
    lossFn = createLoss("cross_entropy")
    optimizer = createOptimizer(
        model,
        name=getattr(args, "optimizer", "adam"),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = createScheduler(
        optimizer,
        name="plateau",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )

    # 日志
    logDir = getLogDir(args.model, args.dataset)
    ensureDir(logDir)
    logger = TrainingLogger(logDir, args.model, args.dataset)

    # 早停
    earlyStopping = EarlyStopping(patience=10, minDelta=0.001, mode="max")

    # checkpoint 目录
    checkpointDir = getCheckpointDir(args.model, args.dataset)
    ensureDir(checkpointDir)

    # 训练器
    trainer = Trainer(
        model=model,
        trainLoader=trainLoader,
        valLoader=valLoader,
        testLoader=testLoader,
        optimizer=optimizer,
        scheduler=scheduler,
        lossFn=lossFn,
        device=device,
        epochs=args.epochs,
        checkpointDir=checkpointDir,
        logger=logger,
        earlyStopping=earlyStopping,
        gradClip=getattr(args, "grad_clip", 0.0),
        resumeFrom=args.resume if hasattr(args, "resume") else None,
    )

    result = trainer.train()

    bestPath = getBestModelPath(args.model, args.dataset)
    print(f"\n训练完成，最佳模型: {bestPath}")
    print(f"最佳 val_acc: {result['best_metric']:.2f}% (epoch {result['best_epoch']})")
    if result["test_metrics"]:
        print(
            f"测试集 loss={result['test_metrics']['loss']:.4f} "
            f"acc={result['test_metrics']['accuracy']:.2f}%"
        )
