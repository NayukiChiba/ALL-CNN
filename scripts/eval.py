"""
评估脚本

加载已训练模型的 checkpoint，在测试集上评估并输出指标。

用法:
    python main.py --model vgg16 --dataset cifar10 eval \
        --checkpoint checkpoints/vgg16/cifar10/best_model.pth
"""

import argparse

import torch

from cnnlib.data.loader import build_dataloaders
from cnnlib.models.factory import create_model_for_dataset
from cnnlib.training import createLoss, loadCheckpoint, validate


def run(args: argparse.Namespace) -> None:
    """评估入口"""

    device = torch.device(args.device)

    # checkpoint 必填
    checkpointPath = getattr(args, "checkpoint", None)
    if checkpointPath is None:
        print("错误: --checkpoint 为必填参数")
        return

    # 模型
    model = create_model_for_dataset(args.model, args.dataset, device=str(device))

    # 数据（仅测试集）
    _, _, testLoader = build_dataloaders(
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        seed=args.seed,
    )

    # 加载 checkpoint
    ckpt = loadCheckpoint(checkpointPath, model, device=str(device))
    print(f"加载 checkpoint: {checkpointPath}")
    print(f"  epoch: {ckpt['epoch']}, best_metric: {ckpt['best_metric']:.4f}")

    # 评估
    lossFn = createLoss("cross_entropy")
    metrics = validate(model, testLoader, lossFn, device, desc="Test")

    print("\n测试结果:")
    print(f"  Loss:     {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")

    # 可视化（可选）
    if not getattr(args, "no_visualize", False):
        _visualizePredictions(model, testLoader, device, args)


def _visualizePredictions(model, loader, device, args) -> None:
    """可视化若干预测结果"""
    import matplotlib.pyplot as plt
    import numpy as np

    from cnnlib.registry.datasets import get_dataset_info

    info = get_dataset_info(args.dataset)
    mean = np.array(info["mean"])
    std = np.array(info["std"])

    model.eval()
    images, labels = next(iter(loader))
    images = images[:8].to(device)
    labels = labels[:8]

    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)

    images = images.cpu()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(8):
        img = images[i].permute(1, 2, 0).numpy()
        img = img * std + mean  # 反归一化
        img = np.clip(img, 0, 1)

        axes[i].imshow(img if img.shape[-1] == 3 else img[:, :, 0], cmap="gray")
        axes[i].set_title(f"True: {labels[i].item()} | Pred: {preds[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()

    from config.paths import ensureDir, getVisualizationDir

    visDir = getVisualizationDir(args.model, args.dataset)
    ensureDir(visDir)
    savePath = visDir / "eval_predictions.png"
    plt.savefig(str(savePath))
    plt.close()
    print(f"可视化保存至: {savePath}")
