"""
评估脚本

加载已训练模型的 checkpoint，在测试集上评估并输出完整指标.

用法:
    python main.py --model vgg16 --dataset cifar10 eval \
        --checkpoint checkpoints/vgg16/cifar10/best_model.pth
"""

import argparse

import torch

from cnnlib.data.loader import build_dataloaders
from cnnlib.evaluation.evaluator import Evaluator
from cnnlib.models.factory import create_model_for_dataset
from cnnlib.training import createLoss, loadCheckpoint


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

    # 评估 - 用 Evaluator 获得完整指标
    from cnnlib.registry.datasets import get_dataset_info

    datasetInfo = get_dataset_info(args.dataset)
    lossFn = createLoss("cross_entropy")
    evaluator = Evaluator(model, testLoader, lossFn, device, datasetInfo["num_classes"])
    metrics = evaluator.evaluate()

    print("\n测试结果:")
    print(f"  Loss:       {metrics['loss']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.2f}%")
    if "top5_accuracy" in metrics:
        print(f"  Top-5 Acc:  {metrics['top5_accuracy']:.2f}%")
    if "macro_f1" in metrics:
        print(f"  Macro F1:   {metrics['macro_f1']:.4f}")

    # 出图
    if not getattr(args, "no_visualize", False):
        try:
            from cnnlib.evaluation.visualize import generateAllCharts
            from config.paths import getVisualizationDir

            visDir = getVisualizationDir(args.model, args.dataset) / "eval"
            generateAllCharts(
                model=model,
                loader=testLoader,
                datasetInfo=datasetInfo,
                saveDir=visDir,
                device=str(device),
                titlePrefix=f"{args.model}/{args.dataset} ",
                precomputedMetrics=metrics,
            )
            print(f"图表已保存至: {visDir}")
        except Exception as e:
            print(f"  图表生成失败: {e}")
