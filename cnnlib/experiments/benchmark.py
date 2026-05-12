"""
基准测试

对所有模型 x 数据集组合进行标准化评测,对比参数量、训练速度、准确率等指标.

用法:
    from cnnlib.experiments.benchmark import runBenchmark

    result = runBenchmark("vgg16", "cifar10", epochs=5)
    # 或直接命令行:
    python -m cnnlib.experiments.benchmark
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnnlib.data.loader import build_dataloaders
from cnnlib.models.factory import create_model_for_dataset
from cnnlib.registry.models import list_models
from cnnlib.training import (
    EarlyStopping,
    Trainer,
    TrainingLogger,
    createLoss,
    createOptimizer,
    createScheduler,
)
from config.defaults import DefaultParams
from config.paths import ensureDir


def _countParams(model: nn.Module) -> int:
    """参数量"""
    return sum(p.numel() for p in model.parameters())


def _measureInferenceTime(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int = 5,
    repeats: int = 50,
) -> float:
    """测量单 batch 平均推理时间(ms)"""
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(device)

    # 预热
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / repeats) * 1000  # ms per batch


def runBenchmark(
    modelName: str,
    datasetName: str,
    device: str = DefaultParams.DEVICE,
    epochs: int = 5,
    batchSize: int = 64,
    numWorkers: int = 4,
    dataDir: str = "datasets/",
    seed: int = 42,
    outputDir: Optional[str | Path] = None,
) -> Dict:
    """
    对单个(模型, 数据集)组合运行基准测试

    Args:
        modelName:   模型名称
        datasetName: 数据集名称
        device:      计算设备
        epochs:      训练轮数
        batchSize:   批次大小
        numWorkers:  DataLoader 子进程数
        dataDir:     数据目录
        seed:        随机种子
        outputDir:   结果输出目录(可选)

    Returns:
        基准测试结果字典
    """
    deviceObj = torch.device(device)
    result = {
        "model": modelName,
        "dataset": datasetName,
        "device": device,
        "epochs": epochs,
    }

    # 模型
    model = create_model_for_dataset(modelName, datasetName, device=device)
    result["params"] = _countParams(model)
    result["model_size_mb"] = result["params"] * 4 / (1024 * 1024)  # float32

    # 数据
    trainLoader, valLoader, testLoader = build_dataloaders(
        model_name=modelName,
        dataset_name=datasetName,
        batch_size=batchSize,
        num_workers=numWorkers,
        val_split=0.1,
        data_dir=dataDir,
        seed=seed,
    )
    result["train_samples"] = len(trainLoader.dataset)
    result["test_samples"] = len(testLoader.dataset)

    # 推理速度
    result["inference_time_ms"] = _measureInferenceTime(model, testLoader, deviceObj)

    # 训练
    lossFn = createLoss("cross_entropy")
    optimizer = createOptimizer(model, "adam", lr=0.001, weight_decay=1e-4)
    scheduler = createScheduler(optimizer, "plateau", factor=0.5, patience=2)

    with torch.TemporaryDirectory() as tmp:
        ckptDir = Path(tmp) / "checkpoints"
        logDir = Path(tmp) / "logs"
        logger = TrainingLogger(logDir, modelName, datasetName)
        es = EarlyStopping(patience=5)

        trainer = Trainer(
            model=model,
            trainLoader=trainLoader,
            valLoader=valLoader,
            testLoader=testLoader,
            optimizer=optimizer,
            scheduler=scheduler,
            lossFn=lossFn,
            device=deviceObj,
            epochs=epochs,
            checkpointDir=ckptDir,
            logger=logger,
            earlyStopping=es,
        )

        trainResult = trainer.train()

    result["best_val_acc"] = trainResult["best_metric"]
    result["best_epoch"] = trainResult["best_epoch"]
    if trainResult["test_metrics"]:
        result["test_acc"] = trainResult["test_metrics"]["accuracy"]
        result["test_loss"] = trainResult["test_metrics"]["loss"]

    # 保存结果
    if outputDir is not None:
        outputDir = ensureDir(Path(outputDir))
        outputFile = outputDir / f"{modelName}_{datasetName}.json"
        with open(outputFile, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def runAllBenchmarks(
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    device: str = DefaultParams.DEVICE,
    epochs: int = 5,
    batchSize: int = 64,
    dataDir: str = "datasets/",
    outputDir: Optional[str | Path] = None,
) -> List[Dict]:
    """
    对所有模型 x 数据集组合运行基准测试

    Args:
        models:    模型名称列表(None=全部)
        datasets:  数据集名称列表(None=全部)
        device:    计算设备
        epochs:    每组合训练轮数
        batchSize: 批次大小
        dataDir:   数据目录
        outputDir: 结果输出目录

    Returns:
        所有结果列表
    """
    if models is None:
        models = list_models()
    if datasets is None:
        # 仅选匹配的小数据集
        datasets = ["mnist", "fashionmnist", "cifar10", "cifar100"]

    results = []
    total = len(models) * len(datasets)

    for i, modelName in enumerate(models):
        for j, datasetName in enumerate(datasets):
            idx = i * len(datasets) + j + 1
            print(f"\n[{idx}/{total}] {modelName} + {datasetName}")

            try:
                result = runBenchmark(
                    modelName=modelName,
                    datasetName=datasetName,
                    device=device,
                    epochs=epochs,
                    batchSize=batchSize,
                    dataDir=dataDir,
                    outputDir=outputDir,
                )
                results.append(result)
                print(
                    f"  params={result['params']:,}, "
                    f"val_acc={result.get('best_val_acc', 0):.2f}%, "
                    f"inf_time={result['inference_time_ms']:.1f}ms"
                )
            except Exception as e:
                print(f"  跳过: {e}")
                continue

    # 汇总表
    _printSummary(results, outputDir)
    return results


def _printSummary(results: List[Dict], outputDir: Optional[str | Path] = None) -> None:
    """打印基准测试汇总表"""
    if not results:
        print("\n无有效结果")
        return

    print("\n" + "=" * 100)
    print("Benchmark Summary")
    print("=" * 100)
    print(
        f"{'Model':<12} {'Dataset':<14} {'Params':<10} {'Val Acc':<8} {'Test Acc':<8} {'Inf Time':<10}"
    )
    print("-" * 100)

    for r in sorted(results, key=lambda x: (x["model"], x["dataset"])):
        print(
            f"{r['model']:<12} {r['dataset']:<14} "
            f"{r['params']:>8,}  "
            f"{r.get('best_val_acc', 0):>6.2f}%  "
            f"{r.get('test_acc', 0):>6.2f}%  "
            f"{r['inference_time_ms']:>8.1f}ms"
        )

    print("-" * 100)

    if outputDir:
        outputDir = Path(outputDir)
        summaryFile = outputDir / "benchmark_summary.json"
        with open(summaryFile, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n汇总已保存至: {summaryFile}")


if __name__ == "__main__":
    import sys

    device = DefaultParams.DEVICE
    print(f"设备: {device}")

    if len(sys.argv) > 1:
        model = sys.argv[1]
        dataset = sys.argv[2] if len(sys.argv) > 2 else "cifar10"
        runBenchmark(
            model, dataset, device=device, epochs=3, outputDir="outputs/benchmarks/"
        )
    else:
        # 不指定参数时,跑一个快速单例
        print("用法: python -m cnnlib.experiments.benchmark <model> <dataset>")
        print("示例: python -m cnnlib.experiments.benchmark lenet mnist")
