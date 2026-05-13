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
from config.paths import (
    ensureDir,
    getBenchmarkResultPath,
    getBenchmarkSummaryPath,
    getCheckpointDir,
    getLogDir,
)


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
    numWorkers: int = 0,
    dataDir: str = "datasets/",
    seed: int = 42,
    outputDir: Optional[str | Path] = None,
    visualize: bool = True,
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
        visualize:   是否生成可视化图表

    Returns:
        基准测试结果字典(含 history)
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

    ckptDir = ensureDir(getCheckpointDir(modelName, datasetName))
    logDir = ensureDir(getLogDir(modelName, datasetName))
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
    result["history"] = trainResult["history"]
    if trainResult["test_metrics"]:
        result["test_acc"] = trainResult["test_metrics"]["accuracy"]
        result["test_loss"] = trainResult["test_metrics"]["loss"]

    # 可视化
    if visualize:
        try:
            from cnnlib.evaluation.visualize import generateAllCharts
            from cnnlib.registry.datasets import get_dataset_info
            from config.paths import getVisualizationDir

            datasetInfo = get_dataset_info(datasetName)
            visDir = getVisualizationDir(modelName, datasetName) / "benchmark"
            generateAllCharts(
                model=model,
                loader=testLoader,
                datasetInfo=datasetInfo,
                saveDir=visDir,
                history=result["history"],
                device=device,
                titlePrefix=f"{modelName}/{datasetName} ",
            )
        except Exception as e:
            print(f"  可视化生成失败: {e}")

    # 保存结果（不含 history，JSON 太大）
    if outputDir is not None:
        outputPath = (
            ensureDir(Path(outputDir) / modelName / datasetName) / "result.json"
        )
    else:
        outputPath = getBenchmarkResultPath(modelName, datasetName)
        ensureDir(outputPath.parent)
    saveResult = {k: v for k, v in result.items() if k != "history"}
    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(saveResult, f, ensure_ascii=False, indent=2)

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
        from cnnlib.registry.datasets import list_datasets

        datasets = list_datasets()

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
    _printSummary(results)
    return results


def _printSummary(results: List[Dict]) -> None:
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

    summaryPath = getBenchmarkSummaryPath()
    ensureDir(summaryPath.parent)
    with open(summaryPath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存至: {summaryPath}")


# ═══════════════════════════════════════════════════════════════
# 基准测试专用可视化（跨模型对比，非单模型评估）
# ═══════════════════════════════════════════════════════════════


def _autoLabel(results: List[Dict]) -> List[str]:
    return [f"{r['model']}\n{r['dataset']}" for r in results]


def plotBenchmarkSingle(result: Dict, savePath: Optional[str | Path] = None) -> None:
    """
    单组基准测试元数据汇总图（参数量、推理时间、模型大小）

    Args:
        result:   runBenchmark() 的结果
        savePath: 保存路径（目录或完整路径）
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    savePath = Path(savePath) if savePath else None
    if savePath and savePath.is_dir():
        savePath = savePath / f"{result['model']}_{result['dataset']}_summary.png"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 参数量
    axes[0].bar(["Params"], [result["params"] / 1e6], color="#3498db")
    axes[0].set_ylabel("Million")
    axes[0].set_title(f"参数量: {result['params']:,}")

    # 模型大小
    axes[1].bar(["Size"], [result["model_size_mb"]], color="#e74c3c")
    axes[1].set_ylabel("MB")
    axes[1].set_title(f"模型大小: {result['model_size_mb']:.1f} MB")

    # 推理时间
    axes[2].bar(["Inference"], [result["inference_time_ms"]], color="#2ecc71")
    axes[2].set_ylabel("ms/batch")
    axes[2].set_title(f"推理时间: {result['inference_time_ms']:.1f} ms")

    fig.suptitle(f"{result['model']} + {result['dataset']} Summary")
    fig.tight_layout()

    if savePath:
        savePath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(savePath), dpi=150, bbox_inches="tight")
        print(f"  → {savePath}")
    plt.close(fig)


def plotBenchmarkAll(results: List[Dict], outputDir: str | Path) -> None:
    """
    全量基准测试跨模型对比图

    Args:
        results:   runAllBenchmarks() 的结果列表
        outputDir: 输出目录
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    matplotlib.use("Agg")

    if not results:
        print("无数据，跳过出图")
        return

    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    labels = _autoLabel(results)
    models = sorted(set(r["model"] for r in results))
    colors = plt.cm.tab10.colors
    modelColors = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    print("\n生成基准测试图表...")

    # 1. 参数量对比
    fig, ax = plt.subplots(figsize=(max(12, len(results) * 0.8), 6))
    barColors = [modelColors[r["model"]] for r in results]
    ax.bar(labels, [r["params"] / 1e6 for r in results], color=barColors)
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Model Parameter Count")
    ax.tick_params(axis="x", labelsize=7)
    legendPatches = [Patch(color=c, label=m) for m, c in modelColors.items()]
    ax.legend(handles=legendPatches, fontsize=7)
    fig.tight_layout()
    path = outputDir / "benchmark_params.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

    # 2. 准确率对比
    valAccs = [r.get("best_val_acc", 0) for r in results]
    testAccs = [r.get("test_acc", 0) for r in results]
    x = np.arange(len(results))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(results) * 0.8), 6))
    ax.bar(x - width / 2, valAccs, width, label="Val Acc", color="#3498db")
    ax.bar(x + width / 2, testAccs, width, label="Test Acc", color="#2ecc71")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = outputDir / "benchmark_accuracy.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

    # 3. 推理时间对比
    fig, ax = plt.subplots(figsize=(max(12, len(results) * 0.8), 6))
    barColors = [modelColors[r["model"]] for r in results]
    ax.bar(labels, [r["inference_time_ms"] for r in results], color=barColors)
    ax.set_ylabel("Inference Time (ms/batch)")
    ax.set_title("Inference Time Comparison")
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(handles=legendPatches, fontsize=7)
    fig.tight_layout()
    path = outputDir / "benchmark_inference.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

    # 4. 热力图
    modelsSorted = sorted(set(r["model"] for r in results))
    datasetsSorted = sorted(set(r["dataset"] for r in results))
    matrix = np.full((len(modelsSorted), len(datasetsSorted)), np.nan)
    for r in results:
        mi = modelsSorted.index(r["model"])
        di = datasetsSorted.index(r["dataset"])
        matrix[mi, di] = r.get("best_val_acc", 0)

    fig, ax = plt.subplots(
        figsize=(len(datasetsSorted) * 1.2 + 4, len(modelsSorted) * 0.8 + 2)
    )
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    for i in range(len(modelsSorted)):
        for j in range(len(datasetsSorted)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if 30 < val < 70 else "white",
                )
    ax.set_xticks(range(len(datasetsSorted)))
    ax.set_xticklabels(datasetsSorted, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(modelsSorted)))
    ax.set_yticklabels(modelsSorted, fontsize=9)
    ax.set_title("Val Accuracy Heatmap (models x datasets)", fontsize=13)
    plt.colorbar(im, ax=ax, label="Val Acc (%)")
    fig.tight_layout()
    path = outputDir / "benchmark_heatmap.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

    print(f"图表已保存至: {outputDir}")


if __name__ == "__main__":
    import sys

    device = DefaultParams.DEVICE
    print(f"设备: {device}")

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        runAllBenchmarks(device=device, epochs=3)
    elif len(sys.argv) > 1:
        model = sys.argv[1]
        dataset = sys.argv[2] if len(sys.argv) > 2 else "cifar10"
        runBenchmark(model, dataset, device=device, epochs=3)
    else:
        print("用法:")
        print("  单组: python -m cnnlib.experiments.benchmark <model> <dataset>")
        print("  全量: python -m cnnlib.experiments.benchmark --all")
        print("示例: python -m cnnlib.experiments.benchmark lenet mnist")
