# 基准测试指南

## 什么是 Benchmark

ALL-CNN 的 Benchmark 系统对所有模型×数据集组合进行标准化评测，输出：

- 参数量（Params）和模型大小（MB）
- 推理速度（ms/batch）
- 训练后的验证准确率和测试准确率
- 跨模型对比可视化图表

---

## CLI 方式

```bash
# 单组评测
python main.py --model vgg16 --dataset cifar10 benchmark --epochs 10

# 全量评测（所有 8 模型 × 10 数据集）
python main.py benchmark --epochs 5 --batch-size 64
```

### Benchmark 专用参数

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `--epochs` | 30 | 每组合训练轮数（benchmark 建议 5-10） |
| `--batch-size` | 64 | 批次大小 |
| `--output-dir` | outputs/benchmarks/ | 结果保存目录 |

---

## 编程方式

**源码**: [cnnlib/experiments/benchmark.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/experiments/benchmark.py)

### 单组评测

```python
from cnnlib.experiments.benchmark import runBenchmark

result = runBenchmark(
    modelName="vgg16",
    datasetName="cifar10",
    device="cuda",
    epochs=10,
    batchSize=64,
    outputDir="outputs/benchmarks/",
    visualize=True,
)

print(f"参数量: {result['params']:,}")
print(f"模型大小: {result['model_size_mb']:.1f} MB")
print(f"推理时间: {result['inference_time_ms']:.1f} ms/batch")
print(f"最佳 val_acc: {result['best_val_acc']:.2f}%")
```

返回结果:

```python
{
    "model": "vgg16",
    "dataset": "cifar10",
    "device": "cuda",
    "epochs": 10,
    "params": 134300000,
    "model_size_mb": 512.3,
    "inference_time_ms": 32.5,
    "train_samples": 45000,
    "test_samples": 10000,
    "best_val_acc": 93.45,
    "best_epoch": 8,
    "test_acc": 93.12,
    "test_loss": 0.2134,
    "history": {
        "train_loss": [...],
        "train_acc": [...],
        "val_loss": [...],
        "val_acc": [...]
    }
}
```

### 全量评测

```python
from cnnlib.experiments.benchmark import runAllBenchmarks

results = runAllBenchmarks(
    models=["lenet", "alexnet", "vgg16"],
    datasets=["mnist", "cifar10", "svhn"],
    device="cuda",
    epochs=5,
    outputDir="outputs/benchmarks/",
)
```

输出汇总表:

```
====================================================================================================
Benchmark Summary
====================================================================================================
Model        Dataset        Params      Val Acc  Test Acc  Inf Time
----------------------------------------------------------------------------------------------------
alexnet      cifar10       58,330,568  87.23%   86.91%     18.3ms
alexnet      mnist         58,330,568  99.12%   98.95%     18.1ms
alexnet      svhn          58,330,568  94.56%   94.01%     18.4ms
lenet        cifar10          61,706   72.34%   71.89%      0.9ms
lenet        mnist            61,706   99.21%   99.14%      0.8ms
lenet        svhn             61,706   88.67%   87.92%      0.9ms
vgg16        cifar10      134,300,000  93.45%   93.12%     32.5ms
vgg16        mnist        134,300,000  99.45%   99.38%     32.3ms
vgg16        svhn         134,300,000  95.23%   94.89%     32.6ms
----------------------------------------------------------------------------------------------------
```

---

## 输出文件

### 评测结果

每组合保存为一个 JSON 文件：

```
outputs/benchmarks/
├── lenet_mnist.json
├── lenet_cifar10.json
├── alexnet_mnist.json
├── ...
└── benchmark_summary.json  # 汇总表
```

### 可视化图表

全量评测自动生成跨模型对比图：

| 图表 | 文件 | 内容 |
|------|------|------|
| 参数量对比 | `benchmark_params.png` | 所有组合参数量柱状图 |
| 准确率对比 | `benchmark_accuracy.png` | Val/Test 准确率并排柱状图 |
| 推理时间对比 | `benchmark_inference.png` | 推理时间柱状图 |
| 热力图 | `benchmark_heatmap.png` | 模型×数据集 准确率热力图 |

单组评测生成：

| 图表 | 文件 | 内容 |
|------|------|------|
| 元数据汇总 | `{model}_{dataset}_summary.png` | 参数量 + 模型大小 + 推理时间三子图 |
| 训练曲线 | 保存于 `visualizations/{model}/{dataset}/benchmark/` | 完整评估图表 |

---

## 评测指标说明

| 指标 | 含义 | 如何解读 |
|------|------|---------|
| params | 可训练参数总数 | 越小越轻量 |
| model_size_mb | float32 模型文件大小 | params × 4 bytes |
| inference_time_ms | 单 batch（64 张）推理耗时 | 越小越快 |
| best_val_acc | 最佳验证准确率 | 越大泛化越好 |
| test_acc | 测试集准确率 | 最终性能指标 |

---

## 相关文档

- [模型对比](/guides/model-comparison) — 8 模型综合对比分析
- [数据集对比](/guides/dataset-comparison) — 10 数据集难度排序
- [基准测试系统](/architecture/benchmark) — benchmark 内部实现
