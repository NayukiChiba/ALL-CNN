# 基准测试系统

## 概述

Benchmark 系统（[cnnlib/experiments/benchmark.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/experiments/benchmark.py)）对所有模型×数据集组合进行标准化评测。

**评测指标**:
- 参数量（Params）和模型大小（MB）
- 推理速度（ms/batch）
- 训练后验证集和测试集准确率
- 训练历史曲线

---

## 核心函数

### runBenchmark

```python
def runBenchmark(
    modelName: str,       # 模型名称
    datasetName: str,     # 数据集名称
    device: str,          # 计算设备
    epochs: int,          # 训练轮数
    batchSize: int,       # 批次大小
    numWorkers: int,      # DataLoader 子进程数
    dataDir: str,         # 数据目录
    seed: int,            # 随机种子
    outputDir: Optional[str | Path],  # 输出目录
    visualize: bool,      # 是否生成可视化图表
) -> Dict
```

**评测流程**:

```
1. 创建模型    — create_model_for_dataset(modelName, datasetName)
2. 统计参数量  — sum(p.numel())
3. 构建数据    — build_dataloaders()
4. 测量推理速度 — _measureInferenceTime() (warmup 5 + repeat 50)
5. 创建训练组件 — lossFn + optimizer + scheduler + Trainer
6. 训练        — trainer.train()
7. 单组可视化  — generateAllCharts() / 元数据汇总图
8. 保存结果    — JSON
```

### runAllBenchmarks

```python
def runAllBenchmarks(
    models: Optional[List[str]],    # None = 全部 8 模型
    datasets: Optional[List[str]],  # None = 全部 10 数据集
    device: str,
    epochs: int,
    batchSize: int,
    dataDir: str,
    outputDir: Optional[str | Path],
) -> List[Dict]
```

双层循环遍历所有模型×数据集组合，逐个调用 `runBenchmark()`。异常不中断——单个组合失败不影响其他组合。

---

## 推理速度测量

**源码**: [benchmark.py:44-72](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/experiments/benchmark.py#L44-L72)

```python
def _measureInferenceTime(model, loader, device, warmup=5, repeats=50):
    model.eval()
    images, _ = next(iter(loader))   # 取一个 batch
    images = images.to(device)

    # 预热（GPU kernel 编译、缓存初始化）
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()      # 确保 GPU 操作完成

    start = time.perf_counter()
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / repeats) * 1000  # ms per batch
```

- `warmup=5`: 排除首次推理的 GPU kernel 编译预热开销
- `repeats=50`: 多次取平均减少噪声
- `torch.cuda.synchronize()`: 确保异步 GPU 操作完成后再计时

---

## 可视化输出

### 单组评测可视化

调用 `plotBenchmarkSingle()` 生成三子图：

<div style="max-width:520px;margin:1em auto;font-size:13px;line-height:1.8;">
  <div style="text-align:center;font-weight:600;margin-bottom:8px;">单组 Benchmark 元数据汇总（VGG16-CIFAR10 示例）</div>
  <div style="display:flex;align-items:center;">
    <span style="width:90px;text-align:right;margin-right:8px;flex-shrink:0;">参数量</span>
    <span style="height:14px;background:#3498db;width:22.33%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">134M</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:90px;text-align:right;margin-right:8px;flex-shrink:0;">模型大小</span>
    <span style="height:14px;background:#3498db;width:85.33%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">512MB</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:90px;text-align:right;margin-right:8px;flex-shrink:0;">推理时间</span>
    <span style="height:14px;background:#3498db;width:5.42%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">32.5ms</span>
  </div>
</div>

- 参数量（M）柱状图
- 模型大小（MB）柱状图  
- 推理时间（ms）柱状图

同时调用 `generateAllCharts()` 生成完整的训练曲线、混淆矩阵、错误样本等（与评估系统相同）。

### 全量评测可视化

调用 `plotBenchmarkAll()` 生成跨模型对比图：



1. **参数量对比** — 所有组合的参数量柱状图（按模型着色）
2. **准确率对比** — Val/Test 准确率并排柱状图
3. **推理时间对比** — 推理时间柱状图（按模型着色）
4. **热力图** — 模型×数据集准确率矩阵（RdYlGn 配色）

---

## 结果输出

### JSON 格式

```json
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
  "test_loss": 0.2134
}
```

（`history` 字段从保存的 JSON 中排除以减小文件体积）

### 目录结构

```
outputs/benchmarks/
├── lenet_mnist.json
├── lenet_cifar10.json
├── alexnet_mnist.json
├── ...
├── benchmark_summary.json
├── benchmark_params.png
├── benchmark_accuracy.png
├── benchmark_inference.png
└── benchmark_heatmap.png
```

---

## CLI 入口

```bash
# 单组
python main.py --model vgg16 --dataset cifar10 benchmark --epochs 10

# 全量
python main.py benchmark --epochs 5 --batch-size 64
```

**源码**: [cnnlib/cli/parser.py:240-243](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/cli/parser.py#L240-L243)

---

## 相关文档

- [基准测试指南](/guides/benchmark-guide) — 使用方式和结果解读
- [模型对比](/guides/model-comparison) — 8 模型参数量/准确率/推理速度对比
- [数据集对比](/guides/dataset-comparison) — 10 数据集难度排序
- [训练流程](/architecture/training) — Trainer 编排类
