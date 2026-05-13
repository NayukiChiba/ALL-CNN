# CLI 命令行系统

## 概述

ALL-CNN 提供两种交互方式：**命令行模式**（一行指令）和**交互式菜单模式**。

---

## 命令行模式

**源码**: [cnnlib/cli/parser.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/cli/parser.py)

### 基本语法

```bash
python main.py [全局参数] <子命令> [子命令参数]
```

### 全局参数

在所有子命令之前指定：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `--device` | str | cpu | 计算设备 (cpu / cuda) |
| `--seed` | int | 42 | 随机种子 |
| `--model` | str | lenet | 模型名称 |
| `--dataset` | str | mnist | 数据集名称 |

### 子命令概览

| 子命令 | 说明 | 示例 |
|------|------|------|
| `train` | 训练模型 | `python main.py train --epochs 50` |
| `eval` | 评估模型 | `python main.py eval --checkpoint best.pth` |
| `infer` | 单张/批量推理 | `python main.py infer --image cat.jpg` |
| `benchmark` | 基准测试 | `python main.py benchmark --epochs 5` |

---

## train — 训练子命令

```
python main.py [全局参数] train [训练参数]
```

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `--epochs` | int | 30 | 最大训练轮数 |
| `--batch-size` | int | 64 | 批次大小 |
| `--num-workers` | int | 4 | DataLoader 子进程数 |
| `--val-split` | float | 0.1 | 验证集比例 |
| `--no-augment` | flag | False | 禁用数据增强 |
| `--optimizer` | str | adam | 优化器 (adam/adamw/sgd/rmsprop) |
| `--lr` | float | 0.001 | 初始学习率 |
| `--weight-decay` | float | 1e-4 | 权重衰减 |
| `--lr-factor` | float | 0.5 | 调度器衰减因子 |
| `--lr-patience` | int | 3 | 调度器耐心值 |
| `--lr-min` | float | 1e-6 | 最小学习率 |
| `--grad-clip` | float | 0 | 梯度裁剪阈值 (0=不裁剪) |
| `--resume` | str | None | 从 checkpoint 恢复路径 |
| `--data-dir` | str | datasets/ | 数据集目录 |
| `--checkpoint-dir` | str | checkpoints/ | checkpoint 目录 |
| `--log-dir` | str | logs/ | 日志目录 |
| `--output-dir` | str | outputs/ | 输出目录 |

### 示例

```bash
# 完整训练示例
python main.py \
    --model vgg16 --dataset cifar10 --device cuda \
    train \
    --epochs 50 --batch-size 128 \
    --optimizer adamw --lr 0.001 --weight-decay 1e-4 \
    --lr-factor 0.5 --lr-patience 5 --grad-clip 1.0
```

---

## eval — 评估子命令

```
python main.py [全局参数] eval [评估参数]
```

| 参数 | 说明 |
|------|------|
| `--checkpoint` | checkpoint 路径（必填） |
| `--batch-size` | 批次大小 |
| `--no-visualize` | 跳过可视化图表生成 |
| `--data-dir` | 数据集目录 |
| `--output-dir` | 输出目录 |

### 示例

```bash
python main.py --model vgg16 --dataset cifar10 eval \
    --checkpoint checkpoints/best_model.pth
```

---

## infer — 推理子命令

```
python main.py [全局参数] infer [推理参数]
```

| 参数 | 说明 |
|------|------|
| `--checkpoint` | checkpoint 路径（必填） |
| `--image` | 单张图片路径 |
| `--image-dir` | 批量推理图片目录 |
| `--top-k` | 返回 top-K 预测（默认 3） |

`--image` 和 `--image-dir` 至少指定一个。

### 示例

```bash
# 单图
python main.py --model lenet --dataset mnist infer \
    --checkpoint checkpoints/best_model.pth --image digit.png

# 批量
python main.py --model vgg16 --dataset cifar10 infer \
    --checkpoint checkpoints/best_model.pth --image-dir ./test_imgs/

# Top-5
python main.py --model googlenet --dataset cifar100 infer \
    --checkpoint checkpoints/best_model.pth --image bird.jpg --top-k 5
```

---

## benchmark — 基准测试子命令

```
python main.py [全局参数] benchmark [基准测试参数]
```

| 参数 | 说明 |
|------|------|
| `--epochs` | 每组合训练轮数 |
| `--batch-size` | 批次大小 |
| `--num-workers` | DataLoader 子进程数 |
| `--optimizer` | 优化器 |
| `--lr` | 初始学习率 |
| `--weight-decay` | 权重衰减 |
| `--grad-clip` | 梯度裁剪阈值 |
| `--data-dir` | 数据集目录 |
| `--output-dir` | 输出目录 |

当 `--model all` 或 `--dataset all` 时运行全量评测。

### 示例

```bash
# 单组
python main.py --model lenet --dataset mnist benchmark --epochs 5

# 全量
python main.py benchmark --epochs 5 --batch-size 64
```

---

## 交互式 CLI

无参数运行时自动进入交互式菜单：

```bash
python main.py
```

**源码**: [cnnlib/cli/interactive.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/cli/interactive.py)

### 主菜单结构

```
╔══════════════════════════════════════════════╗
║            ALL-CNN 交互式 CLI               ║
╚══════════════════════════════════════════════╝

  ▸ 选择模型
    1. lenet      — LeNet-5 (1998)
    2. alexnet    — AlexNet (2012)
    3. vgg11      — VGG-11 (2015)
    ...
    8. googlenet  — GoogLeNet / Inception v1 (2015)

  ▸ 选择数据集
    1. mnist
    2. fashionmnist
    ...
    10. flowers102

  ▸ 选择操作
    1. 训练
    2. 评估
    3. 推理
    4. 基准测试

  ▸ 配置参数
    Epochs: 30
    Batch Size: 64
    Optimizer: adam
    ...

  ▸ 开始执行
```

### ANSI 配色

```python
class _C:
    HEADER = "\033[95m"   # 品红 — 标题
    BLUE   = "\033[94m"   # 蓝色 — 章节
    CYAN   = "\033[96m"   # 青色 — 边框
    GREEN  = "\033[92m"   # 绿色 — 值、成功
    YELLOW = "\033[93m"   # 黄色 — 警告
    RED    = "\033[91m"   # 红色 — 错误
    BOLD   = "\033[1m"    # 加粗
    DIM    = "\033[2m"    # 暗淡 — 默认值提示
    END    = "\033[0m"    # 重置
```

---

## 参数解析架构

```
main.py
  ├─ getSettings()           → argparse.Namespace
  ├─ 检查 command 字段
  ├─ 根据 command dispatch:
  │   ├─ "train"     → runTrain(args)
  │   ├─ "eval"      → runEval(args)
  │   ├─ "infer"     → runInfer(args)
  │   ├─ "benchmark" → runBenchmark(args)
  │   └─ None        → InteractiveCLI().run()
  └─ 如果 command 非空且无匹配 → 显示帮助
```

全局参数（`--model`, `--dataset`, `--device`, `--seed`）和子命令参数分层管理——全局参数在 parser 顶层，子命令参数在各个 `subparsers.add_parser()` 中定义。

---

## 相关文档

- [快速开始](/guides/quickstart) — 首个训练/评估/推理命令
- [训练指南](/guides/training-guide) — CLI 训练详解
- [推理指南](/guides/inference-guide) — CLI 推理详解
- [基准测试指南](/guides/benchmark-guide) — CLI benchmark 详解
- [架构总览](/architecture/overview) — 系统整体架构
