# 项目总体架构

## 目录结构

```
MNIST-CNN/
├── main.py                       # 唯一入口，子命令分发
├── pyproject.toml                # 项目元数据、依赖、Ruff 配置
│
├── config/                       # 配置层
│   ├── default_params.py         # 超参数默认值（4 组 dataclass-like）
│   ├── paths.py                  # 项目路径常量（基于 ROOT_DIR）
│   └── settings.py               # CLI 参数解析（argparse）
│
├── src/                          # 核心源代码
│   ├── data/                     # 数据层
│   │   ├── dataset.py            # MNIST 数据集封装
│   │   ├── transform.py          # 预处理/增强管线
│   │   ├── dataloader.py         # DataLoader 工厂
│   │   └── process.py            # 原始数据 → 预处理 .pt 文件
│   │
│   ├── model/                    # 模型层
│   │   ├── layers.py             # ConvBlock、LinearBlock 组件
│   │   ├── cnn.py                # MNISTCNN 组装
│   │   └── factory.py            # createModel() 工厂
│   │
│   ├── train/                    # 训练层
│   │   ├── loss.py               # 损失函数工厂
│   │   ├── optimizer.py          # 优化器 + 调度器工厂
│   │   ├── engine.py             # trainEpoch / validateEpoch
│   │   ├── checkpoint.py         # 检查点持久化
│   │   └── logger.py             # 三通道日志（控制台/文件/TensorBoard）
│   │
│   ├── eval/                     # 评估层
│   │   ├── metrics.py            # 混淆矩阵、分类报告
│   │   └── visualize.py          # 训练曲线、热力图、错误样本
│   │
│   └── inference/                # 推理层
│       └── predictor.py          # Predictor 类（多格式输入）
│
├── scripts/                      # 流水线编排
│   ├── train.py                  # 训练流水线
│   ├── eval.py                   # 评估流水线
│   └── infer.py                  # 推理流水线
│
├── tests/                        # 单元测试
│   ├── test_cnn.py               # MNISTCNN 测试
│   ├── test_layers.py            # ConvBlock + LinearBlock 测试
│   ├── test_metrics.py           # 混淆矩阵 + 分类报告测试
│   └── test_predictor.py         # Predictor 测试
│
├── .github/workflows/            # CI/CD
│   ├── ruff.yml                  # Ruff lint 工作流
│   ├── format.yml                # 格式检查工作流
│   ├── ai-code-review.yml        # AI 代码审查
│   └── deploy-docs.yml           # 文档部署到 GitHub Pages
│
├── checkpoints/                  # 模型检查点（自动生成）
├── datasets/                     # MNIST 数据（自动下载）
├── outputs/                      # 训练输出（自动生成）
└── docs/                         # 本文档
```

---

## 模型架构图

![CNN 架构数据流图](/visualizations/architecture_diagram.png)

---

## 模块职责表

| 模块 | 职责 | 关键入口 |
|------|------|---------|
| `config/` | 超参数默认值、路径常量、CLI 解析 | `settings.py:125-198` |
| `src/data/` | MNIST 数据集封装、增强、归一化、DataLoader | `dataloader.py:52-132` |
| `src/model/` | 可复用层组件、CNN 组装、模型工厂 | `cnn.py:26-103` |
| `src/train/` | 训练引擎、损失/优化器/调度器、检查点、日志 | `engine.py:21-149` |
| `src/eval/` | 混淆矩阵、per-class 指标、可视化 | `metrics.py:259-363` |
| `src/inference/` | 多格式推理预测器 | `predictor.py:42-366` |
| `scripts/` | 流水线编排，串联各模块 | `train.py:28-201` |
| `main.py` | 唯一入口，解析 CLI → 分发子命令 | `main.py:63-121` |

---

## 数据流

```
python main.py <subcommand> --args
        │
        ▼
config/settings.py  ←── parse CLI args (argparse)
        │
        ▼
main.py  ←── set random seeds → dispatch to subcommand handler
        │
        ├── train → scripts/train.py
        │       ├── src/data/dataloader.py      → DataLoaders
        │       ├── src/model/factory.py         → createModel()
        │       ├── src/train/optimizer.py        → optimizer + scheduler
        │       ├── src/train/loss.py             → criterion
        │       ├── src/train/logger.py           → TrainLogger
        │       ├── src/train/engine.py           → trainEpoch / validateEpoch
        │       └── src/train/checkpoint.py       → saveCheckpoint
        │
        ├── eval  → scripts/eval.py
        │       ├── src/data/dataloader.py        → test DataLoader
        │       ├── src/train/checkpoint.py       → loadCheckpoint
        │       ├── src/eval/metrics.py           → evaluateModel
        │       └── src/eval/visualize.py         → plots
        │
        └── infer → scripts/infer.py
                └── src/inference/predictor.py    → Predictor
```

---

## 入口点：main.py

[main.py:63-121](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/main.py#L63-L121) 是整个项目的唯一入口：

```python
def main(argv=None):
    args = getSettings(argv)                    # 解析 CLI
    random.seed(args.seed)                      # 固定 Python 随机种子
    np.random.seed(args.seed)                   # 固定 NumPy 随机种子
    torch.manual_seed(args.seed)               # 固定 PyTorch 随机种子

    # 子命令分发
    handlers = {
        "train": runTrain,
        "eval": runEval,
        "infer": runInference,
    }
    handlers[args.command](args)               # 延迟导入，执行对应流水线
```

三个子命令的 handler 函数（`main.py:37-55`）都是惰性导入，只在需要时加载对应脚本模块。

---

## 设计原则

1. **配置与代码分离** — 所有可调参数集中在 `config/default_params.py`，通过 CLI 覆盖
2. **工厂模式** — 模型、优化器、损失函数通过专门的工厂函数创建，便于替换和测试
3. **惰性导入** — 子命令处理函数延迟导入脚本模块，降低冷启动时间
4. **三通道日志** — 控制台（stderr）+ 文件（DEBUG） + TensorBoard，互不干扰
5. **确定性可复现** — 固定所有随机种子，数据分割使用固定种子的 permutation
