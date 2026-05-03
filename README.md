# MNIST-CNN

基于 PyTorch 的 MNIST 手写数字识别 CNN，完整工程化实现。

## 项目结构

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


## 最佳训练结果

```
Epoch 42 (best) | val_loss=0.0164  val_acc=0.9957  lr=2.50e-04
Epoch 50 (last) | val_loss=0.0150  val_acc=0.9957  lr=1.25e-04
```

| 指标 | 值 |
|------|-----|
| 验证集准确率 | **99.57%** |
| 训练集准确率 (final) | 99.15% |
| 参数量 | 422,090 |
| 训练设备 | NVIDIA RTX 4060 Laptop (cuda) |
| 训练耗时 | ~17 分钟 (50 epochs) |

## 模型架构

```
输入 (1, 28, 28)
  → ConvBlock(1→32,  3×3, MaxPool)  → (32, 14, 14)
  → ConvBlock(32→64, 3×3, MaxPool)  → (64,  7,  7)
  → Flatten                          → (3136,)
  → LinearBlock(3136→128, Dropout 0.5) → (128,)
  → Linear(128→10)                   → (10,) logits
```

## 快速开始

```bash
# 安装依赖
pip install torch torchvision numpy matplotlib tqdm tensorboard

# 训练
python main.py train --epochs 50

# 评估
python main.py eval --checkpoint checkpoints/best_model.pth

# 推理
python main.py infer --checkpoint checkpoints/best_model.pth --image digit.png
```

## 文档

完整文档使用 VitePress 构建，包含：

- **数学原理** — 离散卷积、各层公式推导、完整前向传播、损失函数、Adam 优化器
- **项目架构** — 数据管道、模型设计、训练流程、评估系统、推理系统

```bash
cd docs && npm install && npm run docs:dev
```

## 开发

```bash
ruff format . && ruff check .    # 代码格式化与检查
pytest                            # 运行测试
```
