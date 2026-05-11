# ALL-CNN

经典 CNN 架构学习项目，支持多模型、多数据集自由组合训练。

> 原名 MNIST-CNN，已合并 [LeNet](https://github.com/NayukiChiba/LeNet) 项目。

## 已支持的架构

| 架构 | 年份 | 特点 | 状态 |
|------|------|------|------|
| CNN（ConvBlock） | — | Conv → BN → ReLU → MaxPool 堆叠 | 已有 |
| LeNet-5 | 1998 | 5×5 Conv + AvgPool + Tanh + Xavier | 已合并 |

**规划中**: AlexNet / VGG-11,13,16,19 / NiN / GoogLeNet

## 已支持的数据集

| 数据集 | 类型 | 类别 | 尺寸 |
|--------|------|------|------|
| MNIST | 灰度数字 | 10 | 28×28 |
| Fashion-MNIST | 灰度服饰 | 10 | 28×28 |

## 项目结构

```
ALL-CNN/
├── main.py                       # 唯一入口，子命令分发
├── pyproject.toml                # 项目元数据、依赖、Ruff 配置
│
├── config/                       # 配置层
│   ├── default_params.py         # 超参数默认值
│   ├── paths.py                  # 项目路径常量
│   └── settings.py               # CLI 参数解析（argparse）
│
├── src/                          # 核心源代码
│   ├── data/                     # 数据层
│   │   ├── dataset.py            # 数据集封装
│   │   ├── transform.py          # 预处理/增强管线
│   │   ├── dataloader.py         # DataLoader 工厂
│   │   └── process.py            # 原始数据预处理
│   │
│   ├── model/                    # 模型层
│   │   ├── layers.py             # 公共构建块（ConvBlock, LinearBlock）
│   │   ├── cnn.py                # ConvBlock 堆叠式 CNN
│   │   ├── lenet.py              # LeNet-5 经典实现
│   │   └── factory.py            # 多模型工厂 createModel()
│   │
│   ├── train/                    # 训练层
│   │   ├── loss.py               # 损失函数工厂
│   │   ├── optimizer.py          # 优化器 + 调度器工厂
│   │   ├── engine.py             # trainEpoch / validateEpoch
│   │   ├── checkpoint.py         # 检查点持久化
│   │   └── logger.py             # 三通道日志
│   │
│   ├── eval/                     # 评估层
│   │   ├── metrics.py            # 混淆矩阵、分类报告
│   │   └── visualize.py          # 训练曲线、热力图、错误样本
│   │
│   └── inference/                # 推理层
│       └── predictor.py          # Predictor 类（多格式输入）
│
├── scripts/                      # 流水线脚本
│   ├── train.py                  # 训练流水线
│   ├── eval.py                   # 评估流水线
│   ├── infer.py                  # 推理流水线
│   └── visualizeModel.py         # 模型可视化
│
├── tests/                        # 单元测试
│   ├── test_cnn.py
│   ├── test_layers.py
│   ├── test_metrics.py
│   └── test_predictor.py
│
├── checkpoints/                  # 检查点（按模型/数据集分层）
│   ├── cnn/mnist/
│   └── lenet/fashionmnist/
│
├── datasets/                     # 数据（自动下载）
│   ├── MNIST/
│   └── FashionMNIST/
│
├── outputs/                      # 产物
│   ├── logs/{model}/{dataset}/       # TensorBoard 日志
│   └── visuals/{model}/{dataset}/    # 可视化图表
│
├── docs/                         # VitePress 文档站点
└── .github/workflows/            # CI/CD
```

## 模型架构

### CNN（ConvBlock 堆叠）

```
输入 (1, 28, 28)
  → ConvBlock(1→32,  3×3, MaxPool)  → (32, 14, 14)
  → ConvBlock(32→64, 3×3, MaxPool)  → (64,  7,  7)
  → Flatten                          → (3136,)
  → LinearBlock(3136→128, Dropout 0.5) → (128,)
  → Linear(128→10)                   → (10,) logits
```

> 使用 ReLU + BatchNorm，现代化训练策略

### LeNet-5

```
输入 (1, 32, 32)
  → Conv2d(1→6, 5×5) + Tanh       → (6, 28, 28)
  → AvgPool2d(2×2)                  → (6, 14, 14)
  → Conv2d(6→16, 5×5) + Tanh      → (16, 10, 10)
  → AvgPool2d(2×2)                  → (16, 5, 5)
  → Conv2d(16→120, 5×5) + Tanh    → (120, 1, 1)
  → Flatten                         → (120,)
  → Linear(120→84) + Tanh          → (84,)
  → Linear(84→10)                   → (10,) logits
```

> 使用 Tanh + Average Pooling + Xavier 初始化，忠实还原 1998 年原始设计

## 训练结果

### CNN + MNIST

```
Epoch 42 (best) | val_loss=0.0164  val_acc=0.9957  lr=2.50e-04
Epoch 50 (last) | val_loss=0.0150  val_acc=0.9957  lr=1.25e-04
```

| 指标 | 值 |
|------|-----|
| 验证集准确率 | **99.57%** |
| 参数量 | 422,090 |
| 训练设备 | NVIDIA RTX 4060 Laptop (cuda) |
| 训练耗时 | ~17 分钟 (50 epochs) |

### LeNet-5 + FashionMNIST

> 来自合并的 LeNet 项目，详见 `outputs/visuals/lenet/fashionmnist/`

## 快速开始

```bash
# 安装依赖
pip install torch torchvision numpy matplotlib tqdm tensorboard

# CNN + MNIST 训练
python main.py train --epochs 50

# LeNet + FashionMNIST 评估
python main.py eval --checkpoint checkpoints/lenet/fashionmnist/best.pth

# 推理
python main.py infer --checkpoint checkpoints/lenet/fashionmnist/best.pth --image shirt.png
```

## 文档

```bash
cd docs && npm install && npm run docs:dev
```

## 开发

```bash
ruff format . && ruff check .    # 代码格式化与检查
pytest                            # 运行测试
```

## 路线图

- [ ] AlexNet（2012）
- [ ] VGG-11/13/16/19（2014）
- [ ] NiN — Network in Network（2014）
- [ ] GoogLeNet — Inception v1（2014）
- [ ] 数据集扩展：KMNIST / CIFAR-10 / CIFAR-100 / SVHN / STL-10
- [ ] Web API（FastAPI）+ 前端（Vue 3）
- [ ] 交互式 CLI 菜单
