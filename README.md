# ALL-CNN

经典 CNN 架构学习项目，支持**多模型 × 多数据集自由组合**训练、评估、推理与基准测试。

> 原名 MNIST-CNN，已合并 [LeNet](https://github.com/NayukiChiba/LeNet) 项目。

## 已支持的架构（8 种）

| 架构 | 年份 | 输入尺寸 | 通道 | 核心特点 |
|------|------|----------|------|----------|
| LeNet-5 | 1998 | 32×32 | 1 | 5×5 Conv、AvgPool、Tanh、Xavier 初始化 |
| AlexNet | 2012 | 224×224 | 3 | 5 层 Conv、ReLU、Dropout、Kaiming 初始化 |
| VGG-11 | 2015 | 224×224 | 3 | 11 层权重，3×3 Conv 堆叠 |
| VGG-13 | 2015 | 224×224 | 3 | 13 层权重 |
| VGG-16 | 2015 | 224×224 | 3 | 16 层权重（最常用） |
| VGG-19 | 2015 | 224×224 | 3 | 19 层权重 |
| NiN | 2014 | 32×32 | 3 | mlpconv 块、全局平均池化、无全连接层 |
| GoogLeNet | 2015 | 224×224 | 3 | 22 层、Inception v1 模块、全局平均池化 |

## 已支持的数据集（12 个）

| 数据集 | 类型 | 类别数 | 图像尺寸 | 通道 |
|--------|------|--------|----------|------|
| MNIST | 手写数字 | 10 | 28×28 | 1 |
| Fashion-MNIST | 服饰类别 | 10 | 28×28 | 1 |
| EMNIST | 字母+数字 | 47 | 28×28 | 1 |
| KMNIST | 日文假名 | 10 | 28×28 | 1 |
| CIFAR-10 | 自然图像 | 10 | 32×32 | 3 |
| CIFAR-100 | 自然图像 | 100 | 32×32 | 3 |
| SVHN | 街景门牌号 | 10 | 32×32 | 3 |
| STL-10 | 自然图像 | 10 | 96×96 | 3 |
| Caltech-101 | 物体类别 | 101 | 可变 | 3 |
| Caltech-256 | 物体类别 | 257 | 可变 | 3 |
| GTSRB | 交通标志 | 43 | 可变 | 3 |
| Flowers-102 | 花卉种类 | 102 | 可变 | 3 |

所有数据集通过 torchvision 自动下载至 `datasets/` 目录。

## 项目结构

```
ALL-CNN/
├── main.py                       # 唯一入口，子命令分发
├── pyproject.toml                # 项目元数据、依赖、Ruff/UV 配置
│
├── config/                       # 配置层
│   ├── defaults.py               # SEED、设备自动选择
│   ├── paths.py                  # 路径常量 + 运行时路径构建
│   ├── data.py                   # DataParams（batch_size、workers、augmentation）
│   └── training.py               # TrainingParams（epochs、lr、optimizer、scheduler）
│
├── cnnlib/                       # 核心库
│   ├── cli/                      # CLI：参数解析（argparse）+ 交互式菜单
│   ├── data/                     # 数据：DataLoader 工厂 + 自动变换管线
│   ├── models/                   # 模型：base、blocks、各架构实现、factory
│   ├── registry/                 # 注册表：模型元数据 + 数据集元数据
│   ├── training/                 # 训练：engine、trainer、loss、optimizer、
│   │                             #       scheduler、checkpoint、logger、early_stopping
│   ├── evaluation/               # 评估：evaluator、metrics、visualization
│   ├── inference/                # 推理：Predictor 类
│   └── experiments/              # 基准测试框架
│
├── scripts/                      # 流水线脚本（train、eval、infer、benchmark）
├── tests/                        # 单元测试
├── docs/                         # VitePress 文档站点
├── datasets/                     # 自动下载的数据集
├── outputs/                      # 输出产物（按 {model}/{dataset}/ 分层）
│   ├── {model}/{dataset}/checkpoints/   # 检查点
│   ├── {model}/{dataset}/logs/          # TensorBoard 日志
│   └── {model}/{dataset}/visualizations/ # 可视化图表
│
└── .github/workflows/            # CI/CD
```

## 快速开始

```bash
# 安装依赖（推荐使用 uv）
uv sync

# 或使用 pip
pip install torch torchvision numpy matplotlib tqdm tensorboard scipy

# 交互式模式（推荐入门方式）
python main.py

# 命令行训练（任意模型 × 任意数据集自由组合）
python main.py --model vgg16 --dataset cifar10 train --epochs 50 --lr 0.01

# 评估
python main.py --model vgg16 --dataset cifar10 eval \
  --checkpoint outputs/vgg16/cifar10/checkpoints/best_model.pth

# 推理
python main.py --model vgg16 --dataset cifar10 infer \
  --checkpoint outputs/vgg16/cifar10/checkpoints/best_model.pth \
  --image cat.jpg --top-k 5

# 基准测试（所有模型 × 所有数据集）
python main.py --model all --dataset all benchmark --epochs 5
```

## CLI 子命令

### `train` — 训练模型

```
python main.py [--model MODEL] [--dataset DATASET] train [选项]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 20 | 训练轮数 |
| `--optimizer` | adam | 优化器（adam/adamw/sgd/rmsprop） |
| `--lr` | 0.001 | 学习率 |
| `--weight-decay` | 1e-4 | 权重衰减 |
| `--lr-factor` | 0.5 | LR 衰减因子 |
| `--lr-patience` | 5 | LR 调度耐心值 |
| `--lr-min` | 1e-6 | 最小学习率 |
| `--grad-clip` | — | 梯度裁剪阈值 |
| `--batch-size` | 64 | 批次大小 |
| `--val-split` | 0.1 | 验证集比例 |
| `--no-augment` | — | 禁用数据增强 |
| `--resume` | — | 从检查点恢复训练 |

### `eval` — 评估模型

```
python main.py --model MODEL --dataset DATASET eval --checkpoint PATH [选项]
```

自动生成：混淆矩阵热力图、各类别准确率柱状图、训练曲线、预测样本图。

### `infer` — 单张/批量推理

```
python main.py --model MODEL --dataset DATASET infer --checkpoint PATH [选项]
```

| 选项 | 说明 |
|------|------|
| `--image` | 单张图片路径 |
| `--image-dir` | 批量推理目录 |
| `--top-k` | 显示 Top-K 预测结果（默认 3） |

### `benchmark` — 基准测试

```
python main.py --model all --dataset all benchmark --epochs 5
```

支持 `--model all` 和 `--dataset all`，自动生成：参数量对比图、准确率对比图、推理时间对比图、综合热力图。

## 核心特性

- **注册表机制** — `@register_model()` 装饰器注册模型，数据集元数据声明式注册，新增模型/数据集只需添加一个文件/条目
- **自动变换管线** — 根据模型所需输入尺寸 + 数据集统计量，自动构建预处理管线（通道转换、缩放、归一化、增强）
- **学习率调度** — 支持 ReduceLROnPlateau / StepLR / CosineAnnealingLR / CosineAnnealingWarmRestarts 四种策略
- **早停机制** — 监控验证集准确率，patience=10 自动停止
- **检查点管理** — 按 `{model}/{dataset}/` 分层存储，自动保存最佳模型和每轮检查点
- **三通道日志** — 控制台进度条 + 文件日志 + TensorBoard 可视化
- **训练后自动可视化** — 训练结束自动加载最佳模型评估测试集，生成全套图表

## 训练结果

### LeNet-5 + MNIST

| 指标 | 值 |
|------|-----|
| 验证集准确率 | **99.57%** |
| 参数量 | 422,090 |
| 训练设备 | NVIDIA RTX 4060 Laptop (CUDA) |
| 训练耗时 | ~17 分钟（50 epochs） |

更多基准测试结果参见 `outputs/benchmarks/`。

## 文档

```bash
cd docs && npm install && npm run docs:dev
```

## 开发

```bash
# 代码格式化与检查
ruff format . && ruff check .

# 运行测试
pytest

# 安装 pre-commit 钩子
pre-commit install
```

## 路线图

- [ ] Web API（FastAPI）+ 前端（Vue 3）
- [ ] 更多架构：ResNet / DenseNet / MobileNet / EfficientNet
- [ ] 更多数据集：ImageNet-1K 子集 / Oxford Pets / Stanford Cars
- [ ] 混合精度训练（AMP）
- [ ] 分布式训练（DDP）
- [ ] 模型导出（ONNX / TorchScript）
