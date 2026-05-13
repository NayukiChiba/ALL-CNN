---
layout: home

hero:
  name: "ALL-CNN"
  text: "经典 CNN 架构学习框架"
  tagline: 8 种经典架构 x 10 个数据集 | 数学原理推导 | 工程架构文档
  actions:
    - theme: brand
      text: 快速开始
      link: /guides/quickstart
    - theme: alt
      text: 模型架构
      link: /models/overview
    - theme: alt
      text: 数学原理
      link: /math/convolution
    - theme: alt
      text: GitHub
      link: https://github.com/NayukiChiba/ALL-CNN

features:
  - icon: 🏗️
    title: 8 种经典架构
    details: 覆盖 1998~2015 年 CNN 发展史——LeNet-5、AlexNet、VGG-11/13/16/19、NiN、GoogLeNet。每种架构逐层详解，包含参数量、感受野、前向传播方程。所有模型通过统一注册系统管理，可自由组合训练。
    link: /models/overview
    linkText: 浏览模型

  - icon: 📊
    title: 10 个标准数据集
    details: MNIST / FashionMNIST / EMNIST / CIFAR-10 / CIFAR-100 / SVHN / STL-10 / Caltech-101 / GTSRB / Flowers-102。涵盖灰度和 RGB，28×28 到可变尺寸，10 到 102 个类别。自动适配通道、尺寸、归一化参数。
    link: /datasets/overview
    linkText: 浏览数据集

  - icon: 🔌
    title: 解耦注册系统
    details: 模型注册表 + 数据集注册表完全解耦。数据管道自动根据模型需求和数据集属性完成通道转换、尺寸缩放、归一化适配。工厂模式一键创建任意模型 × 数据集组合，无需手动调整任何参数。
    link: /architecture/registry
    linkText: 了解注册系统

  - icon: 📐
    title: 严格数学推导
    details: 从离散卷积定义出发，覆盖 Kaiming/Xavier 初始化推导、BatchNorm 前向/反向公式、CrossEntropy 梯度优美形式、Adam/AdamW/SGD/RMSprop 更新规则、4 种学习率调度器对比、感受野递推公式、FLOPs 估算。
    link: /math/convolution
    linkText: 开始阅读

  - icon: ⚙️
    title: 完整工程架构
    details: 配置层、训练引擎、4 种优化器、4 种调度器、早停机制、检查点持久化、三通道日志（控制台+文件+TensorBoard）、完整评估体系（混淆矩阵+per-class 指标+可视化）、多格式推理 Predictor、交互式 CLI。
    link: /architecture/overview
    linkText: 浏览架构

  - icon: 🧪
    title: 实验与基准测试
    details: 内置全模型 × 全数据集交叉基准测试系统。自动测量参数量、模型大小、推理时间、训练/验证/测试准确率。生成跨模型对比图表（参数量热力图、准确率柱状图、推理时间对比）。结果保存为 JSON 可复现。
    link: /architecture/benchmark
    linkText: 了解基准测试

  - icon: 🚀
    title: 生产级推理部署
    details: Predictor 支持 5 种输入格式（PIL/numpy/tensor/文件路径/批量）。自动灰度转换、尺寸缩放、归一化。Top-K 置信度输出。CPU 毫秒级推理，可直接集成到应用中。
    link: /architecture/inference
    linkText: 了解推理系统

  - icon: 💻
    title: 双模式 CLI
    details: 命令行模式支持 train / eval / infer / benchmark 四个子命令，一条命令完成全流程。交互式菜单模式提供 ANSI 彩色 TUI，无需记忆参数即可引导完成训练、评估、推理、基准测试。
    link: /architecture/cli
    linkText: 了解 CLI
---

## 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy matplotlib scipy tqdm tensorboard pillow
```

### 训练模型

```bash
# LeNet-5 + MNIST（经典入门组合）
python main.py train --model lenet --dataset mnist --epochs 20

# AlexNet + CIFAR-10（GPU 推荐）
python main.py --device cuda train --model alexnet --dataset cifar10 --epochs 50 --batch-size 128

# VGG16 + CIFAR-100
python main.py --device cuda train --model vgg16 --dataset cifar100 --epochs 100 --optimizer sgd --lr 0.01
```

### 评估模型

```bash
python main.py eval --model lenet --dataset mnist --checkpoint checkpoints/lenet/mnist/best_model.pth
```

### 推理预测

```bash
python main.py infer --model lenet --dataset mnist --checkpoint checkpoints/lenet/mnist/best_model.pth --image digit.png
```

### 交互模式

```bash
python main.py  # 进入 ANSI 彩色菜单，无需记忆任何参数
```

---

## 模型架构速览

| 架构 | 年份 | 关键创新 | 输入尺寸 | 输入通道 | 激活函数 | 分类器 |
|------|------|---------|---------|---------|---------|--------|
| LeNet-5 | 1998 | CNN 开山之作 | 32×32 | 1 | Tanh | FC(84)→FC(out) |
| AlexNet | 2012 | ReLU + Dropout + 深度突破 | 224×224 | 3 | ReLU | FC(4096)×2→FC(out) |
| VGG11/13/16/19 | 2015 | 全部 3×3 卷积 + 深度至 19 层 | 224×224 | 3 | ReLU | FC(4096)×2→FC(out) |
| NiN | 2014 | mlpconv + 全局平均池化替代 FC | 32×32 | 3 | ReLU | GAP→(直接输出) |
| GoogLeNet | 2015 | Inception 多尺度 + 1×1 瓶颈 | 224×224 | 3 | ReLU | GAP→Dropout→FC(out) |

> 详见 [模型总览与对比](/models/overview)

---

## 数据集速览

| 数据集 | 类型 | 类别数 | 图像尺寸 | 通道 | 训练/测试样本 |
|--------|------|--------|---------|------|-------------|
| MNIST | 手写数字 | 10 | 28×28 | 1 | 60,000 / 10,000 |
| FashionMNIST | 服饰 | 10 | 28×28 | 1 | 60,000 / 10,000 |
| EMNIST | 字母+数字 | 47 | 28×28 | 1 | 112,800 / 18,800 |
| CIFAR-10 | 自然图像 | 10 | 32×32 | 3 | 50,000 / 10,000 |
| CIFAR-100 | 自然图像 | 100 | 32×32 | 3 | 50,000 / 10,000 |
| SVHN | 街景门牌号 | 10 | 32×32 | 3 | 73,257 / 26,032 |
| STL-10 | 自然图像 | 10 | 96×96 | 3 | 5,000 / 8,000 |
| Caltech-101 | 物体 | 101 | 可变 | 3 | ~9,000 (需自行划分) |
| GTSRB | 交通标志 | 43 | 可变 | 3 | 39,209 / 12,630 |
| Flowers-102 | 花卉 | 102 | 可变 | 3 | 6,149 / 6,149 |

> 详见 [数据集总览与对比](/datasets/overview)

---

## 文档导航

| 板块 | 内容 |
|------|------|
| [快速开始](/guides/quickstart) | 环境安装、首个训练、评估、推理 |
| [模型族系](/models/overview) | 8 种架构总览、逐层详解、技术对比 |
| [数据集](/datasets/overview) | 10 个数据集详情、兼容性矩阵、预处理管线 |
| [数学原理](/math/convolution) | 离散卷积 → 初始化 → 归一化 → 正则化 → 优化器 → 调度器 → 感受野 → FLOPs |
| [工程架构](/architecture/overview) | 注册系统 → 数据管道 → 模型工厂 → 训练 → 评估 → 推理 → 基准测试 → CLI |
