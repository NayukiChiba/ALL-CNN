---
layout: home

hero:
  name: "MNIST-CNN"
  text: "手写数字识别 CNN"
  tagline: 基于 PyTorch 实现 · 数学原理详解 · 工程架构文档
  actions:
    - theme: brand
      text: 数学原理
      link: /math/convolution
    - theme: alt
      text: 项目架构
      link: /architecture/overview
    - theme: alt
      text: GitHub
      link: https://github.com/NayukiChiba/MNIST-CNN

features:
  - icon: 📐
    title: 数学严格的推导
    details: 从离散卷积定义出发，逐层推导公式。覆盖 Conv2d、BatchNorm、ReLU、MaxPool、Dropout 的完整数学原理，以及 CrossEntropy 损失和 Adam 优化器的更新规则。
    link: /math/convolution
    linkText: 开始阅读

  - icon: 🏗️
    title: 清晰的工程架构
    details: 配置与代码分离、工厂模式、三通道日志、检查点系统。每个模块有明确的职责边界和精确到行号的源码引用。
    link: /architecture/overview
    linkText: 浏览架构

  - icon: 🚀
    title: 生产级推理部署
    details: Predictor 支持 5 种输入格式（PIL/numpy/tensor/文件路径），批量推理，top-k 置信度输出。模型仅 422K 参数，CPU 毫秒级推理。
    link: /architecture/inference
    linkText: 了解推理系统

  - icon: 📊
    title: 完整评估体系
    details: 混淆矩阵热力图、per-class 精度/召回/F1、错误样本可视化。训练曲线与 TensorBoard 日志。确定性可复现的评估流程。
    link: /architecture/evaluation
    linkText: 查看评估系统
---

## 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy matplotlib tqdm tensorboard
```

### 训练模型

```bash
# 使用默认参数训练（20 epoch, batch=64, Adam lr=0.001）
python main.py train

# GPU 训练，自定义超参数
python main.py --device cuda train --epochs 50 --batch-size 128
```

### 评估模型

```bash
python main.py eval --checkpoint checkpoints/best_model.pth
```

### 推理预测

```bash
python main.py infer \
  --checkpoint checkpoints/best_model.pth \
  --image digit.png
```

---

## 模型架构速览

```
输入 (B, 1, 28, 28)
  → ConvBlock(1→32, 3×3, MaxPool)    → (B, 32, 14, 14)
  → ConvBlock(32→64, 3×3, MaxPool)   → (B, 64,  7,  7)
  → Flatten                           → (B, 3136)
  → LinearBlock(3136→128, Dropout0.5) → (B, 128)
  → Linear(128→10)                    → (B, 10) logits
```

**总参数量：422,090** | 卷积层 < 5% | 全连接层 > 95%

---

## 文档导航

| 板块 | 内容 |
|------|------|
| [数学原理](/math/convolution) | 离散卷积 → 各层公式 → 完整前向传播 → 损失函数 → Adam 优化器 |
| [项目架构](/architecture/overview) | 目录结构 → 数据管道 → 模型设计 → 训练流程 → 评估系统 → 推理系统 |
