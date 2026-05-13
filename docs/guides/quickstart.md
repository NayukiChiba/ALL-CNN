# 快速开始

本指南帮你 5 分钟内跑通 ALL-CNN 的第一个训练、评估和推理。

---

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- torchvision
- CUDA（可选，GPU 训练推荐）

## 安装

```bash
git clone https://github.com/NayukiChiba/ALL-CNN.git
cd ALL-CNN
pip install -r requirements.txt
```

首次运行时 torchvision 会自动下载所选数据集到 `datasets/` 目录。

---

## 你的第一个训练

```bash
# LeNet + MNIST（最小模型 + 最小数据集，CPU 即可）
python main.py --model lenet --dataset mnist train --epochs 10
```

预期输出:
```
Epoch   1/10 | train loss=0.3245 acc=90.12% | val loss=0.1521 acc=94.53% | lr=1.00e-03
Epoch   2/10 | train loss=0.1089 acc=96.82% | val loss=0.0886 acc=97.18% | lr=1.00e-03
...
训练完成 | 最佳 val_acc=99.14% (epoch 8)
```

换一个更大的模型和数据集：

```bash
# VGG16 + CIFAR-10（GPU 推荐）
python main.py --model vgg16 --dataset cifar10 --device cuda train --epochs 50 --batch-size 128
```

---

## 评估

```bash
# 使用训练保存的最佳模型评估
python main.py --model lenet --dataset mnist eval --checkpoint checkpoints/best_model.pth
```

输出包含混淆矩阵、per-class Precision/Recall/F1、Macro Avg，并自动生成可视化图表保存到 `visualizations/`。

---

## 推理

```bash
# 单张图片推理
python main.py --model lenet --dataset mnist infer \
    --checkpoint checkpoints/best_model.pth \
    --image test_digit.png
```

输出:

```
==================================================
  预测结果
==================================================
  预测: 7
  置信度: 99.35%

  Top-3 预测:
  1. 7: 99.35%
  2. 9:  0.41%
  3. 1:  0.12%
```

批量推理:

```bash
python main.py --model lenet --dataset mnist infer \
    --checkpoint checkpoints/best_model.pth \
    --image-dir ./test_images/
```

---

## 交互模式

不带任何子命令直接运行 `python main.py` 进入交互式菜单：

```
╔══════════════════════════════════════════════╗
║            ALL-CNN 交互式 CLI               ║
╚══════════════════════════════════════════════╝

  请选择模型:
  1. lenet      — LeNet-5 (1998)
  2. alexnet    — AlexNet (2012)
  3. vgg11      — VGG-11 (2015)
  4. vgg13      — VGG-13 (2015)
  5. vgg16      — VGG-16 (2015)
  6. vgg19      — VGG-19 (2015)
  7. nin        — Network In Network (2014)
  8. googlenet  — GoogLeNet / Inception v1 (2015)
```

按提示逐步选择模型、数据集、操作即可。

---

## 常用命令速查

```bash
# 训练
python main.py --model <模型> --dataset <数据集> train --epochs <轮数>

# 评估
python main.py --model <模型> --dataset <数据集> eval --checkpoint <路径>

# 推理
python main.py --model <模型> --dataset <数据集> infer --checkpoint <路径> --image <图片>

# 基准测试（所有模型×所有数据集）
python main.py benchmark --epochs 5
```

---

## 下一步

- [训练指南](/guides/training-guide) — 超参数调优、GPU 训练、断点续训
- [推理指南](/guides/inference-guide) — Predictor 编程 API
- [模型对比](/guides/model-comparison) — 8 模型综合对比
- [数据集对比](/guides/dataset-comparison) — 10 数据集难度排序
