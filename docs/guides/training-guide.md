# 训练指南

## 模型-数据集兼容表

| 模型 | 默认输入 | 支持通道 | 适配策略 |
|------|:---:|:---:|------|
| LeNet-5 | 32×32 | 1 | 灰度直接用；RGB → 灰度 |
| AlexNet | 224×224 | 3 | RGB 直接用；灰度 → 复制为 3 通道 |
| VGG11/13/16/19 | 224×224 | 3 | 同上 |
| NiN | 32×32 | 3 | 同上 |
| GoogLeNet | 224×224 | 3 | 同上 |

所有适配由 `build_transform()` 自动完成，无需手动干预。

---

## CLI 训练命令

```bash
python main.py [全局参数] train [训练参数]
```

### 全局参数

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `--model` | lenet | lenet / alexnet / vgg11-19 / nin / googlenet |
| `--dataset` | mnist | mnist / fashionmnist / emnist / cifar10 / cifar100 / svhn / stl10 / caltech101 / gtsrb / flowers102 |
| `--device` | cpu | cpu / cuda |
| `--seed` | 42 | 随机种子 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `--epochs` | 30 | 训练轮数 |
| `--batch-size` | 64 | 批次大小 |
| `--optimizer` | adam | adam / adamw / sgd / rmsprop |
| `--lr` | 0.001 | 初始学习率 |
| `--weight-decay` | 1e-4 | 权重衰减 |
| `--lr-factor` | 0.5 | LR 衰减因子（plateau 用） |
| `--lr-patience` | 3 | LR 调度器耐心值 |
| `--lr-min` | 1e-6 | 最小学习率 |
| `--grad-clip` | 0 | 梯度裁剪阈值（0=不裁剪） |
| `--resume` | None | 从 checkpoint 恢复训练 |
| `--data-dir` | datasets/ | 数据集目录 |
| `--checkpoint-dir` | checkpoints/ | checkpoint 目录 |
| `--log-dir` | logs/ | 日志目录 |

---

## 超参数调优建议

### 学习率

| 数据集 | 推荐初始 lr | 说明 |
|------|:---:|------|
| MNIST / FashionMNIST / EMNIST | 0.001 | 简单灰度数据集，收敛快 |
| CIFAR-10 / CIFAR-100 | 0.001 | 标准起点 |
| SVHN / GTSRB | 0.001 | 与 CIFAR 类似 |
| STL-10 / Caltech-101 | 0.001 | 样本少，可能需更小 lr |
| Flowers-102 | 0.0005 | 复杂细粒度分类 |

### 优化器选择

```bash
# 默认：Adam（快速原型）
python main.py train

# 追求最优：AdamW（解耦 weight decay）
python main.py train --optimizer adamw --weight-decay 1e-4

# 大 batch 训练：SGD+Momentum
python main.py train --optimizer sgd --lr 0.01 --batch-size 256

# RNN 或特殊场景：RMSprop
python main.py train --optimizer rmsprop
```

### 调度器选择

当前默认使用 ReduceLROnPlateau（当前通过配置文件管理调度器设置）。通过 CLI 调整调度器参数：

```bash
python main.py train --lr-factor 0.5 --lr-patience 5 --lr-min 1e-6
```

---

## GPU 训练

```bash
# 单 GPU
python main.py --device cuda train --epochs 50 --batch-size 128

# 增大 batch size 充分利用 GPU
python main.py --device cuda train --batch-size 256 --num-workers 8
```

- `--batch-size`: GPU 显存越大可设越大（如 128/256）
- `--num-workers`: 设置为 CPU 核心数（如 4/8）

---

## 断点续训

```bash
# 从最近的 checkpoint 恢复
python main.py train --resume checkpoints/last_model.pth

# 恢复后继续训练更多 epoch
python main.py train --resume checkpoints/last_model.pth --epochs 100
```

恢复训练会：
1. 加载模型权重、优化器状态、调度器状态
2. 从上次结束的 epoch+1 继续
3. 保持之前的 best_val_acc 记录

---

## 训练输出

每个 epoch 后:

```
Epoch   5/50 | train loss=0.2103 acc=92.45% | val loss=0.1852 acc=93.10% | lr=1.00e-03
  >> 保存最佳模型 (acc=93.10%)
```

训练完成后在测试集评估:

```
测试集 | test loss=0.1801 test acc=93.25%
```

### 生成文件

| 文件 | 说明 |
|------|------|
| `checkpoints/best_model.pth` | 最佳验证准确率时的模型 |
| `checkpoints/last_model.pth` | 最后一次 epoch 的模型 |
| `logs/train_YYYYMMDD_HHMMSS.log` | 完整训练日志 |
| `outputs/tensorboard/YYYYMMDD_HHMMSS/` | TensorBoard 事件文件 |

### 查看 TensorBoard

```bash
tensorboard --logdir outputs/tensorboard/
```

---

## 常见问题

### loss 突然变成 NaN

可能原因：学习率太大 / 梯度爆炸。

解决方案：
```bash
# 降低学习率
python main.py train --lr 0.0001

# 或启用梯度裁剪
python main.py train --grad-clip 1.0
```

### 训练速度太慢

```bash
# GPU 训练 + 增大 batch size + 多 worker
python main.py --device cuda train --batch-size 256 --num-workers 8
```

### 验证准确率不上升

可能原因：过拟合 / 学习率太小。

解决方案：
- 增大 weight decay: `--weight-decay 5e-4`
- 减小 batch size（更强的随机性）: `--batch-size 32`
- 检查学习率是否过低（被 ReduceLROnPlateau 降到 1e-6）: `--lr-min 1e-5`

---

## 相关文档

- [模型对比](/guides/model-comparison) — 选择适合的模型
- [数据集对比](/guides/dataset-comparison) — 了解各数据集难度
- [优化器](/math/optimizer) — Adam/AdamW/SGD/RMSprop 详解
- [调度器](/math/schedulers) — 学习率调度策略
- [训练流程](/architecture/training) — Trainer 内部实现
