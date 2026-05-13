# 训练流程

## 训练流水线总览

ALL-CNN 的训练由 `Trainer` 编排类全权管理，替代旧版本的手写训练循环。

**源码**: [cnnlib/training/trainer.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/trainer.py)

```
1. 构建 DataLoaders       — build_dataloaders(model, dataset)
2. 创建模型               — create_model_for_dataset(model, dataset)
3. 创建损失/优化器/调度器  — createLoss / createOptimizer / createScheduler
4. (可选) 从检查点恢复     — loadCheckpoint(resumeFrom)
5. 初始化 Trainer + 三通道日志
6. Trainer.train():
     for epoch in 1..epochs:
         trainOneEpoch()       → 训练
         validate()            → 验证
         scheduler.step()      → 调整学习率
         logEpoch()            → 记录指标
         saveCheckpoint()      → 持久化 (best + last)
         earlyStopping.step()  → 早停判断
7. 加载最佳模型 + 测试集评估
8. 返回训练结果
```

---

## Trainer 编排类

**源码**: [cnnlib/training/trainer.py:39-294](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/trainer.py#L39-L294)

```python
class Trainer:
    def __init__(
        self,
        model,           # nn.Module
        trainLoader,     # DataLoader
        valLoader,       # DataLoader
        optimizer,       # Optimizer
        scheduler,       # LR scheduler
        lossFn,          # Loss function
        device,          # torch.device
        epochs,          # int
        checkpointDir,   # Path
        testLoader=None,         # 可选, 训练后评估
        logger=None,             # 可选, 日志器
        earlyStopping=None,      # 可选, 早停控制器
        gradClip=0.0,            # 梯度裁剪阈值
        resumeFrom=None,         # 恢复训练路径
    ):
        ...

    def train(self) -> Dict:
        """
        Returns:
            {"best_metric": ..., "best_epoch": ..., "history": {...}, "test_metrics": {...}}
        """
```

### 训练循环细节

```python
for epoch in range(startEpoch, self.epochs + 1):
    # 1. 训练
    trainMetrics = trainOneEpoch(model, trainLoader, lossFn,
                                  optimizer, device, epoch,
                                  logger, gradClip)

    # 2. 验证
    valMetrics = validate(model, valLoader, lossFn, device, desc="Val")

    # 3. 调度器 step（Plateau 需 val_loss）
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(valMetrics["loss"])
    else:
        scheduler.step()

    # 4. 记录
    history["train_loss"].append(trainMetrics["loss"])
    history["train_acc"].append(trainMetrics["accuracy"])
    history["val_loss"].append(valMetrics["loss"])
    history["val_acc"].append(valMetrics["accuracy"])

    # 5. 保存最佳模型
    if valMetrics["accuracy"] > bestMetric:
        bestMetric = valMetrics["accuracy"]
        saveCheckpoint(bestModelPath, model, optimizer, epoch,
                       bestMetric, valMetrics, scheduler)

    # 6. 保存最新模型
    saveCheckpoint(lastModelPath, model, optimizer, epoch,
                   bestMetric, valMetrics, scheduler)

    # 7. 早停
    if earlyStopping and earlyStopping.step(valMetrics["accuracy"]):
        break
```

---

## 单 Epoch 训练：trainOneEpoch

**源码**: [cnnlib/training/engine.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/engine.py)

```python
def trainOneEpoch(model, dataloader, criterion, optimizer,
                   device, epoch, logger=None, gradClip=0.0):
    model.train()
    runningLoss = 0.0
    correct = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # 清空上批梯度
        logits = model(images)           # 前向传播
        loss = criterion(logits, labels) # 计算损失
        loss.backward()                  # 反向传播
        if gradClip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradClip)
        optimizer.step()                 # 参数更新

        runningLoss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()

    return {"loss": runningLoss / total, "accuracy": 100. * correct / total}
```

**关键步骤**:

| 步骤 | 代码 | 作用 |
|------|------|------|
| 梯度清零 | `optimizer.zero_grad()` | PyTorch 默认累积梯度 |
| 前向传播 | `model(images)` | 返回 (B, num_classes) logits |
| 损失计算 | `criterion(logits, labels)` | CrossEntropy 内部含 LogSoftmax |
| 反向传播 | `loss.backward()` | 自动求导 |
| 梯度裁剪 | `clip_grad_norm_()` | 仅在 gradClip > 0 时执行 |
| 参数更新 | `optimizer.step()` | 按优化器规则更新权重 |

---

## 验证：validate

```python
@torch.no_grad()
def validate(model, dataloader, criterion, device, desc="Val"):
    model.eval()
    # ... 前向 + 统计，无 backward / optimizer.step()
```

与训练的关键区别：

| | 训练 (trainOneEpoch) | 验证 (validate) |
|---|---|---|
| 模型模式 | `model.train()` | `model.eval()` |
| Dropout | 启用 | 关闭 |
| BatchNorm | 使用 batch 统计 | 使用运行均值/方差 |
| 梯度 | 需要 grad | `@torch.no_grad()` |
| 梯度裁剪 | ✓ (可选) | ✗ |
| 参数更新 | ✓ | ✗ |

---

## 损失函数

**源码**: [cnnlib/training/loss.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/loss.py)

```python
def createLoss(name="cross_entropy", **kwargs):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    raise ValueError(f"未知损失函数: '{name}'")
```

CrossEntropyLoss 内部计算 `LogSoftmax + NLLLoss`，接受原始 logits 而非概率。支持 `label_smoothing` 参数（通过 `**kwargs` 透传）。

---

## 优化器工厂

**源码**: [cnnlib/training/optimizer.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/optimizer.py)

```python
def createOptimizer(model, name="adam", lr=0.001, weight_decay=0.0, **kwargs):
    # name: adam / adamw / sgd / rmsprop
    # 查表返回对应优化器实例
```

---

## 调度器工厂

**源码**: [cnnlib/training/scheduler.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/scheduler.py)

```python
def createScheduler(optimizer, name="plateau", **kwargs):
    # name: plateau / step / cosine / cosine_warm
```

---

## 检查点持久化

**源码**: [cnnlib/training/checkpoint.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/checkpoint.py)

### 保存

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),  # 可选
    "epoch": epoch,
    "best_metric": bestMetric,
    "metrics": valMetrics,  # {loss, accuracy}
}
torch.save(checkpoint, filepath)
```

### 文件策略

| 文件 | 保存时机 | 用途 |
|------|---------|------|
| `best_model.pth` | val_acc 创新高时 | 部署/评估用的最佳模型 |
| `last_model.pth` | 每 epoch | 中断后恢复训练 |

### 加载

```python
def loadCheckpoint(filepath, model, optimizer=None,
                    scheduler=None, device="cpu"):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
```

---

## 早停

**源码**: [cnnlib/training/earlyStopping.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/earlyStopping.py)

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode="max"):
        ...

    def step(self, metric) -> bool:
        """返回 True = 应停止训练"""
        ...
```

最佳模型在 `best_model.pth` 中已保存，早停不影响已保存的最佳权重。

---

## 三通道日志

**源码**: [cnnlib/training/logger.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/training/logger.py)

| 通道 | 目标 | 内容 |
|------|------|------|
| 控制台 | `sys.stderr` | 每 epoch 的 loss/acc/lr 摘要 |
| 文件 | `logs/train_YYYYMMDD_HHMMSS.log` | 完整训练记录 |
| TensorBoard | `outputs/tensorboard/YYYYMMDD_HHMMSS/` | 标量曲线 |

TensorBoard 标签：`Loss/train`、`Loss/val`、`Accuracy/train`、`Accuracy/val`、`LR`。

---

## 训练命令示例

```bash
# 默认参数
python main.py train

# 完整参数
python main.py --model vgg16 --dataset cifar10 --device cuda train \
    --epochs 50 --batch-size 128 \
    --optimizer adamw --lr 0.001 --weight-decay 1e-4 \
    --grad-clip 1.0

# 恢复训练
python main.py train --resume checkpoints/last_model.pth

# 换调度器参数
python main.py train --lr-factor 0.5 --lr-patience 5 --lr-min 1e-6
```

---

## 相关文档

- [优化器](/math/optimizer) — Adam / AdamW / SGD / RMSprop 详解
- [调度器](/math/schedulers) — ReduceLROnPlateau / StepLR / Cosine 等
- [梯度裁剪](/math/gradient-clipping) — 防止梯度爆炸
- [L1/L2/Weight Decay](/math/regularization) — 正则化技术
- [训练指南](/guides/training-guide) — 使用指南与超参数调优
- [模型工厂](/architecture/model-factory) — create_model_for_dataset
