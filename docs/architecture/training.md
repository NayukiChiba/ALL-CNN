# 训练流程

## 训练流水线总览

[scripts/train.py:28-201](https://github.com/NayukiChiba/MNIST-CNN/blob/main/scripts/train.py#L28-L201) 中的 `run()` 函数掌管完整的训练生命周期：

```
1. 构建 DataLoaders
2. 创建模型 → model.to(device)
3. 创建损失函数 / 优化器 / 调度器
4. (可选) 从检查点恢复
5. 初始化三通道日志
6. for epoch in 1..epochs:
     trainEpoch()       → 训练
     validateEpoch()    → 验证
     scheduler.step()   → 调整学习率
     logEpoch()         → 记录指标
     saveCheckpoint()   → 持久化
7. 打印总结 & 关闭日志
```

---

## 单 Epoch 训练：trainEpoch

[trainEpoch()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/engine.py#L21-L149) 实现标准的训练循环：

```python
def trainEpoch(model, dataloader, criterion, optimizer, epoch, device):
    model.train()                        # 启用 BN 训练模式 + Dropout
    runningLoss = 0.0
    correct = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # 清空上一批次的梯度
        logits = model(images)           # 前向传播
        loss = criterion(logits, labels) # 计算损失
        loss.backward()                  # 反向传播（自动求导）
        optimizer.step()                 # Adam 更新参数

        # 累积统计量
        runningLoss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/((pbar.n+1)*batch):.1f}%'
        })

    return runningLoss / total, correct / total
```

**关键步骤解析：**

| 步骤 | 代码 | 作用 |
|------|------|------|
| 梯度清零 | `optimizer.zero_grad()` | PyTorch 默认累积梯度，不清零会叠加 |
| 前向传播 | `model(images)` | 返回 (B, 10) 原始 logits |
| 损失计算 | `criterion(logits, labels)` | CrossEntropy 内部含 LogSoftmax |
| 反向传播 | `loss.backward()` | 自动求导计算所有参数的梯度 |
| 参数更新 | `optimizer.step()` | Adam 规则更新权重 |

---

## 单 Epoch 验证：validateEpoch

[validateEpoch()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/engine.py#L152-L231) 是评估模式下的前向传播：

```python
@torch.no_grad()               # 禁用梯度计算（省显存、加速）
def validateEpoch(model, dataloader, criterion, epoch, device):
    model.eval()               # BN 冻结统计量，Dropout 关闭

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)   # 前向传播（无反向）
        loss = criterion(logits, labels)
        # ...累积统计量...
```

与训练的关键区别：

| | 训练 (trainEpoch) | 验证 (validateEpoch) |
|---|---|---|
| 模型模式 | `model.train()` | `model.eval()` |
| Dropout | 启用（50% 失活） | 关闭 |
| BatchNorm | 使用当前 batch 统计 | 使用运行均值/方差 |
| 梯度 | 需要 grad → backward | `@torch.no_grad()` 禁用 |
| 显存 | 存储梯度 + 中间激活 | 仅存储前向激活 |

---

## 损失函数

[createLossFunction()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/loss.py#L15-L26) 返回 `nn.CrossEntropyLoss()`：

```python
criterion = createLossFunction()  # → nn.CrossEntropyLoss()
loss = criterion(logits, labels)  # logits: (B,10), labels: (B,)
```

CrossEntropyLoss 内部计算 `LogSoftmax + NLLLoss`，接受原始 logits 而非概率。

---

## 优化器与调度器

[createOptimizer()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/optimizer.py#L19-L48)：

```python
optimizer = createOptimizer(model, lr=0.001, weight_decay=1e-4)
# → Adam(params, lr=0.001, weight_decay=1e-4)
```

[createScheduler()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/optimizer.py#L51-L86)：

```python
scheduler = createScheduler(optimizer, factor=0.5, patience=3)
# → ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
#                     patience=3, min_lr=1e-6)
```

在训练循环中（[train.py:149](https://github.com/NayukiChiba/MNIST-CNN/blob/main/scripts/train.py#L149)）：

```python
scheduler.step(valLoss)  # 每 epoch 调用，按验证损失调整学习率
```

---

## 检查点持久化

### 保存（[checkpoint.py:33-88](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/checkpoint.py#L33-L88)）

```python
def saveCheckpoint(model, optimizer, epoch, metrics, filepath):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,  # {train_loss, train_acc, val_loss, val_acc}
    }
    torch.save(checkpoint, filepath)
```

每个 epoch 保存两个文件：

| 文件 | 何时保存 | 用途 |
|------|---------|------|
| `checkpoints/last_model.pth` | 每 epoch | 训练中断后恢复 |
| `checkpoints/best_model.pth` | val_acc 创新高时 | 部署/评估用 |

### 加载（[checkpoint.py:91-147](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/checkpoint.py#L91-L147)）

```python
def loadCheckpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["metrics"]
```

- `map_location=device` 确保在不同设备间（CPU/dGPU/cPU）安全加载
- `optimizer=None` 时只加载模型权重（评估和推理场景不需要优化器状态）

---

## 三通道日志

[TrainLogger](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/logger.py#L37-L355) 同时输出到三个通道：

| 通道 | 目标 | 级别 | 内容 |
|------|------|------|------|
| 控制台 | `sys.stderr` | INFO | 每 epoch 的 loss/acc/lr 摘要 |
| 文件 | `outputs/logs/train_YYYYMMDD_HHMMSS.log` | DEBUG | 完整详细的训练记录 |
| TensorBoard | `outputs/tensorboard/YYYYMMDD_HHMMSS/` | — | 标量曲线 + 模型图 |

**为什么用 stderr 而非 stdout？** — tqdm 进度条也使用 stderr。两者共享通道互不干扰，不会出现进度条和日志行交叉错位的问题。

记录的关键指标：

```python
logger.logEpoch(epoch, trainLoss, trainAcc, valLoss, valAcc, lr)
```

TensorBoard 标签：`Loss/train`、`Loss/val`、`Accuracy/train`、`Accuracy/val`、`LR`。

---

## 断点续训

[scripts/train.py:101-109](https://github.com/NayukiChiba/MNIST-CNN/blob/main/scripts/train.py#L101-L109)：

```python
if args.resume:
    startEpoch, bestMetrics = loadCheckpoint(
        str(LAST_MODEL_PATH), model, optimizer
    )
    bestValAcc = bestMetrics.get("val_acc", 0)
```

恢复后从 `startEpoch + 1` 继续训练，保持之前的 best_val_acc 记录。

---

## 训练命令示例

```bash
# 默认参数训练
python main.py train

# 自定义训练 50 epoch，GPU 训练
python main.py --device cuda train --epochs 50 --batch-size 128

# 从检查点恢复
python main.py train --resume

# 更深网络
python main.py train --conv-channels 32 64 128 --fc-hidden-size 256
```
