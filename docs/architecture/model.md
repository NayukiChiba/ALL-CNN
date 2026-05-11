# 模型设计与工厂模式

## 设计概览

模型采用**可复用组件 + 动态组装**的架构。基础组件（ConvBlock、LinearBlock）封装标准操作序列，MNISTCNN 通过配置参数驱动组件组装，工厂函数提供统一的创建入口。

```
Config (conv_channels, fc_hidden_size, dropout)
        │
        ▼
createModel() 工厂 ──→ MNISTCNN
        │               ├── nn.ModuleList[ConvBlock x L]
        │               ├── nn.Flatten()
        │               ├── LinearBlock
        │               └── nn.Linear(→10)
        │
        ▼
model.to(device)  →  (B, 1, 28, 28) → (B, 10) logits
```

---

## ConvBlock：卷积组件

`ConvBlock`（[layers.py:20-145](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/model/layers.py#L20-L145)）封装了标准的卷积→归一化→激活→池化序列：

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, use_pool=True):
        padding = kernel_size // 2  # "same" padding
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
```

**设计决策：**

| 决策 | 理由 |
|------|------|
| BN 在 ReLU 之前 | 标准化分布对称，ReLU 的稀疏性更好 |
| `inplace=True` | 节省显存，ReLU 操作原地修改张量 |
| `padding = k//2` | 奇数核实现 "same" 输出，空间尺寸由 MaxPool 主动控制 |
| MaxPool 可选 (`use_pool`) | 灵活决定是否降采样 |

forward 顺序（[layers.py:133-145](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/model/layers.py#L133-L145)）：

```python
def forward(self, x):
    x = self.conv(x)     # Conv2d
    x = self.bn(x)       # BatchNorm2d
    x = self.relu(x)     # ReLU
    if self.pool is not None:
        x = self.pool(x) # MaxPool2d
    return x
```

**ConvBlock #1 输出的 32 个特征图（28x28）：**

![ConvBlock #1 特征图](/visualizations/feature_maps_conv1.png)

**ConvBlock #2 输出的 64 个特征图（14x14）：**

![ConvBlock #2 特征图](/visualizations/feature_maps_conv2.png)

---

## LinearBlock：全连接组件

`LinearBlock`（[layers.py:148-243](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/model/layers.py#L148-L243)）类似 ConvBlock 但用于全连接层：

```python
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
```

与 ConvBlock 的区别：
- 使用 `BatchNorm1d`（1D 归一化，在特征维度上操作）而非 `BatchNorm2d`
- **Dropout 仅在全连接层之后**，卷积层不设 Dropout（卷积参数少且有 MaxPool 提供正则）
- dropout 值可配置，默认 `0.0`（无 Dropout），但在 CNN 中使用 `0.5`

---

## MNISTCNN：模型组装

[cnn.py:26-103](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/model/cnn.py#L26-L103) 中 `MNISTCNN` 通过参数驱动的循环动态构建卷积层序列：

```python
class MNISTCNN(nn.Module):
    def __init__(self, conv_channels=[32, 64],
                 fc_hidden_size=128, dropout=0.5):
        # 动态构建卷积块
        self.convBlocks = nn.ModuleList([
            ConvBlock(
                in_channels=conv_channels[i-1] if i > 0 else 1,
                out_channels=ch,
                kernel_size=3,
                use_pool=True,
            )
            for i, ch in enumerate(conv_channels)
        ])

        # 自动计算扁平化尺寸
        numConvLayers = len(conv_channels)
        pooledSize = 28 // (2 ** numConvLayers)    # → 28/(2^2) = 7
        lastChannels = conv_channels[-1]            # → 64
        flattenedSize = lastChannels * (pooledSize ** 2)  # → 64x49 = 3136

        self.flatten = nn.Flatten()
        self.fcBlock = LinearBlock(flattenedSize, fc_hidden_size, dropout)
        self.classifier = nn.Linear(fc_hidden_size, 10)
```

**关键设计：自动计算扁平化尺寸**

`flattenedSize = lastChannels x (28 / 2^L)^2` 中的各参数：

- `28`：MNIST 原始图尺寸
- `2^L`：每层 MaxPool(2,2) 将空间尺寸减半，L 层后变为 1/2^L
- 两个 ConvBlock → 28/(2^2) = 7 → 64x7x7 = 3136

如果改为 `conv_channels=[32, 64, 128]`（3 层），自动计算为 `128 x (28/2^3)^2 = 128 x 3.5^2 = 128 x 12 = 1536`，无需任何手动修改。

**forward 方法**（[cnn.py:88-102](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/model/cnn.py#L88-L102)）：

```python
def forward(self, x):
    for convBlock in self.convBlocks:
        x = convBlock(x)       # 依次通过所有 ConvBlock
    x = self.flatten(x)        # (B, 64, 7, 7) → (B, 3136)
    x = self.fcBlock(x)        # Linear+BN+ReLU+Dropout → (B, 128)
    x = self.classifier(x)     # Linear(128, 10) → (B, 10)
    return x                   # 返回原始 logits，不含 softmax
```

**注意：** 输出是原始 logits，不是概率。softmax 在 CrossEntropyLoss 内部或 Predictor 的后处理中进行。这让训练代码可以高效利用 PyTorch 内置的 fused CrossEntropyLoss。

---

## 工厂模式

[createModel()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/model/factory.py#L17-L41) 是模型的唯一创建入口：

```python
def createModel(conv_channels=None, hidden_size=None,
                dropout=None, device=None):
    model = MNISTCNN(
        conv_channels=conv_channels or ModelParams.CONV_CHANNELS,
        fc_hidden_size=hidden_size or ModelParams.FC_HIDDEN_SIZE,
        dropout=dropout or ModelParams.DROPOUT_RATE,
    )
    model.to(device or DefaultParams.DEVICE)  # GPU/CPU
    return model
```

**工厂模式的好处：**

1. **统一默认值来源** — 所有模块从 `ModelParams` 读取默认值，而不是分散在各处
2. **设备迁移一步完成** — 构建模型后立即 `model.to(device)`
3. **测试友好** — `conv_channels=[16]` 构建小型模型用于单元测试（[test_cnn.py:41-43](https://github.com/NayukiChiba/MNIST-CNN/blob/main/tests/test_cnn.py#L41-L43)）
4. **CLI 可切换架构** — 运行 `python main.py train --conv-channels 32 64 128` 即创建三层卷积模型

---

## 默认架构参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `CONV_CHANNELS` | `[32, 64]` | 两层卷积，输出通道分别为 32、64 |
| `CONV_KERNEL_SIZE` | `3` | 3x3 卷积核 |
| `FC_HIDDEN_SIZE` | `128` | 全连接隐层 128 个神经元 |
| `DROPOUT_RATE` | `0.5` | Dropout 概率 50% |

全部定义在 [default_params.py:44-56](https://github.com/NayukiChiba/MNIST-CNN/blob/main/config/default_params.py#L44-L56)。
