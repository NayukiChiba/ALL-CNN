# 模型工厂

## 工厂模式总览

模型工厂提供统一的模型创建入口，将注册表查询、参数填充、设备迁移一步完成。

**源码**: [cnnlib/models/factory.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/factory.py)

### create_model()

```python
def create_model(name: str, num_classes: int, device: str = "cpu", **kwargs):
    """根据模型名称创建模型实例"""
    info = get_model_info(name)
    model_class = info["class"]
    model = model_class(
        input_size=info["input_size"],
        in_channels=info["channels"],
        num_classes=num_classes,
        **kwargs
    )
    model.to(device)
    return model
```

### create_model_for_dataset()

```python
def create_model_for_dataset(model_name: str, dataset_name: str, device: str = "cpu", **kwargs):
    """根据模型名称和数据集名称创建模型实例（自动填充 num_classes）"""
    dataset_info = get_dataset_info(dataset_name)
    return create_model(model_name, dataset_info["num_classes"], device, **kwargs)
```

### 工厂创建流程

```
create_model_for_dataset("vgg16", "cifar100", "cuda")
        │
        ├─→ get_model_info("vgg16")
        │       → {"class": VGG16, "input_size": 224, "channels": 3, "description": "..."}
        │
        ├─→ get_dataset_info("cifar100")
        │       → {"num_classes": 100, ...}
        │
        ├─→ VGG16(input_size=224, in_channels=3, num_classes=100)
        │
        └─→ model.to("cuda")
```

```mermaid
graph LR
    Name["model_name + dataset_name"] --> LookupModel["get_model_info"]
    LookupModel --> LookupDataset["get_dataset_info"]
    LookupDataset --> Construct["model_class(input_size, in_channels, num_classes)"]
    Construct --> ToDevice["model.to(device)"]
```

---

## BaseModel：模型基类

**源码**: [cnnlib/models/base.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/base.py)

所有模型继承自 `BaseModel(nn.Module)`，获得统一的基础能力：

```python
class BaseModel(nn.Module):
    def __init__(self, input_size: int, in_channels: int, num_classes: int):
        super().__init__()
        self.input_size = input_size      # 模型期望输入尺寸
        self.in_channels = in_channels    # 模型期望输入通道
        self.num_classes = num_classes    # 分类类别数
```

### infer_feature_dim() — 自动计算展平维度

通过一次虚拟前向传播自动推断展平后的特征维度，避免手动计算：

```python
def infer_feature_dim(self, module: nn.Module) -> int:
    """对模块做一次虚拟前传，返回展平后的维度"""
    dummy = torch.randn(1, self.in_channels, self.input_size, self.input_size)
    with torch.no_grad():
        out = module(dummy)
    return out.numel() // out.shape[0]
```

这在 AlexNet 等模型中用于自动计算 `Conv → Flatten → FC` 之间的维度，无需硬编码。

```mermaid
graph LR
    Dummy["dummy_input(1, C, H, W)"] --> Forward["module forward"]
    Forward --> Flatten["flatten"]
    Flatten --> Dim["return dim"]
```

### param_count() — 参数量统计

```python
def param_count(self) -> int:
    return sum(p.numel() for p in self.parameters())
```

### summary() — 模型摘要

```python
def summary(self) -> str:
    """返回模型结构的字符串表示"""
```

---

## 公共构建块

**源码**: [cnnlib/models/blocks.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/blocks.py)

所有模型共享 5 种可复用构建块，每种封装一个标准的操作序列。

### 1. conv_block — 卷积块

```
Conv2d(k, pad=k/2) → BatchNorm2d → ReLU(inplace) → [MaxPool2d(2×2, stride=2)]
```

**用途**: 通用卷积块，AlexNet 等模型使用。`pool` 参数控制是否追加池化。

**关键设计**:
- `padding = kernel_size // 2` 实现 "same" 卷积，空间尺寸不变
- BN 在 ReLU 之前：标准化后的对称分布有利于 ReLU 产生稀疏激活
- MaxPool 可选：某些架构需要手动控制池化位置

### 2. linear_block — 全连接块

```
Linear → BatchNorm1d → ReLU(inplace) → [Dropout(p)]
```

**用途**: 全连接隐层。Dropout 默认可选，仅在 `dropout > 0` 时启用。

**与 conv_block 的区别**: 使用 `BatchNorm1d`（1D 归一化，沿特征维度）而非 `BatchNorm2d`。

### 3. vgg_conv — VGG 卷积块

```
Conv2d(3×3, pad=1) → BatchNorm2d → ReLU(inplace)
```

**用途**: VGG 全系列专用。核心哲学是只用 3×3 卷积 + same padding，空间缩减完全交给 MaxPool。

**为什么只用 3×3**: 两个 3×3 卷积的感受野等价于一个 5×5 卷积，但参数量更少（$2 \times 9 < 25$），且中间多一层非线性变换。三个 3×3 等价于一个 7×7。

### 4. nin_block — Network in Network 块 (mlpconv)

```
Conv2d(k×k) → ReLU → Conv2d(1×1) → ReLU → Conv2d(1×1) → ReLU
```

**用途**: NiN 专用。用微 MLP（多层 1×1 卷积）替代单层卷积，增强局部感受野内的非线性表达能力。

**1×1 卷积的本质**: 对每个像素位置独立做一次全连接变换——不改变空间尺寸，只混合通道信息。参数量极少（$C_{in} \times C_{out}$）。

### 5. inception_block — Inception 模块

```
输入 ──→ 4 条并行分支 ──→ 通道拼接
         ├─ 1×1 Conv
         ├─ 1×1 Conv → 3×3 Conv
         ├─ 1×1 Conv → 5×5 Conv
         └─ 3×3 MaxPool → 1×1 Conv
```

**用途**: GoogLeNet 专用。同时用不同尺寸的卷积核（1×1、3×3、5×5）和池化提取多尺度特征，最后在通道维拼接。

**1×1 瓶颈**: 3×3 和 5×5 分支先用 1×1 卷积降维，大幅减少参数量。例如 192→96 的 1×1 降维后接 3×3 卷积，比直接 192→128 的 3×3 卷积减少约 60% 参数量。

**参数说明** (`inception_block` 构造函数):
```python
inception_block(in_channels=192,
                c1=64,                    # 分支1输出通道
                c2_reduce=96, c2=128,     # 分支2瓶颈+输出通道
                c3_reduce=16, c3=32,      # 分支3瓶颈+输出通道
                c4=32)                    # 分支4输出通道
# 输出通道 = 64 + 128 + 32 + 32 = 256
```

---

## 构建块使用矩阵

| 构建块 | LeNet | AlexNet | VGG | NiN | GoogLeNet |
|--------|:-----:|:-------:|:---:|:---:|:---------:|
| `conv_block` | — | ✓ | — | — | — |
| `linear_block` | — | ✓ | — | — | — |
| `vgg_conv` | — | — | ✓ | — | — |
| `nin_block` | — | — | — | ✓ | — |
| `inception_block` | — | — | — | — | ✓ |

> LeNet 不使用任何公共构建块——其设计早于这些模式（1998 年），直接使用原生 `nn.Conv2d` + `nn.AvgPool2d` + Tanh。

---

## 模型专属参数

每个模型在工厂创建时可能接收额外的 `**kwargs`：

| 模型 | 额外参数 | 默认值 | 说明 |
|------|---------|--------|------|
| LeNet | — | — | 无额外参数 |
| AlexNet | `dropout` | 0.5 | FC 层 Dropout 概率 |
| VGG11-19 | `dropout` | 0.5 | FC 层 Dropout 概率 |
| NiN | — | — | 无额外参数 |
| GoogLeNet | `dropout` | 0.4 | 分类器 Dropout 概率 |

---

## 设备迁移

所有模型创建后立即迁移到目标设备：

```python
model.to(device)  # device = "cuda" 或 "cpu"
```

设备自动检测逻辑（[config/defaults.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/config/defaults.py)）：

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 使用示例

```python
from cnnlib.models.factory import create_model, create_model_for_dataset

# 方式1: 指定 num_classes
model = create_model("lenet", num_classes=10, device="cuda")

# 方式2: 从数据集自动获取 num_classes（推荐）
model = create_model_for_dataset("vgg16", "cifar100", device="cuda")
# → VGG16(input_size=224, in_channels=3, num_classes=100)

# 查看模型信息
print(model.summary())
print(f"参数量: {model.param_count():,}")

# 传递额外参数
model = create_model("alexnet", num_classes=10, device="cuda", dropout=0.3)
```

---

## 源码位置

- 工厂函数: [cnnlib/models/factory.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/factory.py)
- 模型基类: [cnnlib/models/base.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/base.py)
- 公共构建块: [cnnlib/models/blocks.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/blocks.py)
- 模型注册表: [cnnlib/registry/models.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/registry/models.py)
