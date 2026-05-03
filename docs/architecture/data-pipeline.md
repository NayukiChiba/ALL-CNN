# 数据管道

## 数据流概览

```
torchvision.datasets.MNIST (原始 .gz)
        │
        ▼
src/data/dataset.py:MNISTDataset  (封装、自动下载)
        │
        ▼
src/data/transform.py:getTrainTransform / getTestTransform
        │   ├─ RandomAffine(±10°, 10%平移)  [仅训练]
        │   ├─ ToTensor()    [0,255] → [0,1]
        │   └─ Normalize((0.1307,), (0.3081,))
        │
        ▼
src/data/dataloader.py:buildDataLoaders()
        │   ├─ 54,000 训练样本 (shuffle=True)
        │   ├─  6,000 验证样本 (shuffle=False)
        │   └─ 10,000 测试样本  (shuffle=False)
        │
        ▼
torch.utils.data.DataLoader  (batch=64, workers=4, pin_memory=True)
```

---

## MNISTDataset：数据集封装

[MNISTDataset](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/data/dataset.py#L22-L63) 是对 `torchvision.datasets.MNIST` 的轻量封装：

```python
class MNISTDataset:
    def __init__(self, root, train=True, transform=None, download=True):
        self.dataset = datasets.MNIST(
            root=root, train=train,
            transform=transform, download=download
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
```

![MNIST 数字样本](/visualizations/mnist_samples.png)

- MNIST 首次使用会自动下载到 `datasets/MNIST/raw/`（4 个 `.gz` 文件，约 11MB）
- `train=True` 加载 60,000 张训练图，`train=False` 加载 10,000 张测试图
- `transform` 参数直接透传给 torchvision，使用 Compose 管线

---

## 数据预处理与增强

### 训练管线（[transform.py:21-47](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/data/transform.py#L21-L47)）

```python
transforms.Compose([
    RandomAffine(degrees=10, translate=(0.1, 0.1)),  # 可选：数据增强
    ToTensor(),                                        # PIL → Tensor [0,1]
    Normalize((0.1307,), (0.3081,)),                  # Z-score 标准化
])
```

**RandomAffine（数据增强）**

- 随机旋转：±10° 以内
- 随机平移：水平和垂直方向各不超过图像尺寸的 10%
- **仅训练时启用**（`augment=True`）——验证和测试使用相同的预处理但**不**做增强（[dataloader.py:83-86](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/data/dataloader.py#L83-L86)）

**ToTensor**

将 PIL 图像 `(H, W)` uint8 [0, 255] 转换为 Tensor `(1, H, W)` float32 [0, 1]：

$$\text{tensor} = \frac{\text{pil\_image}}{255.0}$$

**Normalize（Z-score 标准化）**

$$\text{normalized} = \frac{x - 0.1307}{0.3081}$$

其中 `0.1307` 和 `0.3081` 是 MNIST 训练集的全局均值和标准差（[default_params.py:33-34](https://github.com/NayukiChiba/MNIST-CNN/blob/main/config/default_params.py#L33-L34)）。

标准化后的数据均值为 0、方差接近 1，这是 BatchNorm 和权重初始化的理想输入分布。

### 验证/测试管线（[transform.py:50-61](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/data/transform.py#L50-L61)）

仅包含 ToTensor + Normalize，**不做增强**。确保验证和测试的评估是确定性的。

---

## 训练/验证集分割

MNIST 原始只分 train（60k）和 test（10k）。本项目从 60k 训练数据中分出验证集。

[buildDataLoaders()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/data/dataloader.py#L52-L132) 中的分割逻辑：

```python
numTrain = int(60000 * (1 - valSplit))  # 54000
indices = torch.randperm(60000, generator=g)  # 固定种子的随机排列
trainIndices = indices[:numTrain]      # 前 54000
valIndices = indices[numTrain:]        # 后 6000
```

- `torch.randperm` 使用固定种子（`DefaultParams.SEED = 42`），确保每次运行分割结果一致
- 训练集和验证集共享相同的原始 MNIST 数据，但使用**不同的 transform**（训练有增强，验证无增强），由此创建两个独立的 `Subset` 对象
- `valSplit = 0.1`（[default_params.py:29](https://github.com/NayukiChiba/MNIST-CNN/blob/main/config/default_params.py#L29)）

---

## 三个 DataLoader

| DataLoader | 样本数 | Shuffle | 增强 | 用途 |
|-----------|--------|---------|------|------|
| trainLoader | 54,000 | True | RandomAffine+ToTensor+Normalize | 训练 |
| valLoader | 6,000 | False | ToTensor+Normalize | 验证（每 epoch 评估） |
| testLoader | 10,000 | False | ToTensor+Normalize | 最终测试评估 |

共用参数（[default_params.py:20-26](https://github.com/NayukiChiba/MNIST-CNN/blob/main/config/default_params.py#L20-L26)）：

```python
BATCH_SIZE = 64      # 每批 64 张图
NUM_WORKERS = 4      # 4 个子进程并行加载数据
PIN_MEMORY = True    # 将数据锁页到 CPU 内存，加速 CPU→GPU 传输
```

`pin_memory=True` 对 GPU 训练至关重要——数据传输可以在后台异步完成，与 GPU 计算重叠。

---

## 离线预处理（可选）

[src/data/process.py](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/data/process.py) 提供将原始 MNIST 预先处理为 `.pt` 文件的功能，避免每次训练都重新应用 transform。生成 `datasets/MNIST/processed/train.pt` 和 `test.pt`。

---

## 为什么验证集不用增强？

数据增强的目的是让模型看到更多"变体"，提升泛化能力。但在验证时：

- 验证集应该反映真实的、未加干扰的数据分布
- 如果验证集也被增强，评估结果会随着每次 epoch 的随机增强而变化，指标不稳定
- 保持验证集确定性，才能准确判断模型是否真正在进步（[engine.py:159](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/train/engine.py#L159) `model.eval()` + `torch.no_grad()`）
