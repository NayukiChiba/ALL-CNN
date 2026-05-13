# 推理指南

## 推理模式

ALL-CNN 支持两种推理模式：

1. **CLI 命令行**: `python main.py infer` — 适合单次快速推理
2. **Predictor API**: 编程方式调用 — 适合集成到应用中

---

## CLI 命令行推理

### 单图推理

```bash
python main.py --model lenet --dataset mnist infer \
    --checkpoint checkpoints/best_model.pth \
    --image my_digit.png
```

支持格式: PNG / JPEG / BMP / TIFF 等 PIL 支持的格式。

### Top-K 预测

```bash
python main.py --model vgg16 --dataset cifar10 infer \
    --checkpoint checkpoints/best_model.pth \
    --image cat.jpg \
    --top-k 5
```

输出 Top-5 预测及置信度。

### 批量推理

```bash
python main.py --model lenet --dataset mnist infer \
    --checkpoint checkpoints/best_model.pth \
    --image-dir ./test_images/
```

遍历目录下所有图片文件，逐张推理并汇总。

---

## Predictor 编程 API

**源码**: [cnnlib/inference/predictor.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/inference/predictor.py)

### 基本用法

```python
from cnnlib.data.transform import build_transform
from cnnlib.inference.predictor import Predictor
from cnnlib.models.factory import create_model_for_dataset
import torch

# 1. 创建模型
model = create_model_for_dataset("lenet", "mnist", num_classes=10)

# 2. 加载权重
checkpoint = torch.load("checkpoints/best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# 3. 构建 transform（推理模式：无增强）
transform = build_transform("lenet", "mnist", augment=False)

# 4. 创建 Predictor
predictor = Predictor(model, transform, device="cpu")

# 5. 推理
from PIL import Image
img = Image.open("my_digit.png")
results = predictor.predict(img, topK=3)

for r in results:
    print(f"{r['class_name']}: {r['confidence']:.2%}")
```

### 支持的输入格式

Predictor 接受五种输入格式：

| 格式 | 示例 | 处理方式 |
|------|------|---------|
| PIL Image | `Image.open("cat.jpg")` | 直接走 transform |
| 文件路径 (str/Path) | `"cat.jpg"` | PIL.open → transform |
| numpy (H, W) uint8 | OpenCV 灰度图 | → PIL → transform |
| numpy (H, W, 3) uint8 | OpenCV 彩色图 | RGB→灰度 → PIL → transform |
| torch Tensor (C, H, W) | 已预处理 | 直接传入模型 |

```python
# 文件路径
result = predictor.predictFromFile("cat.jpg", topK=5)

# numpy 数组（OpenCV 读取）
import cv2
img = cv2.imread("cat.jpg")
result = predictor.predict(img, topK=3)

# torch Tensor（另一模型的输出、已归一化）
tensor = torch.randn(3, 224, 224)  # 已预处理的张量
result = predictor.predict(tensor)
```

### 批量推理

```python
images = [Image.open(f"test_{i}.png") for i in range(100)]
results = predictor.predictBatch(images, topK=3, batchSize=32)

for i, r in enumerate(results):
    top1 = r[0]
    print(f"图 {i}: {top1['class_name']} ({top1['confidence']:.2%})")
```

批量推理自动分批处理，最大化 GPU 利用率。

### 返回结果结构

```python
# 单张推理返回 List[Dict]
[
    {
        "class_index": 3,
        "class_name": "cat",
        "confidence": 0.9234,
    },
    {
        "class_index": 5,
        "class_name": "dog",
        "confidence": 0.0456,
    },
    ...
]
```

### 使用 classNames

```python
# CIFAR-10 类别名
classNames = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

predictor = Predictor(model, transform, classNames=classNames)
result = predictor.predictFromFile("cat.jpg")
# → class_name: "cat" 而非 "3"
```

若不提供 `classNames`，则使用类别索引的数字字符串作为名称。

---

## 推理性能

| 模型 | 输入尺寸 | CPU 推理 (ms) | GPU 推理 (ms) |
|------|:---:|:---:|:---:|
| LeNet-5 | 32×32 | <5 | <1 |
| NiN | 32×32 | ~10 | <2 |
| AlexNet | 224×224 | ~80 | ~5 |
| GoogLeNet | 224×224 | ~60 | ~5 |
| VGG16 | 224×224 | ~300 | ~15 |

（仅供参考，具体取决于硬件。GPU 为单 batch 推理，不含数据加载时间。）

---

## 相关文档

- [快速开始](/guides/quickstart) — 首个推理命令
- [训练指南](/guides/training-guide) — 如何训练得到 checkpoint
- [推理系统](/architecture/inference) — Predictor 内部实现
- [Softmax](/math/softmax) — logits → 概率的数学原理
