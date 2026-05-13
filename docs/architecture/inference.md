# 推理系统

## 设计概览

`Predictor` 类（[cnnlib/inference/predictor.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/inference/predictor.py)）封装完整的推理流程，接受 **5 种输入格式**。

| 输入格式 | 示例 | 处理路径 |
|---------|------|---------|
| PIL Image | `Image.open("cat.jpg")` | 直接走 transform |
| 文件路径 (str/Path) | `"cat.jpg"` | PIL.open → transform |
| numpy (H, W) uint8 | OpenCV 灰度图 | → PIL → transform |
| numpy (H, W, 3) uint8 | OpenCV 彩色图 | RGB→灰度 → PIL → transform |
| torch Tensor (C, H, W) | 已预处理 | 直接传入模型 |

---

## 构造函数

```python
class Predictor:
    def __init__(self, model, transform, device="cpu", classNames=None):
        self.model = model.to(device)
        self.transform = transform
        self.device = device
        self.classNames = classNames   # 可选: 类别名称列表
        self.model.eval()              # 冻结 BN, 关闭 Dropout
```

**关键点**:
- `model.eval()` — 推理时 BN 使用运行统计量，Dropout 关闭
- `transform` — 不含数据增强的评估管线（`build_transform(model, dataset, augment=False)`）
- `classNames` — 如 `["airplane", "automobile", ...]`，不提供则用数字字符串

---

## 核心推理方法：predict()

```python
@torch.no_grad()
def predict(self, image, topK=3) -> List[Dict]:
    tensor = self._preprocess(image)          # → (C, H, W)
    tensor = tensor.unsqueeze(0).to(self.device)  # → (1, C, H, W)

    logits = self.model(tensor)               # (1, num_classes)
    probs = F.softmax(logits, dim=1)          # (1, num_classes)

    return self._buildResult(probs[0], topK)
```

返回结构：

```python
[
    {"class_index": 3, "class_name": "cat", "confidence": 0.9234},
    {"class_index": 5, "class_name": "dog", "confidence": 0.0456},
    {"class_index": 7, "class_name": "horse", "confidence": 0.0120},
]
```

---

## 多格式预处理：_preprocess()

核心策略是**统一转换为 PIL → 应用 transform**。

### 1. 文件路径

```python
if isinstance(image, (str, Path)):
    image = Image.open(image)
```

### 2. PIL Image

直接走 transform → tensor。

### 3. numpy 数组

| 形状 | 处理 |
|------|------|
| `(H, W)` | 直接转 PIL 灰度图 |
| `(H, W, 1)` | 去掉尾维，转 PIL 灰度图 |
| `(H, W, 3)` | RGB→灰度: `0.299R + 0.587G + 0.114B` |

亮度加权系数来自 ITU-R BT.601 标准。

### 4. torch Tensor

- `(C, H, W)` 已归一化：假设已经过标准化，直接使用
- `(H, W)` 未归一化：转为 PIL 后走正常管线

### 最终变换

```python
tensor = self.transform(image)    # ToTensor + [channel_convert] + Resize + Normalize
```

---

## 批量推理：predictBatch()

```python
def predictBatch(self, images, topK=3, batchSize=64):
    # 1. CPU 预处理所有图片
    tensors = [self._preprocess(img) for img in images]

    # 2. 分组 GPU 推理
    results = []
    for i in range(0, len(tensors), batchSize):
        batch = torch.stack(tensors[i:i+batchSize]).to(self.device)
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        results.extend([self._buildResult(p, topK) for p in probs])
    return results
```

预处理（CPU）和推理（GPU）分步进行，最大化 GPU 利用率。

---

## 便捷方法

### predictFromFile

```python
def predictFromFile(self, filepath, topK=3):
    image = Image.open(filepath)
    return self.predict(image, topK=topK)
```

### predictFromDir

```python
def predictFromDir(self, dirPath, topK=3):
    results = {}
    for imgPath in Path(dirPath).glob("*"):
        if imgPath.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            results[imgPath.name] = self.predictFromFile(imgPath, topK=topK)
    return results
```

---

## 推理命令示例

```bash
# 单图推理
python main.py --model lenet --dataset mnist infer \
    --checkpoint checkpoints/best_model.pth \
    --image my_digit.png

# Top-5 预测
python main.py --model vgg16 --dataset cifar10 infer \
    --checkpoint checkpoints/best_model.pth \
    --image cat.jpg --top-k 5

# 批量推理
python main.py --model lenet --dataset mnist infer \
    --checkpoint checkpoints/best_model.pth \
    --image-dir ./test_images/
```

---

## 推理 vs 训练的数据管线差异

| | 训练 | 推理 |
|---|:---:|:---:|
| ToTensor | ✓ | ✓ |
| 通道转换 (1→3) | ✓ | ✓ |
| Resize | ✓ | ✓ |
| RandomHorizontalFlip | ✓ | ✗ |
| RandomRotation | ✓ | ✗ |
| Normalize | ✓ | ✓ |

---

## 相关文档

- [推理指南](/guides/inference-guide) — Predictor 编程 API 使用
- [Softmax](/math/softmax) — logits → 概率的数学原理
- [数据管道](/architecture/data-pipeline) — build_transform 流程
- [模型工厂](/architecture/model-factory) — 创建和加载模型
