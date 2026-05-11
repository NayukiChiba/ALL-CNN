# 推理系统

## 设计概览

`Predictor` 类（[predictor.py:42-366](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/inference/predictor.py#L42-L366)）封装了从检查点加载到后处理的完整推理流程，接受 **5 种输入格式**：

| 输入格式 | 示例 | 处理路径 |
|---------|------|---------|
| PIL Image | `Image.open("digit.png")` | 直接 resize |
| numpy (H,W) uint8 | OpenCV 读取结果 | → PIL → resize |
| numpy (H,W,3) uint8 | 彩色照片 | 亮度加权灰度化 → PIL → resize |
| torch Tensor (1,28,28) 已归一化 | 另一个模型的输出 | 直接通过 |
| 文件路径 | `"digit.png"` | PIL.open → resize |

---

## 构造函数

```python
class Predictor:
    def __init__(self, checkpoint_path, device=None,
                 conv_channels=None, hidden_size=None, dropout=None):
        # 1. 创建模型（使用与训练时相同的架构参数）
        self.model = createModel(
            conv_channels=conv_channels,
            hidden_size=hidden_size,
            dropout=dropout,
            device=device,
        )

        # 2. 加载检查点（只加载模型权重，不需要优化器）
        loadCheckpoint(checkpoint_path, self.model, optimizer=None)

        # 3. 切换到评估模式
        self.model.eval()

        # 4. 设置预处理变换（与验证/测试集相同）
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        ])
```

**关键决策：**

- `loadCheckpoint(..., optimizer=None)` — 推理不需要优化器状态，只需模型权重
- `model.eval()` — 冻结 BatchNorm 统计量，关闭 Dropout
- `self.transform` 包含 `ToTensor + Normalize`，与训练时的验证管线一致——但**不含** RandomAffine 增强

---

## 核心推理方法：predict()

[predictor.py:132-179](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/inference/predictor.py#L132-L179)：

```python
def predict(self, image, top_k=1):
    tensor = self._preprocess(image)          # → (1, 1, 28, 28)
    tensor = tensor.to(self.device)

    with torch.no_grad():
        logits = self.model(tensor)           # (1, 10)
        probabilities = F.softmax(logits, dim=1)  # (1, 10)

    return self._buildResult(probabilities, top_k)
```

返回结构（`_buildResult`，[predictor.py:319-354](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/inference/predictor.py#L319-L354)）：

```python
{
    "class_index": 3,
    "class_name": "3",
    "confidence": 0.9876,
    "probabilities": tensor([0.001, 0.002, ..., 0.9876, ...]),  # (10,)
    "top_k": [
        {"class_index": 3, "class_name": "3", "confidence": 0.9876},
        {"class_index": 8, "class_name": "8", "confidence": 0.0065},
        ...
    ]
}
```

---

## 多格式预处理：_preprocess()

[_preprocess()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/inference/predictor.py#L233-L287) 是所有输入格式的统一入口。核心策略是**统一转换为 PIL 灰度图 → resize(28,28) → 应用 transform**。

### 1. 文件路径

```python
if isinstance(image, (str, Path)):
    image = Image.open(image)
```

### 2. PIL Image（任意尺寸、RGB 或灰度）

```python
if image.mode != "L":        # 非灰度
    image = image.convert("L")  # RGB → 灰度
image = image.resize((28, 28), Image.LANCZOS)
```

使用 **LANCZOS 重采样**（高质量降采样）而非 NEAREST（模糊、有锯齿）。

### 3. numpy 数组

[_numpyToPIL()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/inference/predictor.py#L289-L317) 处理三种 numpy 形状：

| 形状 | 处理 |
|------|------|
| `(H, W)` | 直接转 PIL 灰度图 |
| `(H, W, 1)` | 去掉单通道尾维，转灰度 |
| `(H, W, 3)` | RGB→灰度: `0.299R + 0.587G + 0.114B` |

亮度加权系数来自 ITU-R BT.601 标准（人眼对绿色最敏感，蓝色最不敏感）。

```python
# RGB → 灰度（亮度加权）
gray = (0.299 * rgb[:,:,0] +
        0.587 * rgb[:,:,1] +
        0.114 * rgb[:,:,2])
```

uint8 [0,255] 直接转 PIL；float32 [0,1] 先缩放到 [0,255] 再转。

### 4. torch Tensor

- `(1, 28, 28)` 已归一化：假设已经过标准化，直接使用（不做额外处理）
- `(H, W)` 未归一化：转为 PIL 后走正常管线

### 最终的变换序列

```python
# 此时 image 一定是 PIL 灰度图 28x28
tensor = self.transform(image)   # ToTensor() + Normalize()
tensor = tensor.unsqueeze(0)     # (1, 28, 28) → (1, 1, 28, 28)
```

---

## 批量推理：predictBatch()

[predictor.py:181-227](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/inference/predictor.py#L181-L227)：

```python
def predictBatch(self, images, top_k=1, batch_size=64):
    # 第一步：预处理所有图像（CPU 操作）
    tensors = [self._preprocess(img) for img in images]

    # 第二步：按 batch_size 分组进行 GPU 推理
    results = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.cat(tensors[i:i+batch_size]).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
        results.extend([self._buildResult(p, top_k) for p in probs])
    return results
```

分离 CPU 预处理和 GPU 推理以最大化吞吐。

---

## 推理脚本

[scripts/infer.py:21-67](https://github.com/NayukiChiba/MNIST-CNN/blob/main/scripts/infer.py#L21-L67) 支持两种模式：

### 单图推理

```bash
python main.py infer --checkpoint checkpoints/best_model.pth --image digit.png
```

输出示例（[infer.py:70-94](https://github.com/NayukiChiba/MNIST-CNN/blob/main/scripts/infer.py#L70-L94)）：

```
==================================================
  预测结果
==================================================
  预测数字: 7
  置信度: 99.35%

  Top-3 预测:
  1. 7: 99.35% ████████████████████████████████
  2. 9:  0.41% █
  3. 1:  0.12%

  完整概率分布:
    0:  0.01%
    1:  0.12%
    2:  0.03%
    ...
```

### 目录批量推理

```bash
python main.py infer --checkpoint checkpoints/best_model.pth --image-dir test_images/
```

如果文件名以数字开头（如 `3_mydigit.png`），自动提取作为真实标签，推理脚本会计算准确率（[infer.py:96-141](https://github.com/NayukiChiba/MNIST-CNN/blob/main/scripts/infer.py#L96-L141)）。

---

## 便捷函数：predictImage

[predictImage()](https://github.com/NayukiChiba/MNIST-CNN/blob/main/src/inference/predictor.py#L373-L396) 提供一行推理：

```python
result = predictImage("digit.png")
print(f"预测: {result['class_name']}, 置信度: {result['confidence']:.2%}")
```

内部创建临时 `Predictor` 实例并调用 `predict()`。适合在 Jupyter notebook 或交互式 Python 中使用。

---

## 推理命令示例

```bash
# 单图推理
python main.py infer \
    --checkpoint checkpoints/best_model.pth \
    --image my_digit.png

# 返回 top-5 预测
python main.py infer \
    --checkpoint checkpoints/best_model.pth \
    --image my_digit.png \
    --top-k 5

# 批量推理目录下所有图片
python main.py infer \
    --checkpoint checkpoints/best_model.pth \
    --image-dir ./test_images/
```
