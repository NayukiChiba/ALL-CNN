# AlexNet (2012)

## 论文来源

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS.

**历史地位**: 首个赢得 ImageNet 竞赛的 CNN，标志着深度学习革命的开始。将 top-5 错误率从 26.2% 降至 15.3%（2012 年），远超传统计算机视觉方法。

---

## 架构图

```
输入 X₀ : (B, 3, 224, 224)
  │
  ├─ Conv1: Conv2d(3→96, 11×11, stride=4, pad=2) + ReLU    → (B, 96, 55, 55)
  ├─ MaxPool1: MaxPool2d(3×3, stride=2)                     → (B, 96, 27, 27)
  │
  ├─ Conv2: Conv2d(96→256, 5×5, pad=2) + ReLU              → (B, 256, 27, 27)
  ├─ MaxPool2: MaxPool2d(3×3, stride=2)                     → (B, 256, 13, 13)
  │
  ├─ Conv3: Conv2d(256→384, 3×3, pad=1) + ReLU             → (B, 384, 13, 13)
  ├─ Conv4: Conv2d(384→384, 3×3, pad=1) + ReLU             → (B, 384, 13, 13)
  ├─ Conv5: Conv2d(384→256, 3×3, pad=1) + ReLU             → (B, 256, 13, 13)
  ├─ MaxPool5: MaxPool2d(3×3, stride=2)                     → (B, 256, 6, 6)
  │
  ├─ Flatten                                                 → (B, 9216)
  │
  ├─ FC1: Linear(9216→4096) + BN1d + ReLU + Dropout(0.5)   → (B, 4096)
  ├─ FC2: Linear(4096→4096) + BN1d + ReLU + Dropout(0.5)   → (B, 4096)
  └─ Output: Linear(4096→num_classes)                        → (B, num_classes)
```



---

## 逐层详解

### 卷积层组

**Conv1 — 大核 + 大步长快速降维**

- 11×11 卷积核，stride=4：一次性将 224×224 降至 55×55
- 96 个输出通道，学习低级特征（边缘、纹理、颜色斑点）
- **参数量**: $96 \times (3 \times 11 \times 11 + 1) = 34,944$

**MaxPool1** — 55×55 → 27×27（3×3 窗口，2 步长，有重叠）

**Conv2 — 中等核细化特征**

- 5×5 卷积核，padding=2 保持 27×27 空间尺寸
- 256 个输出通道
- **参数量**: $256 \times (96 \times 5 \times 5 + 1) = 614,656$

**MaxPool2** — 27×27 → 13×13

**Conv3-5 — 小核堆叠深度**

- 全部 3×3 卷积，padding=1，保持 13×13
- Conv3: 256→384, Conv4: 384→384, Conv5: 384→256
- **参数量**: Conv3: $384 \times (256 \times 9 + 1) = 885,120$; Conv4: $384 \times (384 \times 9 + 1) = 1,327,488$; Conv5: $256 \times (384 \times 9 + 1) = 884,992$

**MaxPool5** — 13×13 → 6×6

### 全连接层组

利用 `linear_block`（[cnnlib/models/blocks.py](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/blocks.py) — `Linear → BN1d → ReLU → Dropout`）：

- **FC1**: 9216 → 4096，参数量 $4096 \times 9216 + 4096 = 37,752,832$
- **FC2**: 4096 → 4096，参数量 $4096 \times 4096 + 4096 = 16,781,312$
- **Output**: 4096 → num_classes

> FC1 单层即占约 38M 参数——超过总参数的 65%。这是 AlexNet 的主要参数量来源。

---

## 参数量明细

| 层 | 参数 | 累计 |
|----|------|------|
| Conv1 (3→96, 11×11) | 34,944 | 34,944 |
| Conv2 (96→256, 5×5) | 614,656 | 649,600 |
| Conv3 (256→384, 3×3) | 885,120 | 1,534,720 |
| Conv4 (384→384, 3×3) | 1,327,488 | 2,862,208 |
| Conv5 (384→256, 3×3) | 884,992 | 3,747,200 |
| FC1 (9216→4096) | 37,752,832 | 41,500,032 |
| BN1d | 8,192 | 41,508,224 |
| FC2 (4096→4096) | 16,781,312 | 58,289,536 |
| BN1d | 8,192 | 58,297,728 |
| Output (4096→1000) | 4,097,000 | **~62M** (ImageNet) |
| Output (4096→10) | 40,970 | **~57M** (CIFAR-10) |

---

## 关键创新

### 1. ReLU 激活函数

AlexNet 是首个在大规模 CNN 中使用 ReLU 的网络。原始论文通过实验证明 ReLU 训练的收敛速度是 Tanh 的 6 倍。

$$\text{ReLU}(x) = \max(0, x)$$

- **不饱和**: 正区间梯度恒为 1，不存在 Sigmoid/Tanh 的梯度消失
- **稀疏激活**: 负区间输出为 0，约 50% 神经元激活，有利于特征解耦
- **计算简单**: 仅需一次 threshold 操作，无需指数计算



### 2. Dropout 正则化

AlexNet 首次在 CNN 中大规模使用 Dropout。全连接层（FC1/FC2）使用 50% Dropout：

$$y = \frac{1}{1-p} \cdot r \odot x, \quad r_j \sim \text{Bernoulli}(1-p)$$

- 训练时每个 batch 随机丢弃一半神经元
- 等价于训练 $2^{4096}$ 个不同子网络的隐式集成
- 有效防止 FC 层的过拟合（FC 层拥有大量参数）

详见 [Dropout](/math/dropout)

### 3. 大卷积核（11×11, 5×5）

原始 AlexNet 使用 11×11 和 5×5 的大卷积核。这是 2012 年的设计选择——当时认为大核可以更好地捕捉全局纹理信息。现代架构（VGG 之后）几乎全部使用 3×3 的堆叠替代大核。

### 4. 数据增强（原始论文）

原始论文使用了两种增强：
- 随机裁剪 + 水平翻转
- **PCA 颜色增强**: 对 RGB 通道做 PCA，沿主成分方向加扰动（本项目未实现，因为此技术后来被更简单的 ColorJitter 替代）

---

## 与原始论文的实现差异

| 特性 | 原始 AlexNet (2012) | 本项目实现 |
|------|-------------------|-----------|
| GPU 数量 | 双 GPU（每 GPU 各一半通道） | 单 GPU（全通道） |
| Local Response Normalization | ✓ 在每个 MaxPool 后 | ✗（BN 更有效） |
| BatchNorm | ✗（2012 年不存在） | LinearBlock 内含有 BN1d |
| 分组卷积 | Conv2/4/5 使用 group=2 | 无分组（单 GPU） |
| 展平维度 | 硬编码 9216 | `infer_feature_dim()` 自动计算 |
| 权重初始化 | 均值 0 标准差 0.01 高斯 | Conv: Kaiming, Linear: Xavier |

---

## 权重初始化

```python
def _initWeights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
```

- **Conv 层**: Kaiming 正态初始化，考虑 ReLU 的非线性特性（只保留一半方差）
- **Linear 层**: Xavier 正态初始化，假设激活前分布对称

详见 [Kaiming & Xavier 初始化](/math/initialization)

---

## 感受野计算

| 层 | 核大小 | 步长 | 累积步长积 | 感受野 |
|----|--------|------|-----------|--------|
| 输入 | — | — | — | 1 |
| Conv1 | 11 | 4 | 1 | 11 |
| MaxPool1 | 3 | 2 | 4 | 19 |
| Conv2 | 5 | 1 | 8 | 51 |
| MaxPool2 | 3 | 2 | 8 | 67 |
| Conv3 | 3 | 1 | 16 | 99 |
| Conv4 | 3 | 1 | 16 | 131 |
| Conv5 | 3 | 1 | 16 | 163 |
| MaxPool5 | 3 | 2 | 16 | 195 |

> Conv5 层每个神经元的感受野约 195×195，覆盖了 224×224 输入的绝大部分区域。

---

## forward() 方法

```python
def forward(self, x):
    x = self.conv1(x)       # Conv2d+ReLU
    x = self.pool1(x)       # MaxPool
    x = self.conv2(x)       # Conv2d+ReLU
    x = self.pool2(x)       # MaxPool
    x = self.conv3(x)       # Conv2d+ReLU
    x = self.conv4(x)       # Conv2d+ReLU
    x = self.conv5(x)       # Conv2d+ReLU
    x = self.pool5(x)       # MaxPool
    x = torch.flatten(x, 1) # 展平
    x = self.classifier(x)  # FC1 → FC2 → Output
    return x
```

**源码**: [cnnlib/models/alexnet.py:135-146](https://github.com/NayukiChiba/ALL-CNN/blob/main/cnnlib/models/alexnet.py#L135-L146)

---

## 训练建议

- **推荐数据集**: CIFAR-10, CIFAR-100, SVHN（小尺寸 RGB）
- **输入尺寸要求**: 224×224（Transform 自动 Resize）
- **GPU 推荐**: 参数量 ~57M，建议 GPU 训练
- **典型表现**: CIFAR-10 上 80-90% 准确率（依赖训练 epoch 数）
- **过拟合风险**: 在小数据集上容易过拟合，建议增大 Dropout 或减小 FC 维度

---

## 相关文档

- [各层公式与设计原理](/math/layers) — ReLU/MaxPool/Dropout 公式
- [Kaiming & Xavier 初始化](/math/initialization) — Kaiming vs Xavier
- [Dropout](/math/dropout) — Dropout 正则化原理
- [感受野计算](/math/receptive-field) — 逐层递推公式
