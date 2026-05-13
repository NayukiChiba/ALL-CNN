# 参数量计算

## 基本公式

### Conv2d

$$\text{params} = k_h \times k_w \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$$

- 前项是**权重（weight）**：每个输出通道有一个 $k_h \times k_w \times C_{\text{in}}$ 的卷积核，共 $C_{\text{out}}$ 个
- 后项是**偏置（bias）**：每个输出通道一个

有 BatchNorm 时额外加上：

$$\text{BN params} = 2 \times C_{\text{out}} \quad (\gamma \text{ 和 } \beta)$$

### Linear (全连接层)

$$\text{params} = C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$$

FC 层的参数量是输入维度 × 输出维度——这也是为什么 FC 层参数占比极高。

### BatchNorm1d / BatchNorm2d

$$\text{params} = 2 \times C \quad (\gamma, \beta)$$

$\mu$ 和 $\sigma^2$ 是运行时统计量，不计入可训练参数。

---

## LeNet-5 参数量（MNIST: 1×32×32, 10 类）

| 层 | 计算式 | 参数量 |
|---|------|:---:|
| Conv2d C1 | $5\times5\times1\times6 + 6$ | 156 |
| AvgPool S2 | — | 0 |
| Conv2d C3 | $5\times5\times6\times16 + 16$ | 2,416 |
| AvgPool S4 | — | 0 |
| Conv2d C5 | $5\times5\times16\times120 + 120$ | 48,120 |
| Linear F6 | $120\times84 + 84$ | 10,164 |
| Linear Output | $84\times10 + 10$ | 850 |
| **总计** | | **61,706 (~62K)** |

FC 层占比：$(10164 + 850) / 61706 \approx 17.8\%$——相比后续模型非常低，因为输入小且只有一层 FC。

---

## AlexNet 参数量（3×224×224, 10 类）

| 层 | 计算式 | 参数量 |
|---|------|------:|
| Conv2d 1 | $11\times11\times3\times96 + 96$ | 34,944 |
| Conv2d 2 | $5\times5\times96\times256 + 256$ | 614,656 |
| Conv2d 3 | $3\times3\times256\times384 + 384$ | 885,120 |
| Conv2d 4 | $3\times3\times384\times384 + 384$ | 1,327,488 |
| Conv2d 5 | $3\times3\times384\times256 + 256$ | 884,992 |
| Linear FC1 | $9216\times4096 + 4096$ | 37,752,832 |
| BN1d FC1 | $2\times4096$ | 8,192 |
| Linear FC2 | $4096\times4096 + 4096$ | 16,781,312 |
| BN1d FC2 | $2\times4096$ | 8,192 |
| Linear Output | $4096\times10 + 10$ | 40,970 |
| **总计** | | **~58.3M** |

FC 层占比：$(37.8\text{M} + 16.8\text{M} + 0.04\text{M}) / 58.3\text{M} \approx 93.7\%$

AlexNet 的参数分布极度不均衡——两层 FC(4096) 占了整个模型参数的 ~94%。这也是为什么后来的 NiN 要用 GAP 替代 FC。

---

## VGG16 参数量（3×224×224, 10 类）

| Stage | 层 | 计算式 | 参数量 |
|:---:|---|------|------:|
| 1 | Conv3-64 ×2 | $2 \times (3\times3\times3\times64 + 64)$ | 3,584 |
| | BN ×2 | $2 \times 2\times64$ | 256 |
| 2 | Conv3-128 ×2 | $\begin{aligned}&3\times3\times64\times128+128\\ &+ 3\times3\times128\times128+128\end{aligned}$ | 221,440 |
| | BN ×2 | $2\times(128+128)$ | 512 |
| 3 | Conv3-256 ×3 | 三层 3×3, 通道 128→256→256→256 | 1,475,328 |
| | BN ×3 | — | 1,536 |
| 4 | Conv3-512 ×3 | 三层 3×3, 通道 256→512→512→512 | 5,899,776 |
| | BN ×3 | — | 3,072 |
| 5 | Conv3-512 ×3 | 三层 3×3, 通道 512→512→512→512 | 7,078,400 |
| | BN ×3 | — | 3,072 |
| | Linear FC1 | $25088\times4096 + 4096$ | 102,764,544 |
| | BN1d FC1 | $2\times4096$ | 8,192 |
| | Linear FC2 | $4096\times4096 + 4096$ | 16,781,312 |
| | BN1d FC2 | $2\times4096$ | 8,192 |
| | Linear Output | $4096\times10 + 10$ | 40,970 |
| **总计** | | | **~134.3M** |

FC 层占比：$(102.8\text{M} + 16.8\text{M} + 0.04\text{M}) / 134.3\text{M} \approx 89.0\%$

VGG16 的 134M 参数中，~120M 在 FC 层——卷积部分虽然深但参数并不多（全部 3×3 卷积）。

---

## NiN 参数量（3×32×32, 10 类）

| 层 | 计算式 | 参数量 |
|---|------|------:|
| nin_block 1 (3→192) | Conv5 + 1×1 + 1×1 | ~213K |
| nin_block 2 (192→160) | Conv5 + 1×1 + 1×1 | ~311K |
| nin_block 3 (160→96) | Conv3 + 1×1 + 1×1 | ~92K |
| nin_block cls (96→10) | Conv3 + 1×1 + 1×1 | ~10K |
| GAP | — | 0 |
| **总计** | | **~1.0M** |

**完全无 FC 层**！所有参数都在 mlpconv 的 $1\times1$ 和 $k\times k$ 卷积中。1M 参数比 LeNet-5 的 62K 多、但比 AlexNet 的 58M 少了 ~50 倍。

---

## GoogLeNet 参数量（3×224×224, 10 类）

| 模块 | 估算参数量 |
|:---|------:|
| Stem（Conv 7×7 + 1×1 + 3×3） | ~50K |
| Inception 3a | ~160K |
| Inception 3b | ~340K |
| Inception 4a | ~360K |
| Inception 4b | ~390K |
| Inception 4c | ~420K |
| Inception 4d | ~530K |
| Inception 4e | ~900K |
| Inception 5a | ~960K |
| Inception 5b | ~1,200K |
| Head（GAP + Linear 1024→10） | ~10K |
| **总计** | **~5.3M** |

GoogLeNet 参数约 5M~7M（取决于输入大小和类别数），比 AlexNet 的 58M 少了 ~10 倍，比 VGG16 的 134M 少了 ~25 倍。核心原因是：
1. GAP 替代了大 FC 层（与 NiN 相同策略）
2. 1×1 bottleneck 压缩了 Inception 模块的通道数

---

## 八个模型参数量对比

<div style="max-width:520px;margin:1em auto;font-size:13px;line-height:1.8;">
  <div style="text-align:center;font-weight:600;margin-bottom:8px;">八个模型参数量对比 (M)</div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">LeNet</span>
    <span style="height:14px;background:#3498db;width:0.04%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">0.062M</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">NiN</span>
    <span style="height:14px;background:#3498db;width:0.67%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">1.0M</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">GoogLeNet</span>
    <span style="height:14px;background:#3498db;width:3.53%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">5.3M</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">AlexNet</span>
    <span style="height:14px;background:#3498db;width:38.87%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">58.3M</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">VGG11</span>
    <span style="height:14px;background:#3498db;width:88.13%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">132.2M</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">VGG13</span>
    <span style="height:14px;background:#3498db;width:88.53%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">132.8M</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">VGG16</span>
    <span style="height:14px;background:#3498db;width:89.53%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">134.3M</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">VGG19</span>
    <span style="height:14px;background:#3498db;width:90.47%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">135.7M</span>
  </div>
</div>

| 模型 | Conv 参数 | FC 参数 | 总参数 | FC 占比 | 相对于 LeNet |
|------|------:|------:|------:|:---:|:---:|
| LeNet-5 (MNIST) | 50,692 | 11,014 | **61,706** | 17.8% | 1× |
| NiN (CIFAR-10) | ~1,000,000 | 0 | **~1.0M** | 0% | 16× |
| GoogLeNet | ~5,290,000 | 10,250 | **~5.3M** | 0.2% | 86× |
| AlexNet | ~3,747,200 | 54,583,306 | **~58.3M** | 93.7% | 945× |
| VGG11 | ~9,291,000 | 122,897,098 | **~132.2M** | 93.0% | 2,143× |
| VGG13 | ~9,895,000 | 122,897,098 | **~132.8M** | 92.5% | 2,153× |
| VGG16 | ~11,301,000 | 119,596,810 | **~134.3M** | 89.0% | 2,177× |
| VGG19 | ~12,708,000 | 119,596,810 | **~135.7M** | 88.1% | 2,199× |

**关键规律**:
- FC 层是参数爆炸的根源（AlexNet/VGG 有 >88% 参数在 FC 层）
- NiN 和 GoogLeNet 通过 GAP 彻底消除了大 FC，参数量极小
- VGG 四个变体的 FC 结构相同，参数量差异仅来自卷积层数

---

## 1×1 卷积的"神奇"参数压缩

$1\times1$ 卷积的参数量为 $C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$——与 Linear 层完全相同，但作用在空间维度的每个像素上。在 Inception 模块中，$1\times1$ bottleneck 将输入通道压缩后再做大核卷积：

- 无瓶颈：$3\times3\times192\times256 = 442,368$ 参数
- 有瓶颈（1×1 压缩到 96 通道）：$1\times1\times192\times96 + 3\times3\times96\times256 = 18,432 + 221,184 = 239,616$ 参数

减少 $(442,368 - 239,616) / 442,368 \approx 45.8\%$ 的参数，同时仅损失少量表达能力。

---

## 相关文档

- [FLOPs 估算](/math/flops) — 计算量估算
- [感受野计算](/math/receptive-field) — 逐层感受野
- [模型总览](/models/overview) — 8 模型对比
- [GoogLeNet](/models/googlenet) — Inception 模块与 1×1 bottleneck
- [NiN](/models/nin) — GAP 消除 FC 层
