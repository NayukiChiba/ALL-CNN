# FLOPs 计算量估算

## FLOPs 定义

FLOPs（Floating Point Operations）衡量模型做一次前向传播所需的浮点运算次数。它是模型**推理速度**的理论上限指标——FLOPs 越大，推理越慢。

注意：FLOPs 不等于实际推理时间（受内存带宽、IO、并行度等影响），但它是架构层面最常用的计算效率指标。

---

## 基本公式

### Conv2d

一次卷积输出的一个元素需要 $k_h \times k_w \times C_{\text{in}}$ 次乘法和加法：

$$\text{FLOPs} = 2 \times k_h \times k_w \times C_{\text{in}} \times C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}$$

- $k_h \times k_w$: 卷积核空间尺寸
- $C_{\text{in}}$: 输入通道数
- $C_{\text{out}}$: 输出通道数
- $H_{\text{out}} \times W_{\text{out}}$: 输出空间尺寸
- 系数 2: 一次乘法 + 一次加法 = 2 FLOPs

忽略偏置加法（每个输出元素仅 1 次加法，相比乘加运算可忽略）。

### Linear (全连接层)

$$\text{FLOPs} = 2 \times C_{\text{in}} \times C_{\text{out}}$$

因为一个输出元素需要 $C_{\text{in}}$ 次乘加。

### MaxPool2d / AvgPool2d

$$\text{FLOPs} \approx k_h \times k_w \times C \times H_{\text{out}} \times W_{\text{out}}$$

通常远小于卷积，在估算中常忽略。

### ReLU

每个激活元素一次比较（取 max(0, x)），现代硬件上近乎免费：

$$\text{FLOPs} \approx C \times H \times W$$

在实践中常忽略不计。

### BatchNorm（推理时）

推理时 BN 退化为仿射变换 $y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$，每个元素约 5 FLOPs（减、除、乘、加），通常忽略不计。

---

## LeNet-5 FLOPs（1×32×32）

| 层 | 计算式 | FLOPs |
|---|------|------:|
| Conv2d C1 | $2\times5\times5\times1\times6\times28\times28$ | 235,200 |
| AvgPool S2 | $2\times2\times6\times14\times14$ | ~4,704 |
| Conv2d C3 | $2\times5\times5\times6\times16\times10\times10$ | 480,000 |
| AvgPool S4 | $2\times2\times16\times5\times5$ | ~1,600 |
| Conv2d C5 | $2\times5\times5\times16\times120\times1\times1$ | 96,000 |
| Linear F6 | $2\times120\times84$ | 20,160 |
| Linear Output | $2\times84\times10$ | 1,680 |
| **总计** | | **~0.84M** |

LeNet-5 仅需 ~0.84M FLOPs——现代 CPU 可在 <1ms 完成推理。

---

## AlexNet FLOPs（3×224×224）

| 层 | 输出尺寸 | FLOPs |
|---|:---|------:|
| Conv2d 1 (11×11, s=4) | 96×55×55 | 2×11×11×3×96×55×55 = ~210M |
| Conv2d 2 (5×5) | 256×27×27 | 2×5×5×96×256×27×27 = ~448M |
| Conv2d 3 (3×3) | 384×13×13 | 2×3×3×256×384×13×13 = ~299M |
| Conv2d 4 (3×3) | 384×13×13 | 2×3×3×384×384×13×13 = ~448M |
| Conv2d 5 (3×3) | 256×13×13 | 2×3×3×384×256×13×13 = ~299M |
| Linear FC1 | 9216→4096 | 2×9216×4096 = ~75.5M |
| Linear FC2 | 4096→4096 | 2×4096×4096 = ~33.6M |
| Linear Output | 4096→10 | 2×4096×10 = ~0.08M |
| **总计** | | **~1.81G** |

**GFLOPs = 1,810 MFLOPs = 1.81 GFLOPs**（对 10 类分类；ImageNet 1000 类约 1.82G）

---

## VGG16 FLOPs（3×224×224）

| Stage | 层 | 输出尺寸 | FLOPs |
|:---:|---|:---|------:|
| 1 | Conv3-64 ×2 | 64×224×224 | 2×3×3×(3×64+64×64)×224×224 ≈ 2.7G |
| 2 | Conv3-128 ×2 | 128×112×112 | 2×3×3×(64×128+128×128)×112×112 ≈ 2.3G |
| 3 | Conv3-256 ×3 | 256×56×56 | 3×2×3×3×128×256×56×56 ≈ 3.5G |
| 4 | Conv3-512 ×3 | 512×28×28 | 3×2×3×3×256×512×28×28 ≈ 5.5G |
| 5 | Conv3-512 ×3 | 512×14×14 | 3×2×3×3×512×512×14×14 ≈ 5.5G |
| FC1 | 25088→4096 | — | ~205.5M |
| FC2 | 4096→4096 | — | ~33.6M |
| FC3 | 4096→10 | — | ~0.08M |
| **总计** | | | **~15.5G** |

VGG16 的 ~15.5 GFLOPs 中，前两个卷积 stage 占了约 1/3（空间大），FC1 又占了一个不小的比例（输入维度 25088）。

---

## NiN FLOPs（3×32×32, 10 类）

| 层 | FLOPs |
|---|------:|
| nin_block 1 + MaxPool | ~70M |
| nin_block 2 + MaxPool | ~85M |
| nin_block 3 + MaxPool | ~18M |
| nin_block cls + GAP | ~4M |
| **总计** | **~177M** |

~177M FLOPs，比 AlexNet 的 ~1.8G 少了约 10 倍。

---

## GoogLeNet FLOPs（3×224×224）

| 模块 | FLOPs |
|:---|------:|
| Stem | ~130M |
| Inception 3a-3b | ~300M |
| Inception 4a-4e | ~700M |
| Inception 5a-5b | ~300M |
| Head | ~0.1M |
| **总计** | **~1.43G** |

GoogLeNet ~1.43 GFLOPs，比 AlexNet 的 ~1.81G 少 ~21%，比 VGG16 的 ~15.5G 少 ~91%。

---

## 八个模型 FLOPs 对比

<div style="max-width:520px;margin:1em auto;font-size:13px;line-height:1.8;">
  <div style="text-align:center;font-weight:600;margin-bottom:8px;">八个模型 FLOPs 对比 (GFLOPs)</div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">LeNet</span>
    <span style="height:14px;background:#3498db;width:0.004%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">0.00084G</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">NiN</span>
    <span style="height:14px;background:#3498db;width:0.89%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">0.177G</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">GoogLeNet</span>
    <span style="height:14px;background:#3498db;width:7.15%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">1.43G</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">AlexNet</span>
    <span style="height:14px;background:#3498db;width:9.05%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">1.81G</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">VGG11</span>
    <span style="height:14px;background:#3498db;width:66.5%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">13.3G</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">VGG13</span>
    <span style="height:14px;background:#3498db;width:71%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">14.2G</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">VGG16</span>
    <span style="height:14px;background:#3498db;width:77.5%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">15.5G</span>
  </div>
  <div style="display:flex;align-items:center;">
    <span style="width:80px;text-align:right;margin-right:8px;flex-shrink:0;">VGG19</span>
    <span style="height:14px;background:#3498db;width:83.5%;display:inline-block;border-radius:2px;min-width:2px;"></span>
    <span style="margin-left:6px;flex-shrink:0;">16.7G</span>
  </div>
</div>

| 模型 | MFLOPs | GFLOPs | 相对 LeNet | FLOPs/参数比 |
|------|------:|------:|:---:|:---:|
| LeNet-5 | 0.84 | <0.01 | 1× | 13.6 |
| NiN | 177 | 0.18 | 211× | 177 |
| GoogLeNet | 1,430 | 1.43 | 1,702× | 270 |
| AlexNet | 1,810 | 1.81 | 2,155× | 31 |
| VGG11 | 13,300 | 13.3 | 15,833× | 101 |
| VGG13 | 14,200 | 14.2 | 16,905× | 107 |
| VGG16 | 15,500 | 15.5 | 18,452× | 115 |
| VGG19 | 16,700 | 16.7 | 19,881× | 123 |

**关键观察**:
- LeNet-5 的计算量极低（<1 MFLOP），可在嵌入式设备实时运行
- AlexNet Conv1 的 FLOPs 占了总体的 ~11.5%——大卷积核 + 大空间 = 高计算
- VGG 计算量巨大但参数不多——计算密集型的代表
- GoogLeNet FLOPs 合理但参数量小——适合移动端部署

---

## FLOPs vs 参数量 vs 推理速度


这三者相关但不等价：

| 指标 | 含义 | 影响因素 |
|------|------|---------|
| 参数量 | 模型文件大小、显存占用 | 权重数量 |
| FLOPs | 理论计算量上限 | 输入大小、层数、通道数 |
| 推理速度 | 实际耗时（ms/image） | FLOPs + 内存带宽 + 并行度 + 硬件 |

例如：VGG16 的 FLOPs 是 AlexNet 的 8.6 倍，但参数大小只有 2.3 倍——VGG 计算密集（大量 3×3 卷积），推理比 AlexNet 慢得多，尽管参数只多一倍。

---

## 相关文档

- [参数计算](/math/parameter-count) — 参数量公式
- [模型总览](/models/overview) — 8 模型对比表
- [感受野计算](/math/receptive-field) — 逐层感受野
