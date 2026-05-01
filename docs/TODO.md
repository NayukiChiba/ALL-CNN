基于 PyTorch 实现 MNIST 手写数字识别的 CNN 模型，完整工程化项目结构。

## 项目架构概览

```
MNIST-CNN/
├── main.py
│
├── config/
│   ├── __init__.py
│   ├── default.py
│   ├── paths.py
│   └── settings.py
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── transform.py
│   │   └── loader.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   ├── cnn.py
│   │   └── factory.py
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── loss.py
│   │   ├── optimizer.py
│   │   ├── checkpoint.py
│   │   └── logger.py
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualize.py
│   │
│   └── inference/
│       ├── __init__.py
│       └── predictor.py
│
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── infer.py
│
├── docs/
│   ├── architecture.md
│   ├── usage.md
│   └── api.md
│
├── checkpoints/
├── outputs/
├── datasets/
├── requirements.txt
├── pyproject.toml
└── README.md

```

---

## 子任务拆解

### 项目基础设施

- [x] **项目骨架搭建**
  - [x] 创建完整目录结构
  - [x] 完善 `requirements.txt`（torch, torchvision, numpy, matplotlib, tensorboard, tqdm）
  - [x] 完善 `pyproject.toml`（ruff 配置）
  - [x] 编写 `README.md`（项目说明、安装、快速开始）
  - [x] 更新 `.gitignore`（checkpoints/, outputs/, datasets/, logs/）

- [x] **配置模块 `config/`**
  - [x] `paths.py` — 项目路径常量（基于 PROJECT_ROOT 推导）
  - [x] `default.py` — 所有超参数默认值（batch_size, lr, epochs, model 结构, 设备等）
  - [x] `settings.py` — 合并默认值 + 命令行参数覆盖（argparse）
  - [x] `__init__.py` — 统一导出

- [x] **主入口 `main.py`**
  - [x] 子命令分发：`train` / `eval` / `infer`
  - [x] 全局参数：`--config`、`--device`、`--seed`
  - [x] 各子命令独立参数解析与调度

---

### 数据模块 `src/data/`

- [X] **数据集封装**
  - [X] `dataset.py` — `torch.utils.data.Dataset` 子类，封装 MNIST
  - [X] 支持自动下载到 `datasets/` 目录
  - [X] 返回 `(image, label)` 元组

- [ ] **预处理与数据增强**
  - [ ] `transform.py` — `ToTensor()` + `Normalize((0.1307,), (0.3081,))`
  - [ ] 训练集可选增强：`RandomAffine`(±10°旋转, ±10%平移)
  - [ ] 验证/测试集仅归一化
  - [ ] 从 config 读取增强开关

- [ ] **DataLoader 构建**
  - [ ] `loader.py` — 构建 train/val/test 三个 DataLoader
  - [ ] 支持 `num_workers`、`pin_memory` 配置
  - [ ] 验证集从训练集按比例切分

---

### 模型模块 `src/model/`

- [ ] **基础组件**
  - [ ] `layers.py` — `ConvBlock`（Conv2d + BatchNorm2d + ReLU + 可选 MaxPool）
  - [ ] `LinearBlock`（Linear + BatchNorm1d + ReLU + 可选 Dropout）
  - [ ] 组件参数可配置

- [ ] **CNN 模型**
  - [ ] `cnn.py` — 组装完整 CNN
  - [ ] 结构：ConvBlock × 2 → Flatten → LinearBlock × 2 → Linear(10)
  - [ ] `forward()` 返回 logits
  - [ ] 可选：打印每层 shape 和参数量

- [ ] **模型工厂**
  - [ ] `factory.py` — 从 config 字典构建模型实例
  - [ ] 支持切换不同架构配置（浅层/深层）
  - [ ] 模型 `.to(device)` 自动处理

---

### 训练模块 `src/train/`

- [ ] **损失函数**
  - [ ] `loss.py` — `nn.CrossEntropyLoss()` 封装

- [ ] **优化器与调度器**
  - [ ] `optimizer.py` — Adam 优化器构建
  - [ ] `ReduceLROnPlateau` scheduler（val loss 停滞时降 lr）
  - [ ] 从 config 读取参数

- [ ] **训练引擎**
  - [ ] `engine.py`
    - `trainEpoch()` — 单 epoch 训练循环（梯度计算、参数更新）
    - `validateEpoch()` — 单 epoch 验证循环（无梯度）
  - [ ] 返回 epoch 级别的 loss 和 accuracy
  - [ ] 使用 tqdm 显示进度条

- [ ] **Checkpoint 管理**
  - [ ] `checkpoint.py`
    - `save()` — 保存模型权重 + optimizer 状态 + epoch + 指标
    - `load()` — 恢复训练或加载用于推理
  - [ ] 自动保存最佳模型（best.pth）和最近模型（last.pth）

- [ ] **日志系统**
  - [ ] `logger.py`
    - 终端日志：epoch / loss / accuracy / lr
    - TensorBoard：scalar（loss 曲线、accuracy 曲线）、graph（模型图）
    - 日志输出到 `outputs/logs/`

---

### 评估模块 `src/eval/`

- [ ] **评估指标**
  - [ ] `metrics.py`
    - accuracy（整体 + 每类）
    - confusion matrix
    - precision / recall / F1（per-class）
    - 分类报告（classification report）

- [ ] **可视化**
  - [ ] `visualize.py`
    - 训练曲线图（loss / accuracy 双轴）
    - 混淆矩阵热力图
    - 错误预测样本网格（真实标签 vs 预测标签）

---

### 推理模块 `src/inference/`

- [ ] **推理器**
  - [ ] `predictor.py`
    - 加载训练好的 checkpoint
    - 单张图片推理（输入 PIL Image / numpy array / tensor）
    - 批量推理
    - 返回 top-k 类别和置信度
    - softmax 后处理

---

### 脚本与入口

- [ ] **#18 独立脚本**
  - [ ] `scripts/train.py` — 调用 `main.py train`
  - [ ] `scripts/eval.py` — 调用 `main.py eval`
  - [ ] `scripts/infer.py` — 调用 `main.py infer`

---

### 文档

- [ ] **#19 项目文档**
  - [ ] `docs/architecture.md` — 架构设计说明
  - [ ] `docs/usage.md` — 使用指南（安装、训练、评估、推理）
  - [ ] `docs/api.md` — 模块 API 参考

- [ ] **最终验收**
  - [ ] 全流程走通：`python main.py train` 完成训练
  - [ ] 测试集准确率 > 99%
  - [ ] 推理可用：输入手写图片输出预测结果
  - [ ] ruff check 无错误
  - [ ] README 完整

---


