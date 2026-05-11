"""
自适应 Transform 管线

根据模型注册表和数据集注册表，自动拼出正确的预处理流程。

流程：
    原始图像 → ToTensor → [通道转换: 灰度→RGB] → Resize(模型尺寸) → Normalize(数据集统计量)

用法：
    from cnnlib.data.transform import build_transform

    train_tf = build_transform("vgg16", "cifar10", augment=True)
    test_tf  = build_transform("vgg16", "cifar10", augment=False)
"""

from torchvision import transforms

from cnnlib.registry.datasets import get_dataset_info
from cnnlib.registry.models import get_model_info


def build_transform(
    model_name: str,
    dataset_name: str,
    augment: bool = True,
) -> transforms.Compose:
    """
    根据模型和数据集自动构建 transform

    查询两张注册表：
        - 模型注册表 → 目标 input_size, channels
        - 数据集注册表 → 原始 mean, std

    Args:
        model_name:   模型名称，如 "vgg16", "lenet"
        dataset_name: 数据集名称，如 "cifar10", "mnist"
        augment:      训练模式 = True（含数据增强），推理/评估 = False

    Returns:
        torchvision.transforms.Compose 对象
    """
    model_info = get_model_info(model_name)
    dataset_info = get_dataset_info(dataset_name)

    target_size = model_info["input_size"]
    target_channels = model_info["channels"]
    dataset_channels = dataset_info["channels"]
    mean = dataset_info["mean"]
    std = dataset_info["std"]

    ops = []

    # 1. 转 Tensor
    ops.append(transforms.ToTensor())

    # 2. 通道转换：数据集是灰度，模型要 RGB → 将单通道复制为三通道
    if dataset_channels == 1 and target_channels == 3:
        ops.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

    # 3. 尺寸适配
    ops.append(transforms.Resize((target_size, target_size)))

    # 4. 数据增强（仅训练模式）
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.RandomRotation(degrees=10))

    # 5. 归一化
    ops.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(ops)


if __name__ == "__main__":
    # 灰度模型 + 灰度数据集（不发散）
    tf = build_transform("lenet", "mnist")
    print("lenet + mnist:", tf)

    # RGB 模型 + RGB 数据集（不变）
    tf = build_transform("vgg16", "cifar10")
    print("vgg16 + cifar10:", tf)

    # RGB 模型 + 灰度数据集（1→3 通道转换）
    tf = build_transform("vgg16", "mnist")
    print("vgg16 + mnist:", tf)

    # 训练模式
    tf = build_transform("vgg16", "cifar10", augment=True)
    print("vgg16 + cifar10 (train):", tf)
