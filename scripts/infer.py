"""
推理脚本

单张图片或批量推理，输出 top-K 预测结果。

用法:
    python main.py --model vgg16 --dataset cifar10 infer \
        --image cat.jpg --checkpoint checkpoints/vgg16/cifar10/best_model.pth --top-k 5

    python main.py --model vgg16 --dataset cifar10 infer \
        --image-dir ./images/ --checkpoint checkpoints/vgg16/cifar10/best_model.pth
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image

from cnnlib.data.transform import build_transform
from cnnlib.models.factory import create_model_for_dataset
from cnnlib.training import loadCheckpoint


def _loadImage(imagePath: str, transform) -> torch.Tensor:
    """加载并预处理单张图片"""
    img = Image.open(imagePath).convert("RGB")
    return transform(img).unsqueeze(0)


def _predict(model, imageTensor: torch.Tensor, device: torch.device, topK: int):
    """推理并返回 top-k 结果"""
    model.eval()
    imageTensor = imageTensor.to(device)
    with torch.no_grad():
        outputs = model(imageTensor)
        probs = torch.softmax(outputs, dim=1)
        topProbs, topIndices = probs.topk(topK, dim=1)
    return topIndices[0].cpu().numpy(), topProbs[0].cpu().numpy()


def run(args: argparse.Namespace) -> None:
    """推理入口"""

    device = torch.device(args.device)

    # checkpoint 必填
    checkpointPath = getattr(args, "checkpoint", None)
    if checkpointPath is None:
        print("错误: --checkpoint 为必填参数")
        return

    # 模型
    model = create_model_for_dataset(args.model, args.dataset, device=str(device))

    # 加载 checkpoint
    ckpt = loadCheckpoint(checkpointPath, model, device=str(device))
    print(f"加载 checkpoint: {checkpointPath} (epoch {ckpt['epoch']})")

    # transform（不增强）
    transform = build_transform(args.model, args.dataset, augment=False)

    topK = getattr(args, "top_k", 3)

    # 单张推理
    if hasattr(args, "image") and args.image:
        imagePath = args.image
        if not os.path.exists(imagePath):
            print(f"错误: 图片不存在: {imagePath}")
            return

        imageTensor = _loadImage(imagePath, transform)
        indices, probs = _predict(model, imageTensor, device, topK)

        print(f"\n图片: {imagePath}")
        print(f"{'Rank':<6} {'Class':<8} {'Probability':<12}")
        print("-" * 30)
        for rank, (clsIdx, prob) in enumerate(zip(indices, probs), 1):
            print(f"{rank:<6} {clsIdx:<8} {prob:.4f}")

    # 批量推理
    elif hasattr(args, "image_dir") and args.image_dir:
        imageDir = Path(args.image_dir)
        if not imageDir.is_dir():
            print(f"错误: 目录不存在: {imageDir}")
            return

        imageFiles = list(imageDir.glob("*"))
        imageFiles = [
            f
            for f in imageFiles
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ]
        if not imageFiles:
            print(f"目录中无图片文件: {imageDir}")
            return

        print(f"\n目录: {imageDir} ({len(imageFiles)} 张图片)")
        print(f"{'File':<30} {'Top-1':<8} {'Confidence':<12}")
        print("-" * 55)

        for imageFile in sorted(imageFiles):
            imageTensor = _loadImage(str(imageFile), transform)
            indices, probs = _predict(model, imageTensor, device, topK)
            print(f"{imageFile.name:<30} {indices[0]:<8} {probs[0]:.4f}")

    else:
        print("错误: 请指定 --image 或 --image-dir")
