"""Cache clean or controlled-corruption features from a trained ImageFolder model."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


CORRUPTIONS = ("clean", "gaussian_noise", "gaussian_blur", "brightness", "contrast", "jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "calibration", "test"], required=True)
    parser.add_argument("--corruption", choices=CORRUPTIONS, default="clean")
    parser.add_argument("--severity", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


class ControlledCorruption:
    """Deterministic-strength corruption used for the ImageNet100 pilot.

    These conditions are reported as controlled corruptions, not as ImageNet-C.
    The main benchmark uses the official corruption datasets separately.
    """

    def __init__(self, name: str, severity: int) -> None:
        self.name = name
        self.severity = severity

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.name == "clean":
            return image
        if self.name == "gaussian_blur":
            radius = (0.5, 1.0, 1.5, 2.0, 3.0)[self.severity - 1]
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        if self.name == "brightness":
            factor = (0.85, 0.70, 0.55, 0.40, 0.25)[self.severity - 1]
            return ImageEnhance.Brightness(image).enhance(factor)
        if self.name == "contrast":
            factor = (0.80, 0.65, 0.50, 0.35, 0.20)[self.severity - 1]
            return ImageEnhance.Contrast(image).enhance(factor)
        if self.name == "jpeg":
            quality = (80, 65, 50, 35, 20)[self.severity - 1]
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB")
        return image


class TensorGaussianNoise:
    def __init__(self, severity: int) -> None:
        self.std = (0.04, 0.08, 0.12, 0.18, 0.26)[severity - 1]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + self.std * torch.randn_like(tensor)).clamp(0.0, 1.0)


def build_transform(corruption: str, severity: int) -> transforms.Compose:
    operations = [transforms.Resize(256), transforms.CenterCrop(224)]
    if corruption != "gaussian_noise":
        operations.append(ControlledCorruption(corruption, severity))
    operations.append(transforms.ToTensor())
    if corruption == "gaussian_noise":
        operations.append(TensorGaussianNoise(severity))
    operations.append(
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    )
    return transforms.Compose(operations)


def build_model(checkpoint: Dict[str, object]) -> Tuple[nn.Module, nn.Linear]:
    architecture = str(checkpoint["architecture"])
    classes = len(checkpoint["classes"])
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
    elif architecture == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError("unsupported checkpoint architecture: " + architecture)
    model.fc = nn.Linear(model.fc.in_features, classes)
    model.load_state_dict(checkpoint["model"])
    classifier = model.fc
    model.fc = nn.Identity()
    return model, classifier


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, classifier = build_model(checkpoint)
    model.eval().cuda().to(memory_format=torch.channels_last)
    classifier.eval().cuda()

    dataset = datasets.ImageFolder(
        args.data_root,
        transform=build_transform(args.corruption, args.severity),
    )
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    indices = split[args.split]
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    feature_chunks = []
    logit_chunks = []
    label_chunks = []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f"{args.split}:{args.corruption}-{args.severity}"):
            images = images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = model(images)
                logits = classifier(features)
            feature_chunks.append(features.float().cpu())
            logit_chunks.append(logits.float().cpu())
            label_chunks.append(labels.long())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "features": torch.cat(feature_chunks).half(),
        "logits": torch.cat(logit_chunks).half(),
        "labels": torch.cat(label_chunks),
        "indices": torch.as_tensor(indices, dtype=torch.long),
        "classes": checkpoint["classes"],
        "architecture": checkpoint["architecture"],
        "split": args.split,
        "corruption": args.corruption,
        "severity": args.severity,
        "seed": args.seed,
    }
    torch.save(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": int(artifact["labels"].numel()),
                "feature_dim": int(artifact["features"].shape[1]),
                "top1": float((artifact["logits"].argmax(1) == artifact["labels"]).float().mean()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

