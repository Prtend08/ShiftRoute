"""Cache ImageNet/ImageNet-C features from a torchvision pretrained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from tqdm import tqdm


ARCHITECTURES = ("resnet50", "convnext_tiny", "vit_b_16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="resnet50")
    parser.add_argument("--samples-per-class", type=int, default=0)
    parser.add_argument("--class-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--numeric-class-folders", action="store_true")
    return parser.parse_args()


def build_model(name: str) -> Tuple[nn.Module, nn.Linear, object, str]:
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        classifier = model.fc
        model.fc = nn.Identity()
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        classifier = model.classifier[-1]
        model.classifier[-1] = nn.Identity()
    else:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        classifier = model.heads.head
        model.heads.head = nn.Identity()
    return model, classifier, weights.transforms(), str(weights)


def restore_numeric_folder_labels(dataset: datasets.ImageFolder) -> None:
    """Map ImageNetV2 folders 0..999 to their integer classifier indices."""
    numeric_classes = sorted(int(name) for name in dataset.classes)
    if numeric_classes != list(range(len(numeric_classes))):
        raise ValueError("numeric class folders must form the contiguous range 0..K-1")
    samples = [(path, int(Path(path).parent.name)) for path, _ in dataset.samples]
    dataset.classes = [str(index) for index in numeric_classes]
    dataset.class_to_idx = {name: int(name) for name in dataset.classes}
    dataset.samples = samples
    dataset.imgs = samples
    dataset.targets = [target for _, target in samples]


def stratified_indices(
    targets: Sequence[int], samples_per_class: int, offset: int, seed: int
) -> List[int]:
    if samples_per_class <= 0:
        return list(range(len(targets)))
    target_array = np.asarray(targets)
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for class_index in np.unique(target_array):
        candidates = np.flatnonzero(target_array == class_index)
        rng.shuffle(candidates)
        stop = offset + samples_per_class
        if stop > len(candidates):
            raise ValueError(
                f"class {class_index} has {len(candidates)} samples, fewer than {stop} requested"
            )
        selected.extend(candidates[offset:stop].tolist())
    selected.sort()
    return selected


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    model, classifier, preprocessing, weights_name = build_model(args.architecture)
    model.eval().cuda().to(memory_format=torch.channels_last)
    classifier.eval().cuda()
    dataset = datasets.ImageFolder(args.data_root, transform=preprocessing)
    if args.numeric_class_folders:
        restore_numeric_folder_labels(dataset)
    indices = stratified_indices(
        dataset.targets, args.samples_per_class, args.class_offset, args.seed
    )
    loader = DataLoader(
        Subset(dataset, indices),
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
        for images, labels in tqdm(loader, desc=f"{args.architecture}:{args.data_root.name}"):
            images = images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = model(images)
                logits = classifier(features)
            feature_chunks.append(features.float().cpu().half())
            logit_chunks.append(logits.float().cpu().half())
            label_chunks.append(labels.long())

    features = torch.cat(feature_chunks)
    logits = torch.cat(logit_chunks)
    labels = torch.cat(label_chunks)
    artifact = {
        "features": features,
        "logits": logits,
        "labels": labels,
        "indices": torch.as_tensor(indices, dtype=torch.long),
        "classes": dataset.classes,
        "architecture": args.architecture,
        "weights": weights_name,
        "data_root": str(args.data_root),
        "samples_per_class": args.samples_per_class,
        "class_offset": args.class_offset,
        "seed": args.seed,
        "numeric_class_folders": args.numeric_class_folders,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    top1 = float((logits.argmax(1) == labels).float().mean())
    top5 = float((logits.topk(5, dim=1).indices == labels[:, None]).any(1).float().mean())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": int(labels.numel()),
                "feature_dim": int(features.shape[1]),
                "top1": top1,
                "top5": top5,
                "weights": weights_name,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
