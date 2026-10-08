"""Cache clean CIFAR or official CIFAR-C feature/logit banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from train_cifar import build_model, dataset_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "calibration", "test", "corruption"], required=True)
    parser.add_argument("--corruption-root", type=Path)
    parser.add_argument("--corruption")
    parser.add_argument("--severity", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


class ArrayImages(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(int(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model"])
    classifier = model.fc
    model.fc = nn.Identity()
    model.eval().cuda().to(memory_format=torch.channels_last)
    classifier.eval().cuda()
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transform = transforms.Compose([transforms.ToTensor(), normalization])

    if args.split == "corruption":
        if args.corruption_root is None or not args.corruption:
            raise ValueError("corruption extraction needs a root and corruption name")
        images = np.load(args.corruption_root / f"{args.corruption}.npy", mmap_mode="r")
        labels = np.load(args.corruption_root / "labels.npy")
        examples_per_severity = images.shape[0] // 5
        start = (args.severity - 1) * examples_per_severity
        images = images[start : start + examples_per_severity]
        if labels.shape[0] == 5 * examples_per_severity:
            labels = labels[start : start + examples_per_severity]
        else:
            labels = labels[:examples_per_severity]
        dataset = ArrayImages(images, labels, transform)
        indices = torch.arange(len(labels))
    else:
        data_type = dataset_class(args.dataset)
        train = args.split != "test"
        full = data_type(str(args.data_root), train=train, transform=transform, download=True)
        if args.split == "test":
            dataset = full
            indices = torch.arange(len(full))
        else:
            split = json.loads(args.split_file.read_text(encoding="utf-8"))
            indices = torch.as_tensor(split[args.split], dtype=torch.long)
            dataset = Subset(full, indices.tolist())

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    features = []
    logits = []
    labels = []
    with torch.inference_mode():
        for images, batch_labels in tqdm(loader):
            images = images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                batch_features = model(images)
                batch_logits = classifier(batch_features)
            features.append(batch_features.cpu().half())
            logits.append(batch_logits.cpu().half())
            labels.append(batch_labels.long())
    artifact = {
        "features": torch.cat(features),
        "logits": torch.cat(logits),
        "labels": torch.cat(labels),
        "indices": indices,
        "dataset": args.dataset,
        "split": args.split,
        "corruption": args.corruption,
        "severity": args.severity,
        "architecture": "cifar_resnet18",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(artifact["labels"]),
                "top1": float((artifact["logits"].argmax(1) == artifact["labels"]).float().mean()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
