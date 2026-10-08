"""Fine-tune an ImageNet-pretrained classifier on a stratified ImageFolder split."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(
    targets: Sequence[int],
    train_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> Dict[str, List[int]]:
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("train and calibration fractions must sum to less than one")
    generator = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    output: Dict[str, List[int]] = {"train": [], "calibration": [], "test": []}
    for class_index in np.unique(targets_array):
        indices = np.flatnonzero(targets_array == class_index)
        generator.shuffle(indices)
        train_end = int(round(len(indices) * train_fraction))
        calibration_end = train_end + int(round(len(indices) * calibration_fraction))
        output["train"].extend(indices[:train_end].tolist())
        output["calibration"].extend(indices[train_end:calibration_end].tolist())
        output["test"].extend(indices[calibration_end:].tolist())
    for key in output:
        generator.shuffle(output[key])
    return output


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def build_model(architecture: str, classes: int) -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, classes)
    nn.init.trunc_normal_(model.fc.weight, std=0.02)
    nn.init.zeros_(model.fc.bias)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct1 = 0
    correct5 = 0
    examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, labels)
        examples += labels.numel()
        loss_sum += float(loss.item()) * labels.numel()
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        correct1 += int((top5[:, 0] == labels).sum().item())
        correct5 += int((top5 == labels[:, None]).any(dim=1).sum().item())
    return {
        "loss": loss_sum / max(examples, 1),
        "top1": correct1 / max(examples, 1),
        "top5": correct5 / max(examples, 1),
        "examples": examples,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    train_transform, evaluation_transform = build_transforms()
    index_dataset = datasets.ImageFolder(args.data_root)
    split_path = args.output_dir / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        split = stratified_split(
            index_dataset.targets,
            args.train_fraction,
            args.calibration_fraction,
            args.seed,
        )
        split_path.write_text(json.dumps(split), encoding="utf-8")

    train_dataset = Subset(datasets.ImageFolder(args.data_root, transform=train_transform), split["train"])
    calibration_dataset = Subset(
        datasets.ImageFolder(args.data_root, transform=evaluation_transform),
        split["calibration"],
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_kwargs)

    classes = len(index_dataset.classes)
    device = torch.device("cuda")
    model = build_model(args.architecture, classes).to(device)
    model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    best_top1 = -1.0
    latest_path = args.output_dir / "latest.pt"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_top1 = float(checkpoint["best_top1"])

    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        started = time.time()
        loss_sum = 0.0
        correct = 0
        examples = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            examples += labels.numel()
            loss_sum += float(loss.item()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
        scheduler.step()
        validation = evaluate(model, calibration_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(examples, 1),
            "train_top1": correct / max(examples, 1),
            "calibration_loss": validation["loss"],
            "calibration_top1": validation["top1"],
            "calibration_top5": validation["top5"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        state = {
            "epoch": epoch,
            "best_top1": max(best_top1, validation["top1"]),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "architecture": args.architecture,
            "classes": index_dataset.classes,
            "class_to_idx": index_dataset.class_to_idx,
        }
        torch.save(state, latest_path)
        if validation["top1"] > best_top1:
            best_top1 = validation["top1"]
            torch.save(state, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()

