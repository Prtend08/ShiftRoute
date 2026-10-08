"""Train reproducible CIFAR-10/100 source models for corruption experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_class(name: str):
    return datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100


def stratified_train_calibration(targets: List[int], seed: int, calibration_per_class: int = 50) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    train_indices: List[int] = []
    calibration_indices: List[int] = []
    for class_index in np.unique(targets_array):
        indices = np.flatnonzero(targets_array == class_index)
        rng.shuffle(indices)
        calibration_indices.extend(indices[:calibration_per_class].tolist())
        train_indices.extend(indices[calibration_per_class:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(calibration_indices)
    return {"train": train_indices, "calibration": calibration_indices}


def build_model(classes: int) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    examples = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
        correct += int((logits.argmax(1) == labels).sum())
        examples += labels.numel()
        loss_sum += float(loss) * labels.numel()
    return {"top1": correct / examples, "loss": loss_sum / examples}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = str(args.data_root)
    dataset_type = dataset_class(args.dataset)
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose([transforms.ToTensor(), normalization])
    index_dataset = dataset_type(root, train=True, download=True)
    split_path = args.output_dir / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        split = stratified_train_calibration(index_dataset.targets, args.seed)
        split_path.write_text(json.dumps(split), encoding="utf-8")
    train_data = Subset(dataset_type(root, train=True, transform=train_transform), split["train"])
    calibration_data = Subset(dataset_type(root, train=True, transform=evaluation_transform), split["calibration"])
    test_data = dataset_type(root, train=False, transform=evaluation_transform, download=True)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_args)
    calibration_loader = DataLoader(calibration_data, shuffle=False, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)

    classes = 10 if args.dataset == "cifar10" else 100
    device = torch.device("cuda")
    model = build_model(classes).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 0
    best_top1 = -1.0
    latest = args.output_dir / "latest.pt"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_top1 = state["best_top1"]

    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
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
            correct += int((logits.argmax(1) == labels).sum())
            loss_sum += float(loss) * labels.numel()
        scheduler.step()
        calibration = evaluate(model, calibration_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / examples,
            "train_top1": correct / examples,
            "calibration_loss": calibration["loss"],
            "calibration_top1": calibration["top1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        state = {
            "epoch": epoch,
            "best_top1": max(best_top1, calibration["top1"]),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "dataset": args.dataset,
            "classes": classes,
        }
        torch.save(state, latest)
        if calibration["top1"] > best_top1:
            best_top1 = calibration["top1"]
            torch.save(state, args.output_dir / "best.pt")

    test = evaluate(model, test_loader, device)
    (args.output_dir / "test.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    print(json.dumps({"test": test}), flush=True)


if __name__ == "__main__":
    main()

