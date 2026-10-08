"""Benchmark ImageNet backbone inference and ShiftRoute diagnostic overhead."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shiftroute.diagnostics import diagnose_window, fit_source_statistics


ARCHITECTURES = ("resnet50", "convnext_tiny", "vit_b_16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
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
    elif name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        classifier = model.heads.head
        model.heads.head = nn.Identity()
    else:
        raise ValueError(name)
    return model, classifier, weights.transforms(), str(weights)


def feature_path(root: Path, architecture: str) -> Path:
    folder = {
        "resnet50": "imagenet_resnet50_formal",
        "convnext_tiny": "imagenet_convnext_tiny_formal",
        "vit_b_16": "imagenet_vit_b16_formal",
    }[architecture]
    return root / folder / "source_train.pt"


def timed_cuda(operation) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def cycle(loader: Iterable):
    while True:
        yield from loader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{End-to-end ImageNet-C efficiency on one RTX 4090D. Entries are means over 20 measured windows using real images, FP16 inference, a window of 128 samples, and five warm-up windows; standard deviations are retained in the release CSV. Diagnosis runs after the window predictions.}",
        r"\label{tab:imagenet-gpu-efficiency}",
        r"\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Backbone & Fwd. (ms) & Diag. (ms) & Overhead (\%) & img/s & Peak MiB \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.backbone} & {row.forward_ms:.2f} & "
            f"{row.diagnosis_ms:.2f} & "
            f"{row.overhead_percent:.1f} & {row.throughput_images_s:.1f} & "
            f"{row.peak_gpu_memory_mib:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    positions = np.arange(len(summary))
    axis.bar(positions, summary.forward_ms, label="Backbone forward", color="#4C78A8")
    axis.bar(
        positions, summary.diagnosis_ms, bottom=summary.forward_ms,
        label="ShiftRoute diagnosis", color="#F58518",
    )
    axis.set_xticks(positions, summary.backbone)
    axis.set_ylabel("Latency per 128-image window (ms)")
    axis.set_title("Measured ImageNet-C routing overhead")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_dir / "imagenet_gpu_efficiency.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "imagenet_gpu_efficiency.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.batch_size <= 0 or args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("invalid benchmark counts")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model_records: Dict[str, Dict[str, str]] = {}

    for architecture in ARCHITECTURES:
        model, classifier, transform, weights_name = build_model(architecture)
        model.eval().cuda().to(memory_format=torch.channels_last)
        classifier.eval().cuda()
        source_path = feature_path(args.feature_root, architecture)
        source_artifact = torch.load(source_path, map_location="cpu")
        source = fit_source_statistics(
            source_artifact["features"].float(), source_artifact["labels"].long(),
            num_classes=1000,
        ).to("cuda")
        dataset = datasets.ImageFolder(args.data_root, transform=transform)
        sample_count = min(len(dataset), args.batch_size * max(args.warmup + args.repetitions, 2))
        loader = DataLoader(
            Subset(dataset, list(range(sample_count))), batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True,
            persistent_workers=args.workers > 0, drop_last=True,
        )
        batches = cycle(loader)
        latest: Dict[str, torch.Tensor] = {}

        def forward(images: torch.Tensor) -> None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = model(images)
                latest["features"] = features
                latest["logits"] = classifier(features)

        def diagnose() -> None:
            latest["diagnosis"] = diagnose_window(latest["features"].float(), source)

        with torch.inference_mode():
            for _ in range(args.warmup):
                images, _ = next(batches)
                images = images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
                forward(images)
                diagnose()
            torch.cuda.synchronize()
            for repetition in range(args.repetitions):
                images, _ = next(batches)
                images = images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
                torch.cuda.reset_peak_memory_stats()
                forward_ms = timed_cuda(lambda: forward(images))
                diagnosis_ms = timed_cuda(diagnose)
                peak_mib = torch.cuda.max_memory_allocated() / 2**20
                rows.append(
                    {
                        "architecture": architecture,
                        "backbone": {"resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-Tiny", "vit_b_16": "ViT-B/16"}[architecture],
                        "repetition": repetition,
                        "window_size": args.batch_size,
                        "forward_ms": forward_ms,
                        "diagnosis_ms": diagnosis_ms,
                        "end_to_end_ms": forward_ms + diagnosis_ms,
                        "throughput_images_s": args.batch_size * 1000.0 / (forward_ms + diagnosis_ms),
                        "peak_gpu_memory_mib": peak_mib,
                    }
                )
        model_records[architecture] = {
            "weights": weights_name,
            "source_feature_sha256": sha256(source_path),
        }
        del model, classifier, source, source_artifact, loader, dataset, latest
        torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["architecture", "backbone"], as_index=False)
        .agg(
            forward_ms=("forward_ms", "mean"),
            forward_sd_ms=("forward_ms", "std"),
            diagnosis_ms=("diagnosis_ms", "mean"),
            diagnosis_sd_ms=("diagnosis_ms", "std"),
            throughput_images_s=("throughput_images_s", "mean"),
            peak_gpu_memory_mib=("peak_gpu_memory_mib", "max"),
        )
    )
    summary["overhead_percent"] = 100.0 * summary.diagnosis_ms / summary.forward_ms
    raw.to_csv(args.output_dir / "imagenet_gpu_efficiency_raw.csv", index=False)
    summary.to_csv(args.output_dir / "imagenet_gpu_efficiency_summary.csv", index=False)
    write_latex(summary, args.output_dir / "imagenet_gpu_efficiency.tex")
    write_figure(summary, args.output_dir)
    manifest = {
        "device": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "data_root": str(args.data_root),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "models": model_records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
