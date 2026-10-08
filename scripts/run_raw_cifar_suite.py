"""Run full or selected raw-image causal TTA routing comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def ints(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--corruption-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--seeds", type=ints, default=ints("2026,2027,2028"))
    parser.add_argument("--protocol", choices=["full", "selected"], default="full")
    parser.add_argument("--severities", type=ints, default=ints("3,5"))
    parser.add_argument("--method-set", choices=["full", "ods_router"], default="full")
    parser.add_argument("--ods-knn", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_raw_cifar_tta.py")
    if args.protocol == "selected":
        pairs = [
            ("gaussian_noise", 3, "contrast", 3),
            ("motion_blur", 5, "jpeg_compression", 5),
            ("snow", 5, "impulse_noise", 5),
        ]
    else:
        pairs = []
        for severity in args.severities:
            for index in range(0, len(CORRUPTIONS), 2):
                first = CORRUPTIONS[index]
                second = CORRUPTIONS[(index + 1) % len(CORRUPTIONS)]
                pairs.append((first, severity, second, severity))
    for name_a, severity_a, name_b, severity_b in pairs:
        for seed in args.seeds:
            output = args.output_dir / (
                f"{name_a}_s{severity_a}__{name_b}_s{severity_b}_seed{seed}"
            )
            if (output / "metadata.json").exists():
                print(json.dumps({"skip_existing": str(output)}), flush=True)
                continue
            command = [
                sys.executable,
                runner,
                "--dataset", args.dataset,
                "--data-root", args.data_root,
                "--corruption-root", args.corruption_root,
                "--checkpoint", args.checkpoint,
                "--source", args.feature_dir / "source_train.pt",
                "--development", args.feature_dir / "development_train.pt",
                "--calibration", args.feature_dir / "conformal_clean.pt",
                "--clean", args.feature_dir / "test_clean.pt",
                "--representation-a", args.feature_dir / f"{name_a}_s{severity_a}.pt",
                "--representation-b", args.feature_dir / f"{name_b}_s{severity_b}.pt",
                "--output-dir", output,
                "--window-size", str(args.window_size),
                "--phase-windows", "6",
                "--imbalance-ratio", "50",
                "--seed", str(seed),
                "--method-set", args.method_set,
                "--ods-knn", str(args.ods_knn if args.ods_knn is not None else (5 if args.dataset == "cifar10" else 2)),
            ]
            print(json.dumps({"command": [str(item) for item in command]}), flush=True)
            subprocess.run([str(item) for item in command], check=True)


if __name__ == "__main__":
    main()
