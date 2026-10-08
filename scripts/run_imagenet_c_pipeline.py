"""Cache and evaluate the standard 15-corruption ImageNet-C benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import List


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


def ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--corruption-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=["resnet50", "convnext_tiny", "vit_b_16"], default="resnet50"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--severities", type=ints, default=ints("1,2,3,4,5"))
    parser.add_argument("--seeds", type=ints, default=ints("2026,2027,2028"))
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--phase-windows", type=int, default=5)
    parser.add_argument("--imbalance-ratio", type=float, default=50.0)
    parser.add_argument("--prior-strength", type=float, default=0.5)
    parser.add_argument("--prior-momentum", type=float, default=0.5)
    parser.add_argument("--prior-ratio-clip", type=float, default=5.0)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command) -> None:
    rendered = [str(item) for item in command]
    print(json.dumps({"command": rendered}), flush=True)
    subprocess.run(rendered, check=True)


def main() -> None:
    args = parse_args()
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    extractor = Path(__file__).with_name("extract_pretrained_imagenet_features.py")

    clean_specs = [
        ("source_train.pt", args.train_root, 50, 0),
        ("calibration_train.pt", args.train_root, 20, 50),
        ("test_clean.pt", args.val_root, 0, 0),
    ]
    for filename, root, samples_per_class, offset in clean_specs:
        output = args.feature_dir / filename
        if output.exists() and not args.overwrite:
            print(json.dumps({"skip_existing": str(output)}), flush=True)
            continue
        run(
            [
                sys.executable,
                extractor,
                "--data-root", root,
                "--output", output,
                "--architecture", args.architecture,
                "--samples-per-class", str(samples_per_class),
                "--class-offset", str(offset),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
            ]
        )

    condition_paths = []
    for severity in args.severities:
        for corruption in CORRUPTIONS:
            root = args.corruption_root / corruption / str(severity)
            if not root.is_dir():
                raise FileNotFoundError(f"missing ImageNet-C condition: {root}")
            output = args.feature_dir / f"{corruption}_s{severity}.pt"
            condition_paths.append((corruption, severity, output))
            if output.exists() and not args.overwrite:
                print(json.dumps({"skip_existing": str(output)}), flush=True)
                continue
            run(
                [
                    sys.executable,
                    extractor,
                    "--data-root", root,
                    "--output", output,
                    "--architecture", args.architecture,
                    "--batch-size", str(args.batch_size),
                    "--workers", str(args.workers),
                ]
            )

    if args.extract_only:
        return
    runner = Path(__file__).with_name("run_cached_experiment.py")
    for severity in args.severities:
        conditions = [item for item in condition_paths if item[1] == severity]
        pairs = []
        for index in range(0, len(conditions), 2):
            first = conditions[index]
            second = conditions[(index + 1) % len(conditions)]
            pairs.append((first, second))
        for (name_a, _, path_a), (name_b, _, path_b) in pairs:
            for seed in args.seeds:
                tag = f"s{severity}_{name_a}__{name_b}_seed{seed}"
                output = args.result_dir / tag
                if (output / "metadata.json").exists() and not args.overwrite:
                    print(json.dumps({"skip_existing": str(output)}), flush=True)
                    continue
                run(
                    [
                        sys.executable,
                        runner,
                        "--source", args.feature_dir / "source_train.pt",
                        "--calibration", args.feature_dir / "calibration_train.pt",
                        "--clean", args.feature_dir / "test_clean.pt",
                        "--representation-a", path_a,
                        "--representation-b", path_b,
                        "--output-dir", output,
                        "--window-size", str(args.window_size),
                        "--phase-windows", str(args.phase_windows),
                        "--calibration-windows", "300",
                        "--imbalance-ratio", str(args.imbalance_ratio),
                        "--seed", str(seed),
                        "--event-log", "none",
                        "--prior-strength", str(args.prior_strength),
                        "--prior-momentum", str(args.prior_momentum),
                        "--prior-ratio-clip", str(args.prior_ratio_clip),
                    ]
                )


if __name__ == "__main__":
    main()
