"""Cache and evaluate the standard CIFAR-10-C or CIFAR-100-C suite."""

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


def parse_ints(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--corruption-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--severities", type=parse_ints, default=parse_ints("1,2,3,4,5"))
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("2026,2027,2028,2029,2030"))
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--phase-windows", type=int, default=6)
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
    extractor = Path(__file__).with_name("extract_cifar_features.py")
    common = [
        sys.executable,
        extractor,
        "--dataset", args.dataset,
        "--data-root", args.data_root,
        "--checkpoint", args.checkpoint,
        "--split-file", args.split_file,
    ]
    for name, split in (
        ("source_train.pt", "train"),
        ("calibration_train.pt", "calibration"),
        ("test_clean.pt", "test"),
    ):
        output = args.feature_dir / name
        if output.exists() and not args.overwrite:
            print(json.dumps({"skip_existing": str(output)}), flush=True)
        else:
            run(common + ["--output", output, "--split", split])

    conditions = []
    for severity in args.severities:
        for corruption in CORRUPTIONS:
            output = args.feature_dir / f"{corruption}_s{severity}.pt"
            conditions.append((corruption, severity, output))
            if output.exists() and not args.overwrite:
                print(json.dumps({"skip_existing": str(output)}), flush=True)
                continue
            run(
                common
                + [
                    "--output", output,
                    "--split", "corruption",
                    "--corruption-root", args.corruption_root,
                    "--corruption", corruption,
                    "--severity", str(severity),
                ]
            )
    if args.extract_only:
        return

    runner = Path(__file__).with_name("run_cached_experiment.py")
    for severity in args.severities:
        level = [condition for condition in conditions if condition[1] == severity]
        for index in range(0, len(level), 2):
            name_a, _, path_a = level[index]
            name_b, _, path_b = level[(index + 1) % len(level)]
            for seed in args.seeds:
                output = args.result_dir / f"s{severity}_{name_a}__{name_b}_seed{seed}"
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
