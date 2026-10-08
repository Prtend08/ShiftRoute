"""Run feature caching and a multi-seed ImageFolder ShiftRoute suite.

This driver is restartable: completed feature artifacts and result metadata are
left untouched unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable, List


def parse_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--representation-a", default="gaussian_blur")
    parser.add_argument("--representation-b", default="contrast")
    parser.add_argument("--severity-a", type=int, default=3)
    parser.add_argument("--severity-b", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("2026,2027,2028,2029,2030"))
    parser.add_argument("--window-sizes", type=parse_ints, default=parse_ints("128,256,512"))
    parser.add_argument("--imbalance-ratios", type=parse_floats, default=parse_floats("10,50,100"))
    parser.add_argument("--phase-windows", type=int, default=6)
    parser.add_argument("--calibration-windows", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: Iterable[str]) -> None:
    rendered = [str(item) for item in command]
    print(json.dumps({"command": rendered}), flush=True)
    subprocess.run(rendered, check=True)


def main() -> None:
    args = parse_args()
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    feature_specs = [
        ("source_clean.pt", "train", "clean", 1),
        ("calibration_clean.pt", "calibration", "clean", 1),
        ("test_clean.pt", "test", "clean", 1),
        ("test_representation_a.pt", "test", args.representation_a, args.severity_a),
        ("test_representation_b.pt", "test", args.representation_b, args.severity_b),
    ]
    extractor = Path(__file__).with_name("extract_imagefolder_features.py")
    for filename, split, corruption, severity in feature_specs:
        output = args.feature_dir / filename
        if output.exists() and not args.overwrite:
            print(json.dumps({"skip_existing": str(output)}), flush=True)
            continue
        run(
            [
                sys.executable,
                extractor,
                "--data-root",
                args.data_root,
                "--checkpoint",
                args.checkpoint,
                "--split-file",
                args.split_file,
                "--output",
                output,
                "--split",
                split,
                "--corruption",
                corruption,
                "--severity",
                str(severity),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
            ]
        )

    runner = Path(__file__).with_name("run_cached_experiment.py")
    for seed in args.seeds:
        for window_size in args.window_sizes:
            for ratio in args.imbalance_ratios:
                tag = f"seed{seed}_w{window_size}_ir{ratio:g}"
                output_dir = args.result_dir / tag
                if (output_dir / "metadata.json").exists() and not args.overwrite:
                    print(json.dumps({"skip_existing": str(output_dir)}), flush=True)
                    continue
                run(
                    [
                        sys.executable,
                        runner,
                        "--source",
                        args.feature_dir / "source_clean.pt",
                        "--calibration",
                        args.feature_dir / "calibration_clean.pt",
                        "--clean",
                        args.feature_dir / "test_clean.pt",
                        "--representation-a",
                        args.feature_dir / "test_representation_a.pt",
                        "--representation-b",
                        args.feature_dir / "test_representation_b.pt",
                        "--output-dir",
                        output_dir,
                        "--window-size",
                        str(window_size),
                        "--phase-windows",
                        str(args.phase_windows),
                        "--calibration-windows",
                        str(args.calibration_windows),
                        "--imbalance-ratio",
                        str(ratio),
                        "--seed",
                        str(seed),
                    ]
                )


if __name__ == "__main__":
    main()
