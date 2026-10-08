"""Run a strict two-condition controlled-shift suite from formal feature banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def ints(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def floats(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--representation-a", default="test_representation_a.pt")
    parser.add_argument("--representation-b", default="test_representation_b.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-sizes", type=ints, default=ints("128,256,512"))
    parser.add_argument("--imbalance-ratios", type=floats, default=floats("10,50,100"))
    parser.add_argument("--seeds", type=ints, default=ints("2026,2027,2028,2029,2030"))
    parser.add_argument("--phase-windows", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = [
        args.feature_dir / "source_train.pt",
        args.feature_dir / "development_train.pt",
        args.feature_dir / "conformal_clean.pt",
        args.feature_dir / "test_clean.pt",
        args.feature_dir / args.representation_a,
        args.feature_dir / args.representation_b,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"formal feature suite is incomplete: {missing}")
    runner = Path(__file__).with_name("run_cached_experiment.py")
    for window_size in args.window_sizes:
        for imbalance_ratio in args.imbalance_ratios:
            for seed in args.seeds:
                output = args.output_dir / (
                    f"w{window_size}_ir{imbalance_ratio:g}_seed{seed}"
                )
                if (output / "metadata.json").exists() and not args.overwrite:
                    print(json.dumps({"skip_existing": str(output)}), flush=True)
                    continue
                command = [
                    sys.executable,
                    runner,
                    "--source", args.feature_dir / "source_train.pt",
                    "--development", args.feature_dir / "development_train.pt",
                    "--calibration", args.feature_dir / "conformal_clean.pt",
                    "--clean", args.feature_dir / "test_clean.pt",
                    "--representation-a", args.feature_dir / args.representation_a,
                    "--representation-b", args.feature_dir / args.representation_b,
                    "--output-dir", output,
                    "--window-size", str(window_size),
                    "--phase-windows", str(args.phase_windows),
                    "--calibration-windows", "0",
                    "--calibration-mode", "disjoint_max",
                    "--familywise-alpha", "0.05",
                    "--imbalance-ratio", str(imbalance_ratio),
                    "--seed", str(seed),
                    "--event-log", "none",
                    "--prior-strength", "0.5",
                    "--prior-momentum", "0.5",
                    "--prior-ratio-clip", "5.0",
                ]
                print(json.dumps({"command": [str(item) for item in command]}), flush=True)
                subprocess.run([str(item) for item in command], check=True)


if __name__ == "__main__":
    main()
