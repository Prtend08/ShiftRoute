"""Run a restartable component ablation suite from fixed feature artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def ints(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("source", "development", "calibration", "clean", "representation_a", "representation_b"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=ints, default=ints("2026,2027,2028,2029,2030"))
    parser.add_argument("--window-sizes", type=ints, default=ints("128,512"))
    parser.add_argument("--imbalance-ratio", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_cached_experiment.py")
    variants = {
        "full": [],
        "no_prior_axis": ["--disable-prior-axis"],
        "no_representation_axis": ["--disable-representation-axis"],
        "no_prior_reliability_guard": ["--disable-prior-reliability-guard"],
        "bootstrap_bonferroni_calibration": [
            "--calibration-mode", "bootstrap_bonferroni", "--calibration-windows", "300"
        ],
        "no_prior_projection": ["--no-prior-projection"],
        "no_between_class_variance": ["--no-between-class-variance"],
        "no_confirmation": ["--confirmation-windows", "1"],
        "no_minimum_dwell": ["--min-dwell-windows", "0"],
        "stateless_router": ["--confirmation-windows", "1", "--min-dwell-windows", "0"],
        "no_temporal_prior_shrinkage": ["--prior-momentum", "0"],
        "full_unbounded_prior": [
            "--prior-strength", "1", "--prior-momentum", "0", "--prior-ratio-clip", "1000000000"
        ],
        "no_prior_ratio_clip": ["--prior-ratio-clip", "1000000000"],
        "no_source_anchor": ["--source-anchor", "0"],
        "no_prototype_confidence_gate": ["--prototype-confidence", "0"],
        "no_prototype_momentum": ["--prototype-momentum", "0"],
    }
    common = [
        sys.executable,
        runner,
        "--source", args.source,
        "--development", args.development,
        "--calibration", args.calibration,
        "--clean", args.clean,
        "--representation-a", args.representation_a,
        "--representation-b", args.representation_b,
        "--calibration-windows", "0",
        "--calibration-mode", "disjoint_max",
        "--phase-windows", "6",
        "--imbalance-ratio", str(args.imbalance_ratio),
        "--event-log", "none",
    ]
    for variant, extra in variants.items():
        for window_size in args.window_sizes:
            for seed in args.seeds:
                output = args.output_dir / variant / f"w{window_size}_seed{seed}"
                if (output / "metadata.json").exists():
                    print(json.dumps({"skip_existing": str(output)}), flush=True)
                    continue
                command = common + [
                    "--output-dir", output,
                    "--window-size", str(window_size),
                    "--seed", str(seed),
                ] + extra
                print(json.dumps({"command": [str(item) for item in command]}), flush=True)
                subprocess.run([str(item) for item in command], check=True)


if __name__ == "__main__":
    main()
