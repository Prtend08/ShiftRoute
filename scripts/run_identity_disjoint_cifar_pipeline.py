"""Run the formal cached CIFAR-C protocol on identity-disjoint feature banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_cifar_c_pipeline import CORRUPTIONS


def parse_ints(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--severities", type=parse_ints, default=parse_ints("1,2,3,4,5"))
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("2026,2027,2028"))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--phase-windows", type=int, default=6)
    parser.add_argument("--imbalance-ratio", type=float, default=50.0)
    parser.add_argument("--prior-strength", type=float, default=0.5)
    parser.add_argument("--prior-momentum", type=float, default=0.5)
    parser.add_argument("--prior-ratio-clip", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    required = [
        args.feature_dir / "source_train.pt",
        args.feature_dir / "development_train.pt",
        args.feature_dir / "conformal_clean.pt",
        args.feature_dir / "test_clean.pt",
        args.feature_dir / "identity_split_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"formal identity-disjoint features are incomplete: {missing}")

    runner = Path(__file__).with_name("run_cached_experiment.py")
    for severity in args.severities:
        level = [
            (name, args.feature_dir / f"{name}_s{severity}.pt")
            for name in CORRUPTIONS
        ]
        for index in range(0, len(level), 2):
            name_a, path_a = level[index]
            name_b, path_b = level[(index + 1) % len(level)]
            for seed in args.seeds:
                output = args.result_dir / f"s{severity}_{name_a}__{name_b}_seed{seed}"
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
                    "--representation-a", path_a,
                    "--representation-b", path_b,
                    "--output-dir", output,
                    "--window-size", str(args.window_size),
                    "--phase-windows", str(args.phase_windows),
                    "--calibration-windows", "0",
                    "--calibration-mode", "disjoint_max",
                    "--familywise-alpha", "0.05",
                    "--imbalance-ratio", str(args.imbalance_ratio),
                    "--seed", str(seed),
                    "--event-log", "none",
                    "--prior-strength", str(args.prior_strength),
                    "--prior-momentum", str(args.prior_momentum),
                    "--prior-ratio-clip", str(args.prior_ratio_clip),
                ]
                print(json.dumps({"command": [str(item) for item in command]}), flush=True)
                subprocess.run([str(item) for item in command], check=True)


if __name__ == "__main__":
    main()
