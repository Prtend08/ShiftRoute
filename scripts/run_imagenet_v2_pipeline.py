"""Run natural-shift ImageNetV2 streams with correctly aligned numeric labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


VARIANTS = {
    "matched": ("imagenetv2-matched-frequency-format-val", "imagenetv2-matched-frequency"),
    "threshold": ("imagenetv2-threshold0.7-format-val", "imagenetv2-threshold0.7"),
    "top": ("imagenetv2-top-images-format-val", "imagenetv2-top-images"),
}


def ints(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def floats(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=["resnet50", "convnext_tiny", "vit_b_16"], default="resnet50")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seeds", type=ints, default=ints("2026,2027,2028"))
    parser.add_argument("--window-sizes", type=ints, default=ints("512,1024"))
    parser.add_argument("--imbalance-ratios", type=floats, default=floats("10,50,100"))
    parser.add_argument("--phase-windows", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command) -> None:
    rendered = [str(item) for item in command]
    print(json.dumps({"command": rendered}), flush=True)
    subprocess.run(rendered, check=True)


def find_variant(root: Path, candidates) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(f"ImageNetV2 variant not found under {root}: {candidates}")


def main() -> None:
    args = parse_args()
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    extractor = Path(__file__).with_name("extract_pretrained_imagenet_features.py")
    runner = Path(__file__).with_name("run_cached_experiment.py")
    required_formal = [
        args.feature_dir / "source_train.pt",
        args.feature_dir / "development_train.pt",
        args.feature_dir / "conformal_clean.pt",
        args.feature_dir / "test_clean.pt",
        args.feature_dir / "identity_split_manifest.json",
    ]
    missing = [str(path) for path in required_formal if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"ImageNetV2 requires identity-disjoint ImageNet features first: {missing}"
        )
    variant_roots = {name: find_variant(args.v2_root, candidates) for name, candidates in VARIANTS.items()}
    natural_specs = [(f"imagenetv2_{name}.pt", root, 0, 0, True) for name, root in variant_roots.items()]
    for filename, root, samples_per_class, offset, numeric in natural_specs:
        output = args.feature_dir / filename
        if output.exists() and not args.overwrite:
            print(json.dumps({"skip_existing": str(output)}), flush=True)
            continue
        command = [
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
        if numeric:
            command.append("--numeric-class-folders")
        run(command)

    pairs = [("matched", "threshold"), ("matched", "top"), ("threshold", "top")]
    for first, second in pairs:
        for window_size in args.window_sizes:
            for imbalance_ratio in args.imbalance_ratios:
                for seed in args.seeds:
                    tag = f"{first}__{second}_w{window_size}_ir{imbalance_ratio:g}_seed{seed}"
                    output = args.result_dir / tag
                    if (output / "metadata.json").exists() and not args.overwrite:
                        print(json.dumps({"skip_existing": str(output)}), flush=True)
                        continue
                    run(
                        [
                            sys.executable,
                            runner,
                            "--source", args.feature_dir / "source_train.pt",
                            "--development", args.feature_dir / "development_train.pt",
                            "--calibration", args.feature_dir / "conformal_clean.pt",
                            "--clean", args.feature_dir / "test_clean.pt",
                            "--representation-a", args.feature_dir / f"imagenetv2_{first}.pt",
                            "--representation-b", args.feature_dir / f"imagenetv2_{second}.pt",
                            "--output-dir", output,
                            "--window-size", str(window_size),
                            "--phase-windows", str(args.phase_windows),
                            "--calibration-windows", "0",
                            "--calibration-mode", "disjoint_max",
                            "--allow-unpaired-banks",
                            "--imbalance-ratio", str(imbalance_ratio),
                            "--seed", str(seed),
                            "--event-log", "none",
                            "--prior-strength", "0.5",
                            "--prior-momentum", "0.5",
                            "--prior-ratio-clip", "5.0",
                        ]
                    )


if __name__ == "__main__":
    main()
