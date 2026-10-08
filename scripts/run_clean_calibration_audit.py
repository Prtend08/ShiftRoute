"""Audit clean-window familywise alarms without reusing corruption runs.

Each randomization partitions every clean evaluation identity exactly once
into non-overlapping windows. Repeated randomizations deliberately reuse the
same identity pool and are therefore reported as grouping sensitivity, not as
independent replication or as an anytime-valid guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from shiftroute.diagnostics import fit_source_statistics
from shiftroute.experiment import (
    calibrate_from_clean_windows,
    development_prior_reliability_from_clean_windows,
    development_scales_from_clean_windows,
    score_clean_windows,
)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument(
        "--seeds",
        type=parse_ints,
        default=parse_ints(",".join(str(seed) for seed in range(2026, 2046))),
    )
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--development-windows", type=int, default=300)
    parser.add_argument("--development-scale-quantile", type=float, default=0.90)
    parser.add_argument("--prior-reliability-lower-quantile", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)) / denominator
    return float(center - radius), float(center + radius)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load(args.source)
    development = load(args.development)
    calibration = load(args.calibration)
    evaluation = load(args.evaluation)
    calibration_indices = set(calibration.get("indices", torch.arange(len(calibration["labels"]))).tolist())
    evaluation_indices = set(evaluation.get("indices", torch.arange(len(evaluation["labels"]))).tolist())
    overlap = calibration_indices.intersection(evaluation_indices)
    if overlap:
        raise ValueError(f"calibration/evaluation identity overlap: {len(overlap)}")
    available_windows = len(evaluation["labels"]) // args.window_size
    if available_windows < 1:
        raise ValueError("evaluation split cannot form one complete window")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    source_statistics = fit_source_statistics(
        source["features"].float().to(device),
        source["labels"].long().to(device),
        kappa=30.0,
    )
    window_records: list[dict[str, object]] = []
    partition_records: list[dict[str, object]] = []
    for seed in args.seeds:
        prior_scale, representation_scale = development_scales_from_clean_windows(
            development["features"].float(),
            development["labels"].long(),
            source_statistics,
            window_size=args.window_size,
            windows=args.development_windows,
            quantile=args.development_scale_quantile,
            seed=seed,
            device=device,
        )
        reliability_threshold = development_prior_reliability_from_clean_windows(
            development["features"].float(),
            development["labels"].long(),
            source_statistics,
            window_size=args.window_size,
            windows=args.development_windows,
            lower_quantile=args.prior_reliability_lower_quantile,
            seed=seed,
            device=device,
        )
        thresholds = calibrate_from_clean_windows(
            calibration["features"].float(),
            calibration["labels"].long(),
            source_statistics,
            window_size=args.window_size,
            windows=None,
            familywise_alpha=args.familywise_alpha,
            seed=seed,
            device=device,
            sampling="disjoint",
            prior_scale=prior_scale,
            representation_scale=representation_scale,
        )
        prior, representation, confidence = score_clean_windows(
            evaluation["features"].float(),
            evaluation["labels"].long(),
            source_statistics,
            window_size=args.window_size,
            windows=None,
            seed=seed + 100_000,
            device=device,
            sampling="disjoint",
            return_assignment_confidence=True,
        )
        alarms = []
        for window, (prior_score, representation_score, assignment_confidence) in enumerate(
            zip(prior, representation, confidence)
        ):
            prior_alarm = prior_score > thresholds.prior_enter
            representation_alarm = representation_score > thresholds.representation_enter
            alarm = prior_alarm or representation_alarm
            alarms.append(alarm)
            window_records.append(
                {
                    "seed": seed,
                    "window": window,
                    "prior_score": prior_score,
                    "representation_score": representation_score,
                    "prior_threshold": thresholds.prior_enter,
                    "representation_threshold": thresholds.representation_enter,
                    "prior_alarm": prior_alarm,
                    "representation_alarm": representation_alarm,
                    "familywise_alarm": alarm,
                    "assignment_confidence": assignment_confidence,
                    "prior_reliable": assignment_confidence >= reliability_threshold,
                }
            )
        successes = int(np.sum(alarms))
        lower, upper = wilson(successes, len(alarms))
        partition_records.append(
            {
                "seed": seed,
                "identity_disjoint_windows": len(alarms),
                "alarms": successes,
                "alarm_rate": successes / len(alarms),
                "wilson95_lower_descriptive": lower,
                "wilson95_upper_descriptive": upper,
                "calibration_windows": thresholds.calibration_windows,
                "prior_threshold": thresholds.prior_enter,
                "representation_threshold": thresholds.representation_enter,
                "prior_reliability_threshold": reliability_threshold,
            }
        )
    windows = pd.DataFrame(window_records)
    partitions = pd.DataFrame(partition_records)
    windows.to_csv(args.output_dir / "clean_window_audit.csv", index=False)
    partitions.to_csv(args.output_dir / "partition_sensitivity.csv", index=False)
    rates = partitions.alarm_rate.to_numpy(float)
    primary = partitions.iloc[0]
    summary = {
        "nominal_familywise_alpha": args.familywise_alpha,
        "primary_predeclared_seed": int(primary.seed),
        "primary_identity_disjoint_windows": int(primary.identity_disjoint_windows),
        "primary_alarms": int(primary.alarms),
        "primary_alarm_rate": float(primary.alarm_rate),
        "primary_wilson95_descriptive": [
            float(primary.wilson95_lower_descriptive),
            float(primary.wilson95_upper_descriptive),
        ],
        "partition_sensitivity_randomizations": len(partitions),
        "partition_sensitivity_rate_median": float(np.median(rates)),
        "partition_sensitivity_rate_min": float(np.min(rates)),
        "partition_sensitivity_rate_max": float(np.max(rates)),
        "calibration_evaluation_identity_overlap": len(overlap),
        "evaluation_identities": len(evaluation_indices),
        "unused_evaluation_identities_per_partition": len(evaluation_indices) - available_windows * args.window_size,
        "interpretation": (
            "The primary partition contains identity-disjoint windows. Repeated random partitions "
            "reuse the same evaluation identities and quantify grouping sensitivity only; they are "
            "not independent replications and do not imply anytime-valid control."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_sha256": {
            "source": sha256(args.source),
            "development": sha256(args.development),
            "calibration": sha256(args.calibration),
            "evaluation": sha256(args.evaluation),
        },
        "output_sha256": {
            "clean_window_audit": sha256(args.output_dir / "clean_window_audit.csv"),
            "partition_sensitivity": sha256(args.output_dir / "partition_sensitivity.csv"),
            "summary": sha256(args.output_dir / "summary.json"),
        },
        "labels_used_for_controller": False,
        "identity_policy": "calibration/evaluation disjoint; windows disjoint within each partition",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
