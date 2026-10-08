"""Run the auditable prequential experiment from cached feature banks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from shiftroute.adapters import PriorAdapterConfig, PrototypeAdapterConfig
from shiftroute.diagnostics import fit_source_statistics
from shiftroute.experiment import (
    FeatureBank,
    StreamPhase,
    build_stream,
    calibrate_from_clean_windows,
    development_prior_reliability_from_clean_windows,
    development_scales_from_clean_windows,
    long_tailed_prior,
    run_prequential_experiment,
)
from shiftroute.router import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--representation-a", type=Path, required=True)
    parser.add_argument("--representation-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--phase-windows", type=int, default=5)
    parser.add_argument("--calibration-windows", type=int, default=0)
    parser.add_argument("--development-windows", type=int, default=300)
    parser.add_argument("--development-scale-quantile", type=float, default=0.90)
    parser.add_argument("--prior-reliability-lower-quantile", type=float, default=0.05)
    parser.add_argument(
        "--calibration-mode",
        choices=["disjoint_max", "bootstrap_bonferroni"],
        default="disjoint_max",
    )
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--imbalance-ratio", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--event-log", choices=["sample", "none"], default="sample")
    parser.add_argument("--prior-strength", type=float, default=0.5)
    parser.add_argument("--prior-momentum", type=float, default=0.5)
    parser.add_argument("--prior-ratio-clip", type=float, default=5.0)
    parser.add_argument(
        "--stream-protocol", choices=["full", "prior_development"], default="full"
    )
    parser.add_argument("--no-prior-projection", action="store_true")
    parser.add_argument("--no-between-class-variance", action="store_true")
    parser.add_argument("--confirmation-windows", type=int, default=2)
    parser.add_argument("--min-dwell-windows", type=int, default=2)
    parser.add_argument("--minimum-assignment-confidence", type=float, default=0.0)
    parser.add_argument("--prototype-momentum", type=float, default=0.9)
    parser.add_argument("--source-anchor", type=float, default=0.25)
    parser.add_argument("--prototype-confidence", type=float, default=0.8)
    parser.add_argument("--disable-prior-axis", action="store_true")
    parser.add_argument("--disable-representation-axis", action="store_true")
    parser.add_argument("--disable-prior-reliability-guard", action="store_true")
    parser.add_argument("--allow-unpaired-banks", action="store_true")
    return parser.parse_args()


def load_artifact(path: Path, name: str) -> FeatureBank:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return FeatureBank(
        name=name,
        features=artifact["features"].float(),
        logits=artifact["logits"].float(),
        labels=artifact["labels"].long(),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_artifact = torch.load(args.source, map_location="cpu", weights_only=False)
    development_artifact = (
        torch.load(args.development, map_location="cpu", weights_only=False)
        if args.development is not None
        else None
    )
    calibration_artifact = torch.load(args.calibration, map_location="cpu", weights_only=False)
    clean = load_artifact(args.clean, "clean")
    representation_a = load_artifact(args.representation_a, "representation_a")
    representation_b = load_artifact(args.representation_b, "representation_b")
    paired = torch.equal(clean.labels, representation_a.labels) and torch.equal(
        clean.labels, representation_b.labels
    )
    if not paired and not args.allow_unpaired_banks:
        raise ValueError("all test feature banks must use the same ordered examples")
    if args.allow_unpaired_banks:
        expected_classes = clean.classes
        for bank in (clean, representation_a, representation_b):
            present = torch.unique(bank.labels).numel()
            if bank.classes != expected_classes or present != expected_classes:
                raise ValueError("each unpaired feature bank must contain every class")

    classes = clean.classes
    uniform = torch.full((classes,), 1.0 / classes)
    forward = long_tailed_prior(classes, args.imbalance_ratio)
    backward = long_tailed_prior(classes, args.imbalance_ratio, reverse=True)
    if args.stream_protocol == "prior_development":
        phases = [
            StreamPhase("clean_start", "clean", args.phase_windows, Action.FREEZE, uniform),
            StreamPhase("prior_forward", "clean", args.phase_windows, Action.PRIOR_ONLY, forward),
            StreamPhase("clean_middle", "clean", args.phase_windows, Action.FREEZE, uniform),
            StreamPhase("prior_backward", "clean", args.phase_windows, Action.PRIOR_ONLY, backward),
            StreamPhase("clean_end", "clean", args.phase_windows, Action.FREEZE, uniform),
        ]
    else:
        phases = [
            StreamPhase("clean_start", "clean", args.phase_windows, Action.FREEZE, uniform),
            StreamPhase("prior_forward", "clean", args.phase_windows, Action.PRIOR_ONLY, forward),
            StreamPhase("clean_recovery", "clean", args.phase_windows, Action.FREEZE, uniform),
            StreamPhase(
                "representation_a",
                "representation_a",
                args.phase_windows,
                Action.REPRESENTATION_ONLY,
                uniform,
            ),
            StreamPhase("compound_forward", "representation_a", args.phase_windows, Action.JOINT, forward),
            StreamPhase(
                "representation_b",
                "representation_b",
                args.phase_windows,
                Action.REPRESENTATION_ONLY,
                uniform,
            ),
            StreamPhase("compound_backward", "representation_b", args.phase_windows, Action.JOINT, backward),
            StreamPhase("clean_end", "clean", args.phase_windows, Action.FREEZE, uniform),
        ]
    banks = {
        "clean": clean,
        "representation_a": representation_a,
        "representation_b": representation_b,
    }
    windows = build_stream(banks, phases, window_size=args.window_size, seed=args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    source_stats = fit_source_statistics(
        source_artifact["features"].float().to(device),
        source_artifact["labels"].long().to(device),
        kappa=30.0,
    )
    calibration_windows = args.calibration_windows or None
    if args.calibration_mode == "disjoint_max":
        if development_artifact is None:
            raise ValueError("disjoint_max calibration requires --development")
        prior_scale, representation_scale = development_scales_from_clean_windows(
            development_artifact["features"].float(),
            development_artifact["labels"].long(),
            source_stats,
            window_size=args.window_size,
            windows=args.development_windows,
            quantile=args.development_scale_quantile,
            seed=args.seed,
            device=device,
            prior_project_representation=not args.no_prior_projection,
            include_between_class_variance=not args.no_between_class_variance,
        )
        prior_reliability_threshold = development_prior_reliability_from_clean_windows(
            development_artifact["features"].float(),
            development_artifact["labels"].long(),
            source_stats,
            window_size=args.window_size,
            windows=args.development_windows,
            lower_quantile=args.prior_reliability_lower_quantile,
            seed=args.seed,
            device=device,
            prior_project_representation=not args.no_prior_projection,
            include_between_class_variance=not args.no_between_class_variance,
        )
        thresholds = calibrate_from_clean_windows(
            calibration_artifact["features"].float(),
            calibration_artifact["labels"].long(),
            source_stats,
            window_size=args.window_size,
            windows=calibration_windows,
            familywise_alpha=args.familywise_alpha,
            seed=args.seed,
            device=device,
            prior_project_representation=not args.no_prior_projection,
            include_between_class_variance=not args.no_between_class_variance,
            sampling="disjoint",
            prior_scale=prior_scale,
            representation_scale=representation_scale,
        )
    else:
        prior_reliability_threshold = 0.0
        thresholds = calibrate_from_clean_windows(
            calibration_artifact["features"].float(),
            calibration_artifact["labels"].long(),
            source_stats,
            window_size=args.window_size,
            windows=calibration_windows,
            familywise_alpha=args.familywise_alpha,
            seed=args.seed,
            device=device,
            prior_project_representation=not args.no_prior_projection,
            include_between_class_variance=not args.no_between_class_variance,
            sampling="bootstrap",
        )
    if args.disable_prior_axis:
        thresholds = replace(thresholds, prior_enter=float("inf"))
    if args.disable_representation_axis:
        thresholds = replace(thresholds, representation_enter=float("inf"))
    if args.disable_prior_reliability_guard:
        prior_reliability_threshold = 0.0
    window_rows, event_rows, _ = run_prequential_experiment(
        source_artifact["features"].float(),
        source_artifact["labels"].long(),
        calibration_artifact["features"].float(),
        calibration_artifact["labels"].long(),
        windows,
        thresholds,
        adapter_config=PrototypeAdapterConfig(
            temperature=0.07,
            momentum=args.prototype_momentum,
            source_anchor=args.source_anchor,
            confidence_threshold=args.prototype_confidence,
            minimum_class_mass=1.0,
        ),
        prior_config=PriorAdapterConfig(
            strength=args.prior_strength,
            momentum=args.prior_momentum,
            ratio_clip=args.prior_ratio_clip,
        ),
        device=device,
        log_sample_events=args.event_log == "sample",
        prior_project_representation=not args.no_prior_projection,
        include_between_class_variance=not args.no_between_class_variance,
        confirmation_windows=args.confirmation_windows,
        min_dwell_windows=args.min_dwell_windows,
        minimum_assignment_confidence=args.minimum_assignment_confidence,
        prior_reliability_threshold=prior_reliability_threshold,
    )

    window_frame = pd.DataFrame(window_rows)
    event_frame = pd.DataFrame(event_rows)
    source_accuracy = (
        window_frame[window_frame.method == "source"]
        .set_index("window")["accuracy"]
        .rename("source_accuracy")
    )
    window_frame = window_frame.join(source_accuracy, on="window")
    window_frame["gain_over_source"] = window_frame.accuracy - window_frame.source_accuracy

    window_path = args.output_dir / "window_metrics.csv"
    window_frame.to_csv(window_path, index=False)
    event_path = args.output_dir / "events.csv.gz"
    if args.event_log == "sample":
        with gzip.open(event_path, "wt", encoding="utf-8", newline="") as handle:
            event_frame.to_csv(handle, index=False)

    summary = (
        window_frame.groupby("method")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            worst_window_accuracy=("accuracy", "min"),
            mean_nll=("nll", "mean"),
            mean_ece=("ece", "mean"),
            mean_gain=("gain_over_source", "mean"),
            negative_transfer_rate=("gain_over_source", lambda value: float((value < 0).mean())),
            negative_transfer_rate_0p5=(
                "gain_over_source", lambda value: float((value < -0.005).mean())
            ),
            severe_negative_transfer_rate=(
                "gain_over_source", lambda value: float((value < -0.02).mean())
            ),
        )
        .reset_index()
    )
    routed = window_frame[window_frame.method == "shiftroute"].copy()
    routing_summary = {
        "action_applied_accuracy": float(
            (routed.action_applied == routed.true_action).mean()
        ),
        "action_selected_next_accuracy": float(
            (routed.action_selected_next == routed.true_action).mean()
        ),
        "mean_prior_total_variation": float(routed.prior_total_variation.mean()),
    }
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "thresholds": thresholds.__dict__,
        "routing_summary": routing_summary,
        "input_sha256": {
            "source": sha256(args.source),
            "calibration": sha256(args.calibration),
            "clean": sha256(args.clean),
            "representation_a": sha256(args.representation_a),
            "representation_b": sha256(args.representation_b),
        },
        "output_sha256": {
            "window_metrics": sha256(window_path),
            "summary": sha256(summary_path),
        },
        "causal_protocol": "predict_then_diagnose_then_update_for_next_window",
        "paired_test_identities": bool(paired),
        "calibration_protocol": {
            "mode": args.calibration_mode,
            "identity_disjoint_windows": args.calibration_mode == "disjoint_max",
            "effective_windows": thresholds.calibration_windows,
            "familywise_alpha": args.familywise_alpha,
            "development_scale_quantile": args.development_scale_quantile,
            "prior_reliability_lower_quantile": args.prior_reliability_lower_quantile,
            "prior_reliability_threshold": prior_reliability_threshold,
        },
        "labels_used_for_controller": False,
        "oracle_uses_phase_metadata": True,
    }
    if args.development is not None:
        metadata["input_sha256"]["development"] = sha256(args.development)
    if args.event_log == "sample":
        metadata["output_sha256"]["events"] = sha256(event_path)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
