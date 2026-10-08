"""Aggregate hashed ShiftRoute runs into paper-ready statistical tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_input(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must have LABEL=PATH form")
    label, path = value.split("=", 1)
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_interval(values: np.ndarray, repetitions: int, rng) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(repetitions, values.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down family-wise error correction without extra dependencies."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(p_values[valid])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_statistics(differences: np.ndarray) -> Dict[str, float]:
    """Return two-sided Wilcoxon and paired effect sizes in a beneficial direction."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    nonzero = differences[differences != 0]
    if differences.size < 2 or nonzero.size == 0:
        p_value = 1.0
    else:
        p_value = float(wilcoxon(differences, alternative="two-sided", method="auto").pvalue)
    if differences.size > 1 and differences.std(ddof=1) > 0:
        cohen_dz = float(differences.mean() / differences.std(ddof=1))
    else:
        cohen_dz = 0.0 if differences.size else float("nan")
    if nonzero.size:
        ranks = rankdata(np.abs(nonzero), method="average")
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = (positive - negative) / (positive + negative)
    else:
        rank_biserial = 0.0
    return {
        "pairs": int(differences.size),
        "mean_benefit": float(differences.mean()) if differences.size else float("nan"),
        "median_benefit": float(np.median(differences)) if differences.size else float("nan"),
        "cohen_dz": cohen_dz,
        "rank_biserial": float(rank_biserial),
        "p_raw_two_sided": p_value,
    }


def run_descriptor(run_id: str) -> Dict[str, object]:
    """Recover condition labels encoded by the deterministic pipeline directory names."""
    tag = Path(run_id).name
    standard = re.match(r"s(?P<severity>\d+)_(?P<a>.+)__(?P<b>.+)_seed\d+$", tag)
    raw = re.match(r"(?P<a>.+)_s(?P<severity_a>\d+)__(?P<b>.+)_s(?P<severity_b>\d+)_seed\d+$", tag)
    natural = re.match(r"(?P<a>.+)__(?P<b>.+)_w\d+_ir[\d.]+_seed\d+$", tag)
    condition = standard or raw or natural
    severity_a = int(raw.group("severity_a")) if raw else int(standard.group("severity")) if standard else None
    severity_b = int(raw.group("severity_b")) if raw else int(standard.group("severity")) if standard else None
    return {
        "severity": severity_a if severity_a == severity_b else None,
        "severity_a": severity_a,
        "severity_b": severity_b,
        "condition_a": condition.group("a") if condition else None,
        "condition_b": condition.group("b") if condition else None,
        "experiment_variant": str(Path(run_id).parent) if Path(run_id).parent != Path(".") else "main",
    }


def condition_identifier(run_id: str) -> str:
    """Remove only the stream seed so inference clusters matched conditions."""

    return re.sub(r"_seed\d+$", "", str(run_id))


def action_components(action: object) -> set:
    return {
        "freeze": set(),
        "prior_only": {"prior"},
        "representation_only": {"representation"},
        "joint": {"prior", "representation"},
    }.get(str(action), set())


def first_offset(mask: pd.Series) -> Tuple[int, bool]:
    positions = np.flatnonzero(mask.to_numpy(bool))
    return (int(positions[0]), False) if len(positions) else (int(len(mask)), True)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    probability = successes / trials
    denominator = 1 + z * z / trials
    center = (probability + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(probability * (1 - probability) / trials + z * z / (4 * trials * trials)) / denominator
    return float(center - radius), float(center + radius)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    run_summaries: List[pd.DataFrame] = []
    phase_summaries: List[pd.DataFrame] = []
    routed_windows: List[pd.DataFrame] = []
    provenance: List[Dict[str, object]] = []
    tail_records: List[Dict[str, object]] = []
    route_audits: List[Dict[str, object]] = []
    delay_records: List[Dict[str, object]] = []
    recovery_records: List[Dict[str, object]] = []
    diagnostic_records: List[Dict[str, object]] = []

    for dataset, root in args.input:
        for metrics_path in sorted(root.rglob("window_metrics.csv")):
            run_dir = metrics_path.parent
            metadata_path = run_dir / "metadata.json"
            summary_path = run_dir / "summary.csv"
            if not metadata_path.exists() or not summary_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(metrics_path)
            run_id = str(run_dir.relative_to(root))
            arguments = metadata.get("arguments", {})
            descriptor = run_descriptor(run_id)
            common = {
                "dataset": dataset,
                "run_id": run_id,
                "condition_id": condition_identifier(run_id),
                "seed": arguments.get("seed"),
                "window_size": arguments.get("window_size"),
                "imbalance_ratio": arguments.get("imbalance_ratio"),
                "prior_strength": arguments.get("prior_strength"),
                "prior_momentum": arguments.get("prior_momentum"),
                "prior_ratio_clip": arguments.get("prior_ratio_clip"),
                "prior_enter": metadata.get("thresholds", {}).get("prior_enter"),
                "representation_enter": metadata.get("thresholds", {}).get("representation_enter"),
                "peak_gpu_memory_bytes": metadata.get("peak_gpu_memory_bytes"),
                **descriptor,
            }
            summary = pd.read_csv(summary_path).assign(**common)
            run_summaries.append(summary)
            annotated = frame.assign(**common)
            phase = (
                annotated.groupby(["phase", "method"], as_index=False)
                .agg(
                    accuracy=("accuracy", "mean"),
                    nll=("nll", "mean"),
                    ece=("ece", "mean"),
                    gain_over_source=("gain_over_source", "mean"),
                    worst_window_accuracy=("accuracy", "min"),
                )
                .assign(**common)
            )
            phase_summaries.append(phase)
            routed_name = "shiftroute_tent" if "shiftroute_tent" in set(frame.method) else "shiftroute"
            routed = annotated[annotated.method == routed_name].copy()
            for method, method_rows in annotated.groupby("method"):
                accuracy = method_rows.accuracy.to_numpy(float)
                gain = method_rows.gain_over_source.to_numpy(float)
                tail_count = max(1, int(np.ceil(0.05 * len(accuracy))))
                tail_records.append(
                    {
                        **common,
                        "method": method,
                        "windows": len(method_rows),
                        "accuracy_q05": float(np.quantile(accuracy, 0.05)),
                        "accuracy_cvar05": float(np.sort(accuracy)[:tail_count].mean()),
                        "gain_q05": float(np.quantile(gain, 0.05)),
                        "gain_cvar05": float(np.sort(gain)[:tail_count].mean()),
                    }
                )
            if not routed.empty and {"true_action", "action_applied"}.issubset(routed.columns):
                routed = routed.sort_values("window").copy()
                routed["phase_offset"] = routed.groupby("phase", sort=False).cumcount()
                confirmation_windows = int(
                    metadata.get("arguments", {}).get("confirmation_windows", 2)
                )
                min_dwell_windows = int(
                    metadata.get("arguments", {}).get("min_dwell_windows", 2)
                )
                transition_buffer_windows = confirmation_windows + min_dwell_windows
                routed["transition_buffer_windows"] = transition_buffer_windows
                steady = routed.phase_offset >= transition_buffer_windows
                required_components = routed.true_action.map(action_components)
                applied_components = routed.action_applied.map(action_components)
                unnecessary = [bool(applied - required) for applied, required in zip(applied_components, required_components)]
                missed = [bool(required - applied) for applied, required in zip(applied_components, required_components)]
                clean = routed.true_action == "freeze"
                clean_steady = clean & steady
                selected_accuracy = float("nan")
                if "action_selected_next" in routed:
                    next_required = routed.true_action.shift(-1)
                    valid_next = next_required.notna()
                    selected_accuracy = float(
                        (routed.loc[valid_next, "action_selected_next"] == next_required.loc[valid_next]).mean()
                    )
                clean_diagnostic_alarm = float("nan")
                clean_diagnostic_alarm_count = None
                threshold_columns = ["prior_score", "representation_score", "prior_enter", "representation_enter"]
                if clean.any() and set(threshold_columns).issubset(routed.columns) and routed.loc[clean, threshold_columns].notna().all().all():
                    clean_alarm_mask = (
                            (routed.loc[clean, "prior_score"] > routed.loc[clean, "prior_enter"])
                            | (routed.loc[clean, "representation_score"] > routed.loc[clean, "representation_enter"])
                        )
                    clean_diagnostic_alarm = float(clean_alarm_mask.mean())
                    clean_diagnostic_alarm_count = int(clean_alarm_mask.sum())
                route_audits.append(
                    {
                        **common,
                        "windows": len(routed),
                        "action_accuracy": float((routed.action_applied == routed.true_action).mean()),
                        "selected_next_action_accuracy": selected_accuracy,
                        "confirmation_windows": confirmation_windows,
                        "min_dwell_windows": min_dwell_windows,
                        "transition_buffer_windows": transition_buffer_windows,
                        "steady_windows": int(steady.sum()),
                        "steady_action_accuracy": float(
                            (routed.loc[steady, "action_applied"] == routed.loc[steady, "true_action"]).mean()
                        ) if steady.any() else float("nan"),
                        "clean_windows": int(clean.sum()),
                        "clean_false_action_count": int((routed.loc[clean, "action_applied"] != "freeze").sum()),
                        "clean_false_action_rate": float((routed.loc[clean, "action_applied"] != "freeze").mean()) if clean.any() else float("nan"),
                        "clean_steady_windows": int(clean_steady.sum()),
                        "clean_steady_false_action_count": int(
                            (routed.loc[clean_steady, "action_applied"] != "freeze").sum()
                        ),
                        "clean_steady_false_action_rate": float(
                            (routed.loc[clean_steady, "action_applied"] != "freeze").mean()
                        ) if clean_steady.any() else float("nan"),
                        "clean_diagnostic_alarm_count": clean_diagnostic_alarm_count,
                        "clean_diagnostic_alarm_rate": clean_diagnostic_alarm,
                        "unnecessary_component_rate": float(np.mean(unnecessary)),
                        "missed_component_rate": float(np.mean(missed)),
                        "switches_per_100_windows": float(100 * routed.action_applied.ne(routed.action_applied.shift()).iloc[1:].mean()) if len(routed) > 1 else 0.0,
                    }
                )
                ordered = routed.sort_values("window").reset_index(drop=True)
                segment_ids = ordered.phase.ne(ordered.phase.shift()).cumsum()
                segments = list(ordered.groupby(segment_ids, sort=False))
                for segment_number, (_, segment) in enumerate(segments):
                    required_action = str(segment.true_action.iloc[0])
                    delay, censored = first_offset(segment.action_applied == required_action)
                    delay_records.append(
                        {
                            **common,
                            "segment": segment_number,
                            "phase": str(segment.phase.iloc[0]),
                            "required_action": required_action,
                            "segment_windows": len(segment),
                            "route_delay_windows": delay,
                            "route_delay_censored": censored,
                        }
                    )
                    previous_action = str(segments[segment_number - 1][1].true_action.iloc[0]) if segment_number else "none"
                    if required_action == "freeze" and previous_action != "freeze":
                        freeze_delay, freeze_censored = first_offset(segment.action_applied == "freeze")
                        accuracy_delay, accuracy_censored = first_offset(segment.gain_over_source >= -0.005)
                        recovery_records.append(
                            {
                                **common,
                                "segment": segment_number,
                                "phase": str(segment.phase.iloc[0]),
                                "previous_required_action": previous_action,
                                "segment_windows": len(segment),
                                "freeze_recovery_delay_windows": freeze_delay,
                                "freeze_recovery_censored": freeze_censored,
                                "accuracy_recovery_delay_windows": accuracy_delay,
                                "accuracy_recovery_censored": accuracy_censored,
                                "recovery_mean_gain": float(segment.gain_over_source.mean()),
                                "recovery_worst_gain": float(segment.gain_over_source.min()),
                            }
                        )
                prior_target = routed.true_action.isin(["prior_only", "joint"]).astype(int)
                representation_target = routed.true_action.isin(["representation_only", "joint"]).astype(int)
                diagnostic = {**common, "windows": len(routed)}
                for axis, target, score_name in (
                    ("prior", prior_target, "prior_score"),
                    ("representation", representation_target, "representation_score"),
                ):
                    if score_name in routed and target.nunique() == 2:
                        diagnostic[f"{axis}_auroc"] = float(roc_auc_score(target, routed[score_name]))
                        diagnostic[f"{axis}_average_precision"] = float(average_precision_score(target, routed[score_name]))
                    else:
                        diagnostic[f"{axis}_auroc"] = float("nan")
                        diagnostic[f"{axis}_average_precision"] = float("nan")
                diagnostic_records.append(diagnostic)
            routed_windows.append(routed)
            provenance.append(
                {
                    **common,
                    "run_dir": str(run_dir),
                    "metadata_path": str(metadata_path),
                    "window_metrics_sha256": metadata.get("output_sha256", {}).get("window_metrics"),
                    "causal_protocol": metadata.get("causal_protocol", metadata.get("protocol")),
                    "labels_used_for_controller": metadata.get(
                        "labels_used_for_controller",
                        metadata.get("labels_used_for_controller_or_update"),
                    ),
                }
            )

    if not run_summaries:
        raise RuntimeError("no complete runs were found")
    run_frame = pd.concat(run_summaries, ignore_index=True)
    phase_frame = pd.concat(phase_summaries, ignore_index=True)
    routed_frame = pd.concat(routed_windows, ignore_index=True)
    run_frame.to_csv(args.output_dir / "run_summary.csv", index=False)
    phase_frame.to_csv(args.output_dir / "phase_summary.csv", index=False)
    routed_frame.to_csv(args.output_dir / "routed_windows.csv.gz", index=False, compression="gzip")
    pd.DataFrame(provenance).to_csv(args.output_dir / "provenance.csv", index=False)
    pd.DataFrame(tail_records).to_csv(args.output_dir / "tail_risk.csv", index=False)
    route_audit_frame = pd.DataFrame(route_audits)
    route_audit_frame.to_csv(args.output_dir / "route_audit.csv", index=False)
    calibration_records = []
    if not route_audit_frame.empty:
        for dataset, group in route_audit_frame.groupby("dataset"):
            for measure, column, rate_column in (
                ("router_false_action_all", "clean_false_action_count", "clean_false_action_rate"),
                ("router_false_action_steady", "clean_steady_false_action_count", "clean_steady_false_action_rate"),
                ("diagnostic_familywise_alarm", "clean_diagnostic_alarm_count", "clean_diagnostic_alarm_rate"),
            ):
                valid = group.dropna(subset=[column])
                trial_column = "clean_steady_windows" if "steady" in measure else "clean_windows"
                if measure == "diagnostic_familywise_alarm":
                    unique_keys = [
                        key
                        for key in (
                            "seed",
                            "window_size",
                            "imbalance_ratio",
                            "prior_enter",
                            "representation_enter",
                            "experiment_variant",
                        )
                        if key in valid.columns
                    ]
                    valid = valid.drop_duplicates(subset=unique_keys)
                valid_trials = int(valid[trial_column].sum())
                successes = int(valid[column].sum()) if len(valid) else 0
                lower, upper = wilson_interval(successes, valid_trials)
                if measure == "diagnostic_familywise_alarm":
                    cluster_rates = (valid[column] / valid[trial_column]).to_numpy(float)
                    cluster_unit = "unique_clean_stream_configuration"
                else:
                    cluster_totals = valid.groupby("condition_id", dropna=False)[
                        [column, trial_column]
                    ].sum()
                    cluster_rates = (
                        cluster_totals[column] / cluster_totals[trial_column]
                    ).to_numpy(float)
                    cluster_unit = "condition_cell_with_seeds_pooled"
                cluster_lower, cluster_upper = bootstrap_interval(
                    cluster_rates, args.bootstrap_repetitions, rng
                )
                calibration_records.append(
                    {
                        "dataset": dataset,
                        "measure": measure,
                        "alarms": successes,
                        "clean_windows": valid_trials,
                        "rate": successes / valid_trials if valid_trials else float("nan"),
                        "wilson95_lower": lower,
                        "wilson95_upper": upper,
                        "run_cluster_bootstrap95_lower": cluster_lower,
                        "run_cluster_bootstrap95_upper": cluster_upper,
                        "nominal_familywise_alpha": 0.05 if measure == "diagnostic_familywise_alarm" else float("nan"),
                        "clusters": int(len(cluster_rates)),
                        "cluster_unit": cluster_unit,
                        "interval_note": "Wilson is descriptive pooled-window interval; bootstrap resamples the declared cluster unit",
                    }
                )
    pd.DataFrame(calibration_records).to_csv(args.output_dir / "clean_false_route_calibration.csv", index=False)
    pd.DataFrame(delay_records).to_csv(args.output_dir / "route_delay.csv", index=False)
    pd.DataFrame(recovery_records).to_csv(args.output_dir / "recovery_audit.csv", index=False)
    pd.DataFrame(diagnostic_records).to_csv(args.output_dir / "diagnostic_discrimination.csv", index=False)

    metric_names = [
        "mean_accuracy",
        "worst_window_accuracy",
        "mean_nll",
        "mean_ece",
        "mean_gain",
        "negative_transfer_rate_0p5",
        "severe_negative_transfer_rate",
    ]
    records = []
    for (dataset, method), group in run_frame.groupby(["dataset", "method"]):
        for metric in metric_names:
            if metric not in group:
                continue
            run_values = group[metric].dropna().to_numpy(float)
            values = (
                group.groupby("condition_id", dropna=False)[metric]
                .mean()
                .dropna()
                .to_numpy(float)
            )
            lower, upper = bootstrap_interval(values, args.bootstrap_repetitions, rng)
            records.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "runs": len(run_values),
                    "condition_cells": len(values),
                    "inference_unit": "condition_mean_over_seeds",
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    pd.DataFrame(records).to_csv(args.output_dir / "method_bootstrap_ci.csv", index=False)

    higher_is_better = {"mean_accuracy", "worst_window_accuracy", "mean_gain"}
    test_metrics = [
        "mean_accuracy",
        "worst_window_accuracy",
        "mean_nll",
        "mean_ece",
        "negative_transfer_rate_0p5",
        "severe_negative_transfer_rate",
    ]
    test_records = []
    for dataset, dataset_frame in run_frame.groupby("dataset"):
        methods = sorted(dataset_frame.method.unique())
        routed_methods = [
            name for name in methods if name in {"shiftroute", "shiftroute_tent"}
        ]
        for metric in test_metrics:
            if metric not in dataset_frame:
                continue
            rate_metric = metric in {
                "negative_transfer_rate_0p5",
                "severe_negative_transfer_rate",
            }
            comparisons = set()
            if not rate_metric:
                comparisons.update(
                    (name, "source", "method_vs_source")
                    for name in methods
                    if name not in {
                        "source", "oracle", "oracle_tent", "shiftroute_ods"
                    }
                )
            for routed_name in routed_methods:
                comparisons.update(
                    (routed_name, baseline, "routed_vs_baseline")
                    for baseline in methods
                    if baseline not in {routed_name, "oracle", "oracle_tent"}
                    and baseline != "shiftroute_ods"
                    and not (rate_metric and baseline == "source")
                )
            pivot = dataset_frame.pivot_table(
                index="condition_id", columns="method", values=metric, aggfunc="mean"
            )
            for method, reference, family in sorted(comparisons):
                if method not in pivot or reference not in pivot:
                    continue
                paired = pivot[[method, reference]].dropna()
                raw_difference = paired[method].to_numpy(float) - paired[reference].to_numpy(float)
                beneficial_difference = raw_difference if metric in higher_is_better else -raw_difference
                test_records.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "comparison_family": family,
                        "method": method,
                        "reference": reference,
                        "inference_unit": "condition_mean_over_seeds",
                        **paired_statistics(beneficial_difference),
                    }
                )
    tests = pd.DataFrame(test_records)
    if not tests.empty:
        tests["p_holm"] = np.nan
        for _, indices in tests.groupby(["dataset", "metric", "comparison_family"]).groups.items():
            tests.loc[indices, "p_holm"] = holm_adjust(tests.loc[indices, "p_raw_two_sided"].to_numpy())
        tests["significant_0p05"] = tests.p_holm < 0.05
    tests.to_csv(args.output_dir / "paired_tests.csv", index=False)

    confusion_columns = [
        "dataset", "true_action", "action_applied", "windows", "row_fraction"
    ]

    def make_action_confusion(frame: pd.DataFrame) -> pd.DataFrame:
        if not {"true_action", "action_applied"}.issubset(frame.columns):
            return pd.DataFrame(columns=confusion_columns)
        result = (
            frame.groupby(["dataset", "true_action", "action_applied"])
            .size()
            .rename("windows")
            .reset_index()
        )
        totals = result.groupby(["dataset", "true_action"]).windows.transform("sum")
        result["row_fraction"] = result.windows / totals
        return result

    make_action_confusion(routed_frame).to_csv(
        args.output_dir / "action_confusion.csv", index=False
    )
    if {"phase", "window"}.issubset(routed_frame.columns):
        steady_frame = routed_frame.copy()
        steady_frame["phase_offset"] = steady_frame.groupby(
            ["dataset", "run_id", "phase"], sort=False
        ).cumcount()
        if "transition_buffer_windows" in steady_frame:
            buffer_windows = steady_frame.transition_buffer_windows.fillna(2).astype(int)
        else:
            buffer_windows = pd.Series(2, index=steady_frame.index)
        steady_frame = steady_frame[steady_frame.phase_offset >= buffer_windows]
    else:
        steady_frame = routed_frame.iloc[0:0]
    make_action_confusion(steady_frame).to_csv(
        args.output_dir / "action_confusion_steady.csv", index=False
    )
    output_files = sorted(args.output_dir.glob("*.csv")) + sorted(args.output_dir.glob("*.csv.gz"))
    aggregation_manifest = {
        "inputs": [{"label": label, "root": str(root)} for label, root in args.input],
        "complete_runs": int(run_frame[["dataset", "run_id"]].drop_duplicates().shape[0]),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "outputs": [{"path": str(path), "sha256": sha256(path)} for path in output_files],
    }
    (args.output_dir / "aggregation_manifest.json").write_text(
        json.dumps(aggregation_manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "complete_runs": int(run_frame[["dataset", "run_id"]].drop_duplicates().shape[0]),
                "datasets": sorted(run_frame.dataset.unique().tolist()),
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
