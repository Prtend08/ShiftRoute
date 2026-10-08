"""Create traceable CSV and LaTeX tables from an aggregate experiment directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_optional(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def mean_sd(values: pd.Series, scale: float = 1.0, digits: int = 2) -> str:
    clean = values.dropna().astype(float) * scale
    if clean.empty:
        return "--"
    standard_deviation = clean.std(ddof=1) if len(clean) > 1 else 0.0
    return f"{clean.mean():.{digits}f} ± {standard_deviation:.{digits}f}"


def write_table(
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
    manifest: Dict[str, object],
    *,
    caption: str,
    label: str,
) -> None:
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"
    frame.to_csv(csv_path, index=False)
    tabular = frame.to_latex(index=False, escape=True, na_rep="--").replace("±", "$\\pm$")
    latex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{tabular}"
        "}\n"
        "\\end{table*}\n"
    )
    tex_path.write_text(latex, encoding="utf-8")
    manifest.setdefault("outputs", []).extend(
        [
            {"path": str(csv_path), "sha256": sha256(csv_path)},
            {"path": str(tex_path), "sha256": sha256(tex_path)},
        ]
    )


def estimate_ci(
    intervals: pd.DataFrame,
    dataset: str,
    method: str,
    metric: str,
    *,
    scale: float = 1.0,
    digits: int = 2,
) -> str:
    rows = intervals[
        (intervals.dataset == dataset)
        & (intervals.method == method)
        & (intervals.metric == metric)
    ]
    if rows.empty:
        return "--"
    row = rows.iloc[0]
    return (
        f"{float(row['mean']) * scale:.{digits}f} "
        f"[{float(row.ci95_lower) * scale:.{digits}f}, "
        f"{float(row.ci95_upper) * scale:.{digits}f}]"
    )


def display_method(method: str) -> str:
    names = {
        "source": "Source (frozen)",
        "always_prior": "Always prior",
        "always_representation": "Always representation",
        "always_joint": "Always joint",
        "shiftroute": "ShiftRoute",
        "oracle": "Mechanism oracle",
        "prequential_tent": "Tent",
        "prequential_eta": "ETA",
        "prequential_sar": "SAR",
        "prequential_deyo": "DeYO",
        "prequential_ods": "ODS+Tent",
        "shiftroute_ods": "ShiftRoute-gated ODS+Tent",
        "prequential_joint": "Prior + Tent",
        "shiftroute_tent": "ShiftRoute + Tent",
        "oracle_tent": "Mechanism oracle + Tent",
    }
    return names.get(str(method), str(method).replace("_", " "))


def main_table(
    run_summary: pd.DataFrame,
    tail: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for (dataset, method), group in run_summary.groupby(["dataset", "method"]):
        record = {
            "Dataset": dataset,
            "Method": display_method(str(method)),
            "Cells": int(group.condition_id.nunique()),
            "Accuracy (%, 95% CI)": estimate_ci(intervals, dataset, method, "mean_accuracy", scale=100),
            "Gain (pp, 95% CI)": estimate_ci(intervals, dataset, method, "mean_gain", scale=100, digits=3),
            "Worst (%, 95% CI)": estimate_ci(intervals, dataset, method, "worst_window_accuracy", scale=100),
        }
        if "negative_transfer_rate_0p5" in group:
            record["NT >0.5 pp (%, 95% CI)"] = estimate_ci(
                intervals, dataset, method, "negative_transfer_rate_0p5", scale=100
            )
        if "severe_negative_transfer_rate" in group:
            record["NT >2 pp (%, 95% CI)"] = estimate_ci(
                intervals, dataset, method, "severe_negative_transfer_rate", scale=100
            )
        if not tail.empty:
            subset = tail[(tail.dataset == dataset) & (tail.method == method)]
            record["Accuracy CVaR5 (%)"] = mean_sd(subset.accuracy_cvar05, 100) if len(subset) else "--"
        records.append(record)
    return pd.DataFrame(records)


def compact_main_table(
    run_summary: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    """One readable row per evaluation suite for the main manuscript."""

    records: List[Dict[str, object]] = []
    for dataset, dataset_group in run_summary.groupby("dataset"):
        methods = set(dataset_group.method.astype(str))
        if "shiftroute_tent" in methods:
            comparator = "prequential_tent"
            routed = "shiftroute_tent"
        else:
            comparator = "always_representation"
            routed = "shiftroute"
        required = {"source", comparator, routed}
        if not required.issubset(methods):
            continue
        means = dataset_group.groupby("method").mean(numeric_only=True)
        records.append(
            {
                "Dataset": dataset,
                "Source acc. (%)": f"{100 * means.loc['source', 'mean_accuracy']:.2f}",
                "Always-active": display_method(comparator),
                "Comparator acc. (%)": f"{100 * means.loc[comparator, 'mean_accuracy']:.2f}",
                "ShiftRoute acc. (%)": f"{100 * means.loc[routed, 'mean_accuracy']:.2f}",
                "Gain vs source (pp)": f"{100 * (means.loc[routed, 'mean_accuracy'] - means.loc['source', 'mean_accuracy']):+.3f}",
                "Gain vs comparator (pp)": f"{100 * (means.loc[routed, 'mean_accuracy'] - means.loc[comparator, 'mean_accuracy']):+.3f}",
                "NT reduction (pp)": f"{100 * (means.loc[comparator, 'negative_transfer_rate_0p5'] - means.loc[routed, 'negative_transfer_rate_0p5']):+.2f}",
            }
        )
    return pd.DataFrame(records)


def external_raw_baseline_table(
    run_summary: pd.DataFrame,
    paired_tests: pd.DataFrame,
) -> pd.DataFrame:
    """Compact head-to-head table for independently published raw-image TTA methods."""

    methods = [
        "shiftroute_tent",
        "shiftroute_ods",
        "prequential_tent",
        "prequential_eta",
        "prequential_sar",
        "prequential_deyo",
        "prequential_ods",
    ]
    records: List[Dict[str, object]] = []
    for dataset in ("CIFAR10-C-raw",):
        subset = run_summary[
            (run_summary.dataset == dataset) & run_summary.method.isin(methods)
        ].copy()
        if dataset == "CIFAR100-C-raw":
            # The archived ODS runs used k=5; the published CIFAR100 setting is
            # k=2. Exclude those rows until the corrected run is available.
            subset = subset[~subset.method.isin(["prequential_ods", "shiftroute_ods"])]
        if subset.empty or "shiftroute_tent" not in set(subset.method):
            continue
        cell_means = (
            subset.groupby(["method", "condition_id"], as_index=False)
            .mean(numeric_only=True)
        )
        macro = cell_means.groupby("method").mean(numeric_only=True)
        routed = macro.loc["shiftroute_tent"]
        for method in methods:
            if method not in macro.index:
                continue
            row = macro.loc[method]
            accuracy_test = paired_tests[
                (paired_tests.dataset == dataset)
                & (paired_tests.method == "shiftroute_tent")
                & (paired_tests.reference == method)
                & (paired_tests.metric == "mean_accuracy")
                & (paired_tests.comparison_family == "routed_vs_baseline")
            ]
            nt_test = paired_tests[
                (paired_tests.dataset == dataset)
                & (paired_tests.method == "shiftroute_tent")
                & (paired_tests.reference == method)
                & (paired_tests.metric == "negative_transfer_rate_0p5")
                & (paired_tests.comparison_family == "routed_vs_baseline")
            ]
            if method == "shiftroute_tent":
                significance = "--"
            elif method == "shiftroute_ods":
                significance = "exploratory"
            else:
                acc_sig = bool(len(accuracy_test) and accuracy_test.iloc[0].p_holm < 0.05)
                nt_sig = bool(len(nt_test) and nt_test.iloc[0].p_holm < 0.05)
                significance = f"{'yes' if acc_sig else 'no'}/{'yes' if nt_sig else 'no'}"
            records.append(
                {
                    "Dataset": dataset,
                    "Method": display_method(method),
                    "Accuracy (%)": f"{100 * row.mean_accuracy:.2f}",
                    "Worst (%)": f"{100 * row.worst_window_accuracy:.2f}",
                    "NT >0.5 pp (%)": f"{100 * row.negative_transfer_rate_0p5:.2f}",
                    "Acc. vs SR (pp)": f"{100 * (row.mean_accuracy - routed.mean_accuracy):+.2f}",
                    "NT vs SR (pp)": f"{100 * (row.negative_transfer_rate_0p5 - routed.negative_transfer_rate_0p5):+.2f}",
                    "Holm sig. acc./NT": significance,
                }
            )
    return pd.DataFrame(records)


def external_raw_baseline_sentence(run_summary: pd.DataFrame) -> str:
    clauses: List[str] = []
    for dataset in ("CIFAR10-C-raw", "CIFAR100-C-raw"):
        subset = run_summary[run_summary.dataset == dataset]
        cell_means = (
            subset.groupby(["method", "condition_id"], as_index=False)
            .mean(numeric_only=True)
        )
        macro = cell_means.groupby("method").mean(numeric_only=True)
        required = {"shiftroute_tent", "prequential_ods"}
        if not required.issubset(macro.index):
            continue
        routed = macro.loc["shiftroute_tent"]
        ods = macro.loc["prequential_ods"]
        dataset_label = dataset.replace("_", r"\_")
        clause = (
            f"On {dataset_label}, ODS+Tent achieved {100 * ods.mean_accuracy:.2f}\\% "
            f"accuracy and {100 * ods.negative_transfer_rate_0p5:.2f}\\% NT; "
            f"ShiftRoute\\allowbreak+Tent achieved {100 * routed.mean_accuracy:.2f}\\% "
            f"accuracy and {100 * routed.negative_transfer_rate_0p5:.2f}\\% NT"
        )
        if "shiftroute_ods" in macro.index:
            exploratory = macro.loc["shiftroute_ods"]
            clause += (
                f", while exploratory ShiftRoute-gated ODS+Tent achieved "
                f"{100 * exploratory.mean_accuracy:.2f}\\% accuracy and "
                f"{100 * exploratory.negative_transfer_rate_0p5:.2f}\\% NT"
            )
        clauses.append(clause)
    return ". ".join(clauses) + "." if clauses else ""


def routing_table(
    route: pd.DataFrame,
    diagnostic: pd.DataFrame,
    delay: pd.DataFrame,
    recovery: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for dataset, group in route.groupby("dataset"):
        diagnostic_group = diagnostic[diagnostic.dataset == dataset] if not diagnostic.empty else pd.DataFrame()
        delay_group = delay[delay.dataset == dataset] if not delay.empty else pd.DataFrame()
        recovery_group = recovery[recovery.dataset == dataset] if not recovery.empty else pd.DataFrame()
        records.append(
            {
                "Dataset": dataset,
                "Steady action accuracy (%)": f"{group.steady_action_accuracy.mean() * 100:.2f}",
                "Steady clean false action (%)": f"{group.clean_steady_false_action_rate.mean() * 100:.2f}",
                "Prior AUROC": mean_sd(diagnostic_group.prior_auroc, 1, 3) if len(diagnostic_group) else "--",
                "Representation AUROC": mean_sd(diagnostic_group.representation_auroc, 1, 3) if len(diagnostic_group) else "--",
                "Route delay (windows)": mean_sd(delay_group.route_delay_windows) if len(delay_group) else "--",
                "Accuracy recovery (windows)": mean_sd(recovery_group.accuracy_recovery_delay_windows) if len(recovery_group) else "--",
            }
        )
    return pd.DataFrame(records)


def efficiency_table(run_summary: pd.DataFrame) -> pd.DataFrame:
    required = {"mean_prediction_seconds", "mean_update_seconds"}
    if not required.issubset(run_summary.columns):
        return pd.DataFrame()
    valid = run_summary.dropna(subset=list(required)).copy()
    valid = valid[
        valid.method.isin(["source", "prequential_tent", "shiftroute_tent"])
    ]
    records = []
    for (dataset, method), group in valid.groupby(["dataset", "method"]):
        records.append(
            {
                "Dataset": dataset,
                "Method": display_method(str(method)),
                "Prediction (ms/window)": mean_sd(group.mean_prediction_seconds, 1000),
                "Update (ms/window)": mean_sd(group.mean_update_seconds, 1000),
                "Peak GPU memory (MiB)": mean_sd(group.peak_gpu_memory_bytes, 1 / 2**20, 1)
                if "peak_gpu_memory_bytes" in group
                else "--",
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_names = [
        "run_summary.csv",
        "tail_risk.csv",
        "route_audit.csv",
        "diagnostic_discrimination.csv",
        "route_delay.csv",
        "recovery_audit.csv",
        "clean_false_route_calibration.csv",
        "paired_tests.csv",
        "method_bootstrap_ci.csv",
    ]
    inputs = {name: read_optional(args.aggregate_dir, name) for name in input_names}
    if inputs["run_summary.csv"].empty:
        raise RuntimeError("run_summary.csv is required and must not be empty")
    manifest: Dict[str, object] = {
        "inputs": [
            {"path": str(args.aggregate_dir / name), "sha256": sha256(args.aggregate_dir / name)}
            for name in input_names
            if (args.aggregate_dir / name).exists()
        ],
        "note": "Tables are formatted from aggregate CSVs without manual numerical entry.",
    }
    write_table(
        main_table(
            inputs["run_summary.csv"],
            inputs["tail_risk.csv"],
            inputs["method_bootstrap_ci.csv"],
        ),
        args.output_dir,
        "main_performance",
        manifest,
        caption=(
            "Predictive performance from the locked prequential protocol. "
            "Intervals resample corruption/severity condition-cell means after averaging seeds."
        ),
        label="tab:main-performance",
    )
    write_table(
        compact_main_table(
            inputs["run_summary.csv"],
            inputs["method_bootstrap_ci.csv"],
        ),
        args.output_dir,
        "main_performance_compact",
        manifest,
        caption=(
            "Primary predictive and negative-transfer comparison after averaging "
            "stream seeds within condition cells. Positive NT reduction favors "
            "ShiftRoute; the complete method matrix and confidence intervals are "
            "provided in the accompanying reproducibility bundle."
        ),
        label="tab:main-performance-compact",
    )
    external = external_raw_baseline_table(
        inputs["run_summary.csv"],
        inputs["paired_tests.csv"],
    )
    if not external.empty:
        write_table(
            external,
            args.output_dir,
            "external_raw_baselines",
            manifest,
            caption=(
                "Head-to-head raw-image comparison with independently published "
                "TTA methods under the identical prequential streams and checkpoints. "
                "Positive NT versus ShiftRoute (SR) means more damaging windows. "
                "ODS is reported as ODS+Tent to match the implemented update, and "
                "the invalid archived CIFAR100 neighborhood configuration is excluded."
            ),
            label="tab:external-raw-baselines",
        )
        sentence = external_raw_baseline_sentence(inputs["run_summary.csv"])
        if sentence:
            sentence_path = args.output_dir / "external_raw_baseline_sentence.tex"
            sentence_path.write_text(sentence + "\n", encoding="utf-8")
            manifest.setdefault("outputs", []).append(
                {"path": str(sentence_path), "sha256": sha256(sentence_path)}
            )
    if not inputs["route_audit.csv"].empty:
        write_table(
            routing_table(
                inputs["route_audit.csv"],
                inputs["diagnostic_discrimination.csv"],
                inputs["route_delay.csv"],
                inputs["recovery_audit.csv"],
            ),
            args.output_dir,
            "routing_and_recovery",
            manifest,
            caption=(
                "Diagnostic and routing audit. Steady metrics exclude the predeclared "
                "confirmation-plus-dwell transition buffer; all-window metrics retain it."
            ),
            label="tab:routing",
        )
    if not inputs["clean_false_route_calibration.csv"].empty:
        calibration = inputs["clean_false_route_calibration.csv"].copy()
        calibration = calibration[
            calibration.measure.astype(str).str.contains("familywise_alarm")
        ].copy()
        calibration = pd.DataFrame(
            {
                "Dataset": calibration.dataset,
                "Alarms / windows": calibration.alarms.astype(str)
                + " / " + calibration.clean_windows.astype(str),
                "Rate (%)": (100 * calibration.rate).map(lambda value: f"{value:.2f}"),
                "Wilson 95% interval (%)": [
                    f"[{100 * lower:.2f}, {100 * upper:.2f}]"
                    for lower, upper in zip(
                        calibration.wilson95_lower, calibration.wilson95_upper
                    )
                ],
                "Independent configurations": calibration.clusters,
            }
        )
        write_table(
            calibration,
            args.output_dir,
            "clean_false_route_calibration",
            manifest,
            caption=(
                "Joint diagnostic alarms on source-like windows. Wilson intervals "
                "are descriptive; repeated windows are grouped by unique clean-stream configuration."
            ),
            label="tab:calibration",
        )
    if not inputs["paired_tests.csv"].empty:
        primary_tests = inputs["paired_tests.csv"].copy()
        primary_tests = primary_tests[
            (
                (primary_tests.metric.isin(["mean_accuracy", "worst_window_accuracy"]))
                & (primary_tests.reference == "source")
                & (primary_tests.method.isin(["shiftroute", "shiftroute_tent"]))
                & (primary_tests.comparison_family == "method_vs_source")
            )
            |
            (
                (primary_tests.metric.isin(["negative_transfer_rate_0p5", "severe_negative_transfer_rate"]))
                & (primary_tests.method.isin(["shiftroute", "shiftroute_tent"]))
                & (primary_tests.comparison_family == "routed_vs_baseline")
            )
        ].sort_values(["dataset", "metric", "p_holm"])
        primary_tests = primary_tests[
            [
                "dataset",
                "metric",
                "method",
                "reference",
                "pairs",
                "mean_benefit",
                "cohen_dz",
                "rank_biserial",
                "p_holm",
            ]
        ]
        primary_tests["mean_benefit"] = (primary_tests.mean_benefit * 100).map(lambda value: f"{value:+.3f}")
        for column in ("cohen_dz", "rank_biserial"):
            primary_tests[column] = primary_tests[column].map(lambda value: f"{value:.3f}")
        primary_tests["p_holm"] = primary_tests.p_holm.map(lambda value: f"{value:.3g}")
        primary_tests.columns = [
            "Dataset",
            "Metric",
            "Method",
            "Reference",
            "Pairs",
            "Benefit (pp)",
            "Cohen dz",
            "Rank-biserial",
            "Holm p",
        ]
        write_table(
            primary_tests,
            args.output_dir,
            "paired_significance_tests",
            manifest,
            caption=(
                "Paired condition-cell inference. Positive benefit favors the routed method. "
                "Holm correction is applied within each predeclared comparison family."
            ),
            label="tab:paired-tests",
        )
    efficiency = efficiency_table(inputs["run_summary.csv"])
    if not efficiency.empty:
        write_table(
            efficiency,
            args.output_dir,
            "efficiency",
            manifest,
            caption="Measured prediction and update cost per prequential window.",
            label="tab:efficiency",
        )
    manifest_path = args.output_dir / "table_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"tables": len(manifest.get("outputs", [])) // 2, "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
