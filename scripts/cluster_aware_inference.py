"""Recompute ShiftRoute inference at defensible top-level cluster units.

This audit intentionally leaves the manuscript untouched. It reads the final
run-level aggregate, averages stream seeds inside each condition, then averages
nested severity or operating-grid cells inside a declared top-level cluster.

Inferential suites:
  * CIFAR-C / raw CIFAR-C / ImageNet-C: corruption-pair clusters.

Descriptive-only suites:
  * ImageNet100: the nine window-size x imbalance-ratio cells are a fixed grid.
  * ImageNetV2: only three natural-shift variant-pair clusters remain after
    collapsing window size and imbalance ratio.

For inferential suites, comparison families and Holm adjustment match the
existing logic in scripts/aggregate_results.py. Descriptive suites receive no
Wilcoxon p-value and no multiplicity-adjusted significance flag.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


DEFAULT_INPUT = Path(
    "output/results/ShiftRoute_reproducibility_bundle/"
    "final_all_formal/run_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("paper/generated")

INFERENTIAL_DATASETS = {
    "CIFAR10-C",
    "CIFAR100-C",
    "CIFAR10-C-raw",
    "CIFAR100-C-raw",
    "ImageNet-C-R50",
    "ImageNet-C-ConvNeXt",
    "ImageNet-C-ViT",
}
DESCRIPTIVE_IMAGENET100 = {"ImageNet100"}
DESCRIPTIVE_IMAGENETV2 = {
    "ImageNetV2-R50",
    "ImageNetV2-ConvNeXt",
    "ImageNetV2-ViT",
}
EXPECTED_DATASETS = (
    INFERENTIAL_DATASETS | DESCRIPTIVE_IMAGENET100 | DESCRIPTIVE_IMAGENETV2
)

TEST_METRICS = [
    "mean_accuracy",
    "worst_window_accuracy",
    "mean_nll",
    "mean_ece",
    "negative_transfer_rate_0p5",
    "severe_negative_transfer_rate",
]
HIGHER_IS_BETTER = {"mean_accuracy", "worst_window_accuracy"}
RATE_METRICS = {
    "negative_transfer_rate_0p5",
    "severe_negative_transfer_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(p_values), dtype=float)
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


def paired_statistics(differences: np.ndarray) -> dict[str, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    nonzero = differences[differences != 0]
    if differences.size < 2 or nonzero.size == 0:
        p_value = 1.0
    else:
        p_value = float(
            wilcoxon(differences, alternative="two-sided", method="auto").pvalue
        )
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
        "mean_benefit": float(differences.mean()) if differences.size else np.nan,
        "median_benefit": float(np.median(differences)) if differences.size else np.nan,
        "cohen_dz": cohen_dz,
        "rank_biserial": float(rank_biserial),
        "wilcoxon_p_raw": p_value,
    }


def bootstrap_mean_interval(
    values: np.ndarray, repetitions: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(repetitions, values.size), replace=True).mean(1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def validate_input(frame: pd.DataFrame) -> None:
    required = {
        "dataset",
        "method",
        "condition_id",
        "seed",
        "condition_a",
        "condition_b",
        *TEST_METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"run summary is missing required columns: {missing}")

    datasets = set(frame.dataset.dropna().unique())
    missing_datasets = sorted(EXPECTED_DATASETS - datasets)
    unexpected = sorted(datasets - EXPECTED_DATASETS)
    if missing_datasets or unexpected:
        raise ValueError(
            "dataset inventory mismatch: "
            f"missing={missing_datasets}, unexpected={unexpected}"
        )

    duplicate_key = ["dataset", "condition_id", "method", "seed"]
    duplicates = frame.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, duplicate_key].head().to_dict("records")
        raise ValueError(f"duplicate run rows for {duplicate_key}: {examples}")

    metadata = frame[["dataset", "condition_id", "condition_a", "condition_b"]]
    inconsistent = (
        metadata.drop_duplicates()
        .groupby(["dataset", "condition_id"], dropna=False)
        .size()
    )
    if (inconsistent != 1).any():
        bad = inconsistent[inconsistent != 1].head().to_dict()
        raise ValueError(f"condition metadata is not unique: {bad}")

    inferential = frame[frame.dataset.isin(INFERENTIAL_DATASETS)]
    if inferential[["condition_a", "condition_b"]].isna().any().any():
        raise ValueError("inferential suites require condition_a and condition_b")


def attach_analysis_units(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["analysis_status"] = np.where(
        frame.dataset.isin(INFERENTIAL_DATASETS), "inferential", "descriptive_only"
    )
    frame["cluster_scheme"] = "condition_id"
    frame["descriptive_reason"] = ""

    pair_mask = frame.dataset.isin(INFERENTIAL_DATASETS | DESCRIPTIVE_IMAGENETV2)
    frame.loc[pair_mask, "cluster_scheme"] = "condition_a__condition_b"
    frame.loc[pair_mask, "top_cluster"] = (
        frame.loc[pair_mask, "condition_a"].astype(str)
        + "__"
        + frame.loc[pair_mask, "condition_b"].astype(str)
    )

    imagenet100_mask = frame.dataset.isin(DESCRIPTIVE_IMAGENET100)
    frame.loc[imagenet100_mask, "top_cluster"] = frame.loc[
        imagenet100_mask, "condition_id"
    ]
    frame.loc[imagenet100_mask, "descriptive_reason"] = (
        "fixed 3x3 window-size by imbalance-ratio grid; no IID superpopulation"
    )

    imagenetv2_mask = frame.dataset.isin(DESCRIPTIVE_IMAGENETV2)
    frame.loc[imagenetv2_mask, "descriptive_reason"] = (
        "three variant-pair clusters after collapsing window size and imbalance ratio"
    )
    return frame


def aggregate_top_clusters(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_metrics = [metric for metric in TEST_METRICS if metric in frame.columns]

    # First collapse stream seeds inside the exact condition cell.
    condition_means = (
        frame.groupby(
            [
                "dataset",
                "condition_id",
                "method",
                "analysis_status",
                "cluster_scheme",
                "top_cluster",
                "descriptive_reason",
            ],
            dropna=False,
            as_index=False,
        )[numeric_metrics]
        .mean()
    )

    # Then collapse nested severity or operating-grid cells inside the top cluster.
    top_means = (
        condition_means.groupby(
            [
                "dataset",
                "method",
                "analysis_status",
                "cluster_scheme",
                "top_cluster",
                "descriptive_reason",
            ],
            dropna=False,
            as_index=False,
        )[numeric_metrics]
        .mean()
    )

    cell_counts = (
        condition_means.groupby(["dataset", "top_cluster"], as_index=False)
        .condition_id.nunique()
        .rename(columns={"condition_id": "nested_condition_cells"})
    )
    return top_means, cell_counts


def build_comparisons(
    top_means: pd.DataFrame,
    cell_counts: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
    input_hash: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset, dataset_frame in top_means.groupby("dataset"):
        methods = sorted(dataset_frame.method.unique())
        routed_methods = [
            method for method in methods if method in {"shiftroute", "shiftroute_tent"}
        ]
        status = str(dataset_frame.analysis_status.iloc[0])
        scheme = str(dataset_frame.cluster_scheme.iloc[0])
        reason = str(dataset_frame.descriptive_reason.iloc[0])
        cluster_cell_counts = cell_counts[cell_counts.dataset == dataset]

        for metric in TEST_METRICS:
            if metric not in dataset_frame.columns:
                continue
            rate_metric = metric in RATE_METRICS
            comparisons: set[tuple[str, str, str]] = set()
            if not rate_metric:
                comparisons.update(
                    (method, "source", "method_vs_source")
                    for method in methods
                    if method
                    not in {"source", "oracle", "oracle_tent", "shiftroute_ods"}
                )
            for routed_method in routed_methods:
                comparisons.update(
                    (routed_method, baseline, "routed_vs_baseline")
                    for baseline in methods
                    if baseline
                    not in {
                        routed_method,
                        "oracle",
                        "oracle_tent",
                        "shiftroute_ods",
                    }
                    and not (rate_metric and baseline == "source")
                )

            pivot = dataset_frame.pivot_table(
                index="top_cluster", columns="method", values=metric, aggfunc="mean"
            )
            for method, reference, family in sorted(comparisons):
                if method not in pivot or reference not in pivot:
                    continue
                paired = pivot[[method, reference]].dropna()
                raw_difference = (
                    paired[method].to_numpy(float) - paired[reference].to_numpy(float)
                )
                beneficial = (
                    raw_difference if metric in HIGHER_IS_BETTER else -raw_difference
                )
                lower, upper = bootstrap_mean_interval(beneficial, repetitions, rng)
                stats = paired_statistics(beneficial)
                if status != "inferential":
                    stats["wilcoxon_p_raw"] = np.nan
                    stats["cohen_dz"] = np.nan
                    stats["rank_biserial"] = np.nan

                paired_counts = cluster_cell_counts[
                    cluster_cell_counts.top_cluster.isin(paired.index)
                ].nested_condition_cells
                records.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "comparison_family": family,
                        "method": method,
                        "reference": reference,
                        "analysis_status": status,
                        "cluster_scheme": scheme,
                        "descriptive_reason": reason,
                        "top_level_clusters": int(len(paired)),
                        "nested_condition_cells_min": int(paired_counts.min()),
                        "nested_condition_cells_max": int(paired_counts.max()),
                        "benefit_direction": (
                            "method_minus_reference"
                            if metric in HIGHER_IS_BETTER
                            else "reference_minus_method"
                        ),
                        **stats,
                        "cluster_bootstrap95_lower": lower,
                        "cluster_bootstrap95_upper": upper,
                        "mean_benefit_display_x100": 100.0 * stats["mean_benefit"],
                        "cluster_bootstrap95_lower_x100": 100.0 * lower,
                        "cluster_bootstrap95_upper_x100": 100.0 * upper,
                        "source_csv_sha256": input_hash,
                    }
                )

    results = pd.DataFrame(records)
    results["holm_family_size"] = np.nan
    results["holm_p_existing_family"] = np.nan
    inferential = results.analysis_status == "inferential"
    groups = results[inferential].groupby(
        ["dataset", "metric", "comparison_family"]
    ).groups
    for indices in groups.values():
        results.loc[indices, "holm_family_size"] = len(indices)
        results.loc[indices, "holm_p_existing_family"] = holm_adjust(
            results.loc[indices, "wilcoxon_p_raw"].to_numpy(float)
        )
    results["significant_holm_0p05"] = pd.array(
        [pd.NA] * len(results), dtype="boolean"
    )
    results.loc[inferential, "significant_holm_0p05"] = (
        results.loc[inferential, "holm_p_existing_family"] < 0.05
    ).to_numpy()
    return results.sort_values(
        ["dataset", "metric", "comparison_family", "method", "reference"]
    ).reset_index(drop=True)


def primary_rows(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(EXPECTED_DATASETS):
        routed = "shiftroute_tent" if dataset.endswith("-raw") else "shiftroute"
        nt_reference = (
            "prequential_tent" if dataset.endswith("-raw") else "always_representation"
        )
        selectors = [
            ("mean_accuracy", "source", "method_vs_source"),
            (
                "negative_transfer_rate_0p5",
                nt_reference,
                "routed_vs_baseline",
            ),
        ]
        for metric, reference, family in selectors:
            selected = results[
                (results.dataset == dataset)
                & (results.metric == metric)
                & (results.method == routed)
                & (results.reference == reference)
                & (results.comparison_family == family)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"expected one primary row for {dataset}/{metric}/{reference}, "
                    f"found {len(selected)}"
                )
            rows.append(selected.iloc[0])
    return pd.DataFrame(rows)


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "descriptive"
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def write_summaries(primary: pd.DataFrame, output_dir: Path, input_hash: str) -> None:
    mean_rows = primary[primary.metric == "mean_accuracy"]
    nt_rows = primary[primary.metric == "negative_transfer_rate_0p5"]
    inferential_mean = mean_rows[mean_rows.analysis_status == "inferential"]
    inferential_nt = nt_rows[nt_rows.analysis_status == "inferential"]

    mean_gain = int(
        ((inferential_mean.significant_holm_0p05 == True) & (inferential_mean.mean_benefit > 0)).sum()  # noqa: E712
    )
    mean_loss = int(
        ((inferential_mean.significant_holm_0p05 == True) & (inferential_mean.mean_benefit < 0)).sum()  # noqa: E712
    )
    mean_nonsig = len(inferential_mean) - mean_gain - mean_loss
    nt_significant = int((inferential_nt.significant_holm_0p05 == True).sum())  # noqa: E712

    lines = [
        "Cluster-aware ShiftRoute inference audit",
        f"Input SHA-256: {input_hash}",
        "",
        "Inference policy:",
        "- CIFAR-C, raw CIFAR-C, and ImageNet-C use corruption-pair clusters.",
        "- ImageNet100 and ImageNetV2 are descriptive only.",
        "- Holm families match scripts/aggregate_results.py.",
        "",
        (
            "Among the seven inferential suites, ShiftRoute mean accuracy has "
            f"{mean_gain} significant gains, {mean_nonsig} non-significant effects, "
            f"and {mean_loss} significant decreases under the existing Holm families."
        ),
        (
            "Negative-transfer reduction versus the fixed representation/Tent "
            f"comparator is significant in {nt_significant}/7 inferential suites."
        ),
        "",
        "Primary comparisons:",
    ]
    for _, row in primary.sort_values(["dataset", "metric"]).iterrows():
        unit = "pp" if row.metric in {"mean_accuracy", "negative_transfer_rate_0p5"} else "x100"
        lines.append(
            f"- {row.dataset} | {row.metric} | benefit="
            f"{row.mean_benefit_display_x100:+.3f} {unit} | clusters="
            f"{int(row.top_level_clusters)} | raw p={format_p(row.wilcoxon_p_raw)} | "
            f"Holm p={format_p(row.holm_p_existing_family)} | {row.analysis_status}"
        )
    (output_dir / "cluster_aware_inference_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    tex_lines = [
        "% Generated by scripts/cluster_aware_inference.py; do not edit manually.",
        "% This file is an audit artifact and is not included by the manuscript.",
        "\\begin{table*}[t]",
        "\\centering\\small",
        "\\caption{Cluster-aware audit of the two primary comparisons. ImageNet100 and ImageNetV2 are descriptive only.}",
        "\\label{tab:cluster-aware-audit}",
        "\\begin{tabular}{llrrrrl}",
        "\\toprule",
        "Suite & Outcome & Benefit (pp) & Clusters & Raw $p$ & Holm $p$ & Status \\\\",
        "\\midrule",
    ]
    dataset_order = sorted(EXPECTED_DATASETS)
    for dataset in dataset_order:
        for metric in ("mean_accuracy", "negative_transfer_rate_0p5"):
            row = primary[(primary.dataset == dataset) & (primary.metric == metric)].iloc[0]
            outcome = "Mean accuracy" if metric == "mean_accuracy" else "NT reduction"
            raw_p = format_p(float(row.wilcoxon_p_raw))
            holm_p = format_p(float(row.holm_p_existing_family))
            status = "Inferential" if row.analysis_status == "inferential" else "Descriptive"
            safe_dataset = dataset.replace("_", "\\_")
            tex_lines.append(
                f"{safe_dataset} & {outcome} & {row.mean_benefit_display_x100:+.3f} & "
                f"{int(row.top_level_clusters)} & {raw_p} & {holm_p} & {status} \\\\" 
            )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (output_dir / "cluster_aware_inference_summary.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    frame = pd.read_csv(args.input)
    validate_input(frame)
    frame = attach_analysis_units(frame)
    top_means, cell_counts = aggregate_top_clusters(frame)

    expected_clusters = {
        **{dataset: 8 for dataset in INFERENTIAL_DATASETS},
        **{dataset: 3 for dataset in DESCRIPTIVE_IMAGENETV2},
        "ImageNet100": 9,
    }
    observed_clusters = top_means.groupby("dataset").top_cluster.nunique().to_dict()
    if observed_clusters != expected_clusters:
        raise ValueError(
            f"top-level cluster inventory mismatch: observed={observed_clusters}, "
            f"expected={expected_clusters}"
        )

    input_hash = sha256(args.input)
    rng = np.random.default_rng(args.seed)
    results = build_comparisons(
        top_means,
        cell_counts,
        args.bootstrap_repetitions,
        rng,
        input_hash,
    )
    primary = primary_rows(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "cluster_aware_inference.csv", index=False)
    primary.to_csv(args.output_dir / "cluster_aware_primary_comparisons.csv", index=False)
    write_summaries(primary, args.output_dir, input_hash)
    print(
        primary[
            [
                "dataset",
                "metric",
                "analysis_status",
                "top_level_clusters",
                "mean_benefit_display_x100",
                "wilcoxon_p_raw",
                "holm_p_existing_family",
                "significant_holm_0p05",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
