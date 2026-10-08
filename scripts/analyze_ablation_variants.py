"""Aggregate component ablations relative to the locked full controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns


COMPACT_VARIANTS = [
    "no_prior_projection",
    "no_between_class_variance",
    "bootstrap_bonferroni_calibration",
    "stateless_router",
    "no_prior_reliability_guard",
    "no_source_anchor",
    "no_prototype_momentum",
    "full_unbounded_prior",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def safe_wilcoxon(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < 2 or np.allclose(array, 0.0):
        return 1.0
    return float(wilcoxon(array, alternative="two-sided").pvalue)


def display_name(value: str) -> str:
    mapping = {
        "no_prior_projection": "No prior projection",
        "no_between_class_variance": "No between-class variance",
        "bootstrap_bonferroni_calibration": "Bootstrap + Bonferroni",
        "stateless_router": "Stateless router",
        "no_prior_reliability_guard": "No reliability guard",
        "no_source_anchor": "No source anchor",
        "no_prototype_momentum": "No prototype momentum",
        "full_unbounded_prior": "Unbounded prior expert",
    }
    return mapping.get(value, value.replace("_", " ").title())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(summary: pd.DataFrame, statistics: pd.DataFrame, path: Path) -> None:
    stats = statistics.set_index(["experiment_variant", "metric"])
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Component and stress-test effects relative to the full controller. Values are paired percentage-point changes after seed averaging within each dataset--window-size cell. For accuracy and worst-window accuracy, negative is adverse; for negative-transfer frequency, positive is adverse. A dagger marks Holm-adjusted $p<0.05$.}",
        r"\label{tab:ablation-compact}",
        r"\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & Cells & $\Delta$ mean acc. & $\Delta$ NT$>0.5$ & $\Delta$ worst \\",
        r"\midrule",
    ]
    indexed = summary.set_index("experiment_variant")
    for variant in COMPACT_VARIANTS:
        if variant not in indexed.index:
            continue
        row = indexed.loc[variant]
        acc_mark = r"$^{\dagger}$" if stats.loc[(variant, "mean_accuracy"), "holm_p"] < 0.05 else ""
        nt_mark = r"$^{\dagger}$" if stats.loc[(variant, "negative_transfer_rate_0p5"), "holm_p"] < 0.05 else ""
        worst_mark = r"$^{\dagger}$" if stats.loc[(variant, "worst_window_accuracy"), "holm_p"] < 0.05 else ""
        lines.append(
            f"{display_name(variant)} & {int(row['cells'])} & "
            f"{row['delta_mean_accuracy_pp']:+.3f}{acc_mark} & "
            f"{row['delta_negative_transfer_rate_0p5_pp']:+.3f}{nt_mark} & "
            f"{row['delta_worst_window_accuracy_pp']:+.3f}{worst_mark} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_heatmap(dataset_summary: pd.DataFrame, path_pdf: Path, path_png: Path) -> None:
    available = [
        variant for variant in COMPACT_VARIANTS
        if variant in set(dataset_summary.experiment_variant)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    specifications = [
        ("delta_mean_accuracy_pp", "Mean accuracy change (pp)", "RdBu_r"),
        (
            "delta_negative_transfer_rate_0p5_pp",
            "NT>0.5 frequency change (pp)", "RdBu_r",
        ),
    ]
    for axis, (metric, title, palette) in zip(axes, specifications):
        pivot = dataset_summary.pivot(
            index="experiment_variant", columns="dataset", values=metric
        ).reindex(available)
        sns.heatmap(
            pivot, annot=True, fmt="+.2f", center=0.0, cmap=palette,
            linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.75}, ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_yticklabels([display_name(value) for value in pivot.index], rotation=0)
    figure.savefig(path_pdf, bbox_inches="tight")
    figure.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    frames = []
    for directory in args.aggregate_dir:
        path = directory / "run_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "dataset", "experiment_variant", "window_size", "seed", "method",
        "mean_accuracy", "worst_window_accuracy", "negative_transfer_rate_0p5",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = frame.loc[frame.method.eq("shiftroute")].copy()
    if "full" not in set(frame.experiment_variant):
        raise ValueError("the full experiment variant is required")

    metrics = [
        "mean_accuracy", "worst_window_accuracy", "negative_transfer_rate_0p5"
    ]
    cells = (
        frame.groupby(["dataset", "experiment_variant", "window_size"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in metrics}, seeds=("seed", "nunique"))
    )
    reference = (
        cells.loc[cells.experiment_variant.eq("full"), ["dataset", "window_size"] + metrics]
        .rename(columns={metric: f"full_{metric}" for metric in metrics})
    )
    effects = cells.merge(reference, on=["dataset", "window_size"], validate="many_to_one")
    for metric in metrics:
        effects[f"delta_{metric}_pp"] = 100.0 * (
            effects[metric] - effects[f"full_{metric}"]
        )

    dataset_summary = (
        effects.groupby(["dataset", "experiment_variant"], as_index=False)
        .agg(
            delta_mean_accuracy_pp=("delta_mean_accuracy_pp", "mean"),
            delta_negative_transfer_rate_0p5_pp=(
                "delta_negative_transfer_rate_0p5_pp", "mean"
            ),
            delta_worst_window_accuracy_pp=("delta_worst_window_accuracy_pp", "mean"),
            cells=("window_size", "nunique"),
        )
    )
    global_summary = (
        effects.groupby("experiment_variant", as_index=False)
        .agg(
            delta_mean_accuracy_pp=("delta_mean_accuracy_pp", "mean"),
            delta_negative_transfer_rate_0p5_pp=(
                "delta_negative_transfer_rate_0p5_pp", "mean"
            ),
            delta_worst_window_accuracy_pp=("delta_worst_window_accuracy_pp", "mean"),
            cells=("window_size", "size"),
            datasets=("dataset", "nunique"),
        )
    )

    statistics = []
    for variant, subset in effects.loc[~effects.experiment_variant.eq("full")].groupby(
        "experiment_variant"
    ):
        for metric in metrics:
            delta_column = f"delta_{metric}_pp"
            statistics.append(
                {
                    "experiment_variant": variant,
                    "metric": metric,
                    "cells": len(subset),
                    "mean_delta_pp": float(subset[delta_column].mean()),
                    "raw_p": safe_wilcoxon(subset[delta_column]),
                }
            )
    statistics_frame = pd.DataFrame(statistics)
    statistics_frame["holm_p"] = holm_adjust(statistics_frame.raw_p.tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_cell_effects.csv": effects,
        "per_dataset_summary.csv": dataset_summary,
        "global_summary.csv": global_summary,
        "paired_statistics.csv": statistics_frame,
    }
    for filename, output in outputs.items():
        output.to_csv(args.output_dir / filename, index=False)
    write_table(
        global_summary, statistics_frame, args.output_dir / "ablation_compact.tex"
    )
    write_heatmap(
        dataset_summary,
        args.output_dir / "ablation_heatmap.pdf",
        args.output_dir / "ablation_heatmap.png",
    )
    auxiliary = ["ablation_compact.tex", "ablation_heatmap.pdf", "ablation_heatmap.png"]
    manifest = {
        "aggregate_dirs": [str(path) for path in args.aggregate_dir],
        "statistical_unit": "dataset by window-size cell after seed averaging",
        "output_sha256": {},
    }
    manifest["output_sha256"].update(
        {filename: sha256(args.output_dir / filename) for filename in outputs}
    )
    manifest["output_sha256"].update(
        {filename: sha256(args.output_dir / filename) for filename in auxiliary}
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(global_summary.to_string(index=False))
    print(statistics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
