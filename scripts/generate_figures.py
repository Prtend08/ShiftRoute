"""Generate publication-quality ShiftRoute figures from aggregated real runs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


ACTION_ORDER = ["freeze", "prior_only", "representation_only", "joint"]
ACTION_PALETTE = {
    "freeze": "#4C78A8",
    "prior_only": "#F58518",
    "representation_only": "#54A24B",
    "joint": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scatter-points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, manifest: Dict[str, object]) -> None:
    outputs = []
    for suffix, options in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **options)
        outputs.append({"path": str(path), "sha256": sha256(path)})
    manifest.setdefault("figures", []).append({"stem": stem, "outputs": outputs})
    plt.close(fig)


def read_required(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ordered_present(values: Iterable[str]) -> List[str]:
    present = set(values)
    return [name for name in ACTION_ORDER if name in present]


def diagnostic_scatter(
    routed: pd.DataFrame,
    output_dir: Path,
    max_points: int,
    seed: int,
    manifest: Dict[str, object],
) -> None:
    required = {"prior_score", "representation_score", "true_action", "action_applied"}
    if not required.issubset(routed.columns):
        return
    plot = routed.dropna(subset=["prior_score", "representation_score"]).copy()
    if plot.empty:
        return
    if {"prior_enter", "representation_enter"}.issubset(plot.columns):
        prior_denominator = plot.prior_enter.replace(0, np.nan)
        representation_denominator = plot.representation_enter.replace(0, np.nan)
        plot["prior_axis"] = plot.prior_score / prior_denominator
        plot["representation_axis"] = plot.representation_score / representation_denominator
        x_label = r"Normalized prior score $D_t/\tau_D$"
        y_label = r"Normalized representation score $R_t/\tau_R$"
        threshold = 1.0
    else:
        plot["prior_axis"] = plot.prior_score
        plot["representation_axis"] = plot.representation_score
        x_label = "Prior-shift score"
        y_label = "Representation-shift score"
        threshold = None
    if len(plot) > max_points:
        plot = plot.sample(max_points, random_state=seed)
    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    for action in ordered_present(plot.true_action.astype(str)):
        subset = plot[plot.true_action == action]
        correct = subset.action_applied == subset.true_action
        ax.scatter(
            subset.loc[correct, "prior_axis"],
            subset.loc[correct, "representation_axis"],
            s=11,
            alpha=0.50,
            c=ACTION_PALETTE[action],
            edgecolors="none",
            label=action.replace("_", " "),
        )
        ax.scatter(
            subset.loc[~correct, "prior_axis"],
            subset.loc[~correct, "representation_axis"],
            s=18,
            alpha=0.75,
            facecolors="none",
            edgecolors=ACTION_PALETTE[action],
            linewidths=0.7,
        )
    if threshold is not None:
        ax.axvline(threshold, color="0.25", linestyle="--", linewidth=0.9)
        ax.axhline(threshold, color="0.25", linestyle="--", linewidth=0.9)
    ax.set(xlabel=x_label, ylabel=y_label, title="Two-axis diagnosis and applied actions")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="0.35", markersize=4.5))
    labels.append("open: misrouted")
    ax.legend(handles, labels, frameon=True, framealpha=0.86, edgecolor="none", ncol=2, loc="upper left", fontsize=6.3, handletextpad=0.35, columnspacing=0.7, borderpad=0.3)
    save_figure(fig, output_dir, "diagnostic_scatter", manifest)


def confusion_heatmap(
    confusion: pd.DataFrame,
    output_dir: Path,
    manifest: Dict[str, object],
    *,
    stem: str = "action_confusion",
    title: str = "Action-routing confusion",
) -> None:
    if confusion.empty:
        return
    datasets = sorted(confusion.dataset.unique())
    # Four columns keep the 11-suite audit landscape-oriented and readable when
    # placed beside the diagnostic scatter in a full-width manuscript figure.
    grid_columns = min(4, len(datasets))
    rows = math.ceil(len(datasets) / grid_columns)
    fig, axes = plt.subplots(rows, grid_columns, figsize=(3.05 * grid_columns, 2.55 * rows), squeeze=False)
    flat_axes = list(axes.flat)
    for index, (ax, dataset) in enumerate(zip(flat_axes, datasets)):
        subset = confusion[confusion.dataset == dataset]
        matrix = subset.pivot(index="true_action", columns="action_applied", values="row_fraction")
        rows = ordered_present(matrix.index.astype(str))
        columns = ordered_present(matrix.columns.astype(str))
        matrix = matrix.reindex(index=rows, columns=columns, fill_value=0.0)
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            square=True,
            cbar=index == len(datasets) - 1,
            cbar_kws={"label": "Fraction"},
        )
        ax.set_title(str(dataset))
        ax.set_xlabel("Applied action")
        ax.set_ylabel("Required action")
        ax.set_xticklabels([item.get_text().replace("_", "\n") for item in ax.get_xticklabels()])
        ax.set_yticklabels([item.get_text().replace("_", " ") for item in ax.get_yticklabels()], rotation=0)
    for ax in flat_axes[len(datasets):]:
        ax.axis("off")
    fig.suptitle(title, y=1.02)
    save_figure(fig, output_dir, stem, manifest)


def representative_timelines(routed: pd.DataFrame, output_dir: Path, manifest: Dict[str, object]) -> None:
    required = {"dataset", "run_id", "window", "accuracy", "source_accuracy", "phase", "gain_over_source"}
    if not required.issubset(routed.columns):
        return
    preferred = {
        "CIFAR10-C-raw",
        "CIFAR100-C-raw",
        "ImageNet-C-ConvNeXt",
        "ImageNet100",
    }
    selections = []
    for dataset, subset in routed.groupby("dataset"):
        if dataset not in preferred:
            continue
        gains = subset.groupby("run_id").gain_over_source.mean().sort_values()
        if gains.empty:
            continue
        selected = gains.index[(len(gains) - 1) // 2]
        selections.append((dataset, selected, float(gains.loc[selected])))
    if not selections:
        return
    fig, axes = plt.subplots(
        len(selections), 1, figsize=(7.2, 1.8 * len(selections)), squeeze=False,
        constrained_layout=True,
    )
    for ax, (dataset, run_id, mean_gain) in zip(axes.flat, selections):
        plot = routed[(routed.dataset == dataset) & (routed.run_id == run_id)].sort_values("window")
        ax.plot(plot.window, plot.source_accuracy, color="0.45", linewidth=1.0, label="Source")
        ax.plot(plot.window, plot.accuracy, color="#D62728", linewidth=1.35, label="ShiftRoute")
        boundaries = plot.phase.ne(plot.phase.shift()).cumsum()
        phase_abbreviations = {
            "clean_start": "clean",
            "prior_forward": "prior +",
            "clean_recovery": "recover",
            "representation_outbound": "repr. A",
            "representation_forward": "repr. A",
            "compound_forward": "compound +",
            "representation_inbound": "repr. B",
            "compound_backward": "compound -",
            "prior_backward": "prior -",
            "clean_end": "clean",
        }
        for index, (_, phase_rows) in enumerate(plot.groupby(boundaries)):
            if index % 2:
                ax.axvspan(phase_rows.window.min() - 0.5, phase_rows.window.max() + 0.5, color="0.93", zorder=0)
            center = (phase_rows.window.min() + phase_rows.window.max()) / 2
            phase_name = str(phase_rows.phase.iloc[0])
            ax.text(center, 0.985, phase_abbreviations.get(phase_name, phase_name.replace("_", " ")), fontsize=5.2, ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.set_title(f"{dataset}: median-gain run ({mean_gain * 100:+.2f} pp)", fontsize=8.5)
        ax.set_ylabel("Window accuracy")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, ncol=2, loc="lower left")
        manifest.setdefault("representative_runs", []).append(
            {"dataset": str(dataset), "run_id": str(run_id), "mean_gain": mean_gain, "selection": "lower median by mean gain"}
        )
    axes.flat[-1].set_xlabel("Prequential window")
    save_figure(fig, output_dir, "stream_timeline", manifest)


def phase_heatmap(phase: pd.DataFrame, output_dir: Path, manifest: Dict[str, object]) -> None:
    if not {"dataset", "phase", "method", "gain_over_source"}.issubset(phase.columns):
        return
    routed = phase[phase.method.isin(["shiftroute", "shiftroute_tent"])]
    if routed.empty:
        return
    table = routed.pivot_table(index="phase", columns="dataset", values="gain_over_source", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(max(3.4, 1.15 * table.shape[1] + 2.2), max(2.7, 0.35 * table.shape[0] + 1.5)))
    bound = max(0.1, float(np.nanmax(np.abs(table.to_numpy()))))
    sns.heatmap(table, cmap="vlag", center=0, vmin=-bound, vmax=bound, annot=True, fmt="+.2f", cbar_kws={"label": "Gain over source (pp)"}, ax=ax)
    ax.set_title("Phase-wise accuracy gain")
    ax.set_xlabel("")
    ax.set_ylabel("Stream phase")
    ax.set_yticklabels([item.get_text().replace("_", " ") for item in ax.get_yticklabels()], rotation=0)
    save_figure(fig, output_dir, "phase_gain_heatmap", manifest)


def negative_transfer_plot(routed: pd.DataFrame, output_dir: Path, manifest: Dict[str, object]) -> None:
    if not {"dataset", "gain_over_source"}.issubset(routed.columns):
        return
    plot = routed.dropna(subset=["gain_over_source"]).copy()
    if plot.empty:
        return
    plot["gain_pp"] = plot.gain_over_source * 100
    fig, ax = plt.subplots(figsize=(max(3.45, 0.75 * plot.dataset.nunique() + 2.5), 2.8))
    sns.boxplot(data=plot, x="dataset", y="gain_pp", color="#4C78A8", width=0.55, showfliers=False, ax=ax)
    sns.stripplot(data=plot, x="dataset", y="gain_pp", color="0.15", alpha=0.18, size=1.8, jitter=0.22, ax=ax)
    ax.axhline(0, color="0.25", linestyle="--", linewidth=0.9)
    ax.axhline(-0.5, color="#D62728", linestyle=":", linewidth=0.8)
    ax.set(title="Window-level negative-transfer audit", xlabel="", ylabel="Gain over source (pp)")
    ax.tick_params(axis="x", labelrotation=28, labelsize=7)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    save_figure(fig, output_dir, "negative_transfer", manifest)


def performance_forest(ci: pd.DataFrame, output_dir: Path, manifest: Dict[str, object]) -> None:
    required = {"dataset", "method", "metric", "mean", "ci95_lower", "ci95_upper"}
    if not required.issubset(ci.columns):
        return
    plot = ci[ci.metric == "mean_accuracy"].dropna(subset=["mean"]).copy()
    if plot.empty:
        return
    plot["label"] = plot.dataset.astype(str) + " | " + plot.method.astype(str).str.replace("_", " ")
    plot = plot.sort_values(["dataset", "mean"]).reset_index(drop=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(5.3, max(2.8, 0.21 * len(plot) + 1.0)))
    errors = np.vstack([plot["mean"] - plot.ci95_lower, plot.ci95_upper - plot["mean"]])
    ax.errorbar(plot["mean"], y, xerr=errors, fmt="o", markersize=3.5, color="#4C78A8", ecolor="0.45", capsize=2, linewidth=0.8)
    ax.set_yticks(y, plot.label)
    ax.set(xlabel="Mean window accuracy (95% bootstrap CI)", ylabel="", title="Aggregate predictive performance")
    save_figure(fig, output_dir, "performance_forest", manifest)


def severity_curve(routed: pd.DataFrame, output_dir: Path, manifest: Dict[str, object]) -> None:
    if not {"dataset", "severity", "gain_over_source"}.issubset(routed.columns):
        return
    plot = routed.dropna(subset=["severity", "gain_over_source"])
    if plot.empty:
        return
    grouped = plot.groupby(["dataset", "severity"]).gain_over_source.agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for dataset, subset in grouped.groupby("dataset"):
        subset = subset.sort_values("severity")
        ax.errorbar(
            subset.severity,
            subset["mean"] * 100,
            yerr=subset["sem"].fillna(0) * 100,
            marker="o",
            markersize=3.5,
            capsize=2,
            linewidth=1.1,
            label=str(dataset),
        )
    ax.axhline(0, color="0.25", linestyle="--", linewidth=0.9)
    ax.set_xticks(sorted(grouped.severity.unique()))
    ax.set(xlabel="Corruption severity", ylabel="Gain over source (pp)", title="Severity stress test")
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "severity_curve", manifest)


def efficiency_pareto(run_summary: pd.DataFrame, output_dir: Path, manifest: Dict[str, object]) -> None:
    required = {"dataset", "method", "mean_accuracy", "mean_prediction_seconds", "mean_update_seconds"}
    if not required.issubset(run_summary.columns):
        return
    plot = run_summary.dropna(subset=["mean_prediction_seconds", "mean_update_seconds"]).copy()
    if plot.empty:
        return
    plot["latency_ms"] = (plot.mean_prediction_seconds + plot.mean_update_seconds) * 1000
    grouped = plot.groupby(["dataset", "method"]).agg(accuracy=("mean_accuracy", "mean"), latency_ms=("latency_ms", "mean")).reset_index()
    datasets = list(grouped.dataset.astype(str).drop_duplicates())
    columns = min(2, len(datasets))
    rows = int(np.ceil(len(datasets) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.55 * columns, 3.25 * rows),
        squeeze=False,
        sharex=True,
    )
    method_names = list(grouped.method.astype(str).drop_duplicates())
    markers = ["o", "s", "^", "D", "P", "X", "v", "h", "*"]
    marker_by_method = {
        method: markers[index % len(markers)] for index, method in enumerate(method_names)
    }
    handles = {}
    for axis, dataset in zip(axes.flat, datasets):
        subset_dataset = grouped[grouped.dataset.astype(str) == dataset]
        for row in subset_dataset.itertuples():
            label = str(row.method).replace("_", " ")
            artist = axis.scatter(
                row.latency_ms,
                row.accuracy,
                s=34,
                marker=marker_by_method[str(row.method)],
                label=label,
            )
            handles[label] = artist
        axis.set_xscale("log")
        axis.set_title(dataset, fontsize=9)
        axis.set_xlabel("Latency / window (ms, log)", fontsize=7.5)
        axis.set_ylabel("Mean accuracy")
    for axis in list(axes.flat)[len(datasets):]:
        axis.set_visible(False)
    fig.suptitle("Accuracy--efficiency trade-off", fontsize=10)
    fig.legend(
        handles.values(),
        handles.keys(),
        frameon=False,
        fontsize=6.0,
        ncol=min(5, len(handles)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
    )
    fig.subplots_adjust(bottom=0.14, top=0.68 if rows == 1 else 0.78, wspace=0.28)
    save_figure(fig, output_dir, "efficiency_pareto", manifest)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    routed_path = args.aggregate_dir / "routed_windows.csv.gz"
    input_paths = [
        args.aggregate_dir / "run_summary.csv",
        args.aggregate_dir / "phase_summary.csv",
        args.aggregate_dir / "method_bootstrap_ci.csv",
        args.aggregate_dir / "action_confusion.csv",
        routed_path,
    ]
    run_summary = read_required(args.aggregate_dir, "run_summary.csv")
    phase = read_required(args.aggregate_dir, "phase_summary.csv")
    ci = read_required(args.aggregate_dir, "method_bootstrap_ci.csv")
    confusion = read_required(args.aggregate_dir, "action_confusion.csv")
    steady_confusion_path = args.aggregate_dir / "action_confusion_steady.csv"
    steady_confusion = (
        pd.read_csv(steady_confusion_path)
        if steady_confusion_path.exists()
        else pd.DataFrame()
    )
    if steady_confusion_path.exists():
        input_paths.append(steady_confusion_path)
    routed = pd.read_csv(routed_path)
    manifest: Dict[str, object] = {
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in input_paths],
        "seed": args.seed,
        "note": "All figures are generated from aggregated experiment CSVs; no values are imputed or manually entered.",
    }
    diagnostic_scatter(routed, args.output_dir, args.max_scatter_points, args.seed, manifest)
    confusion_heatmap(confusion, args.output_dir, manifest)
    confusion_heatmap(
        steady_confusion,
        args.output_dir,
        manifest,
        stem="action_confusion_steady",
        title="Steady-state routing (transition buffer excluded)",
    )
    representative_timelines(routed, args.output_dir, manifest)
    phase_heatmap(phase, args.output_dir, manifest)
    negative_transfer_plot(routed, args.output_dir, manifest)
    performance_forest(ci, args.output_dir, manifest)
    severity_curve(routed, args.output_dir, manifest)
    efficiency_pareto(run_summary, args.output_dir, manifest)
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(manifest.get("figures", [])), "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
