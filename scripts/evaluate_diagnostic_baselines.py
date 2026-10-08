"""Audit mechanism diagnosis against generic unlabeled scalar scores.

The script is intentionally read-only with respect to formal experiment
directories.  It reconstructs each controlled stream from the hashed input
paths and seed recorded in ``metadata.json``, then compares ShiftRoute's two
diagnostic coordinates with frozen-source predictive entropy and an unprojected
global feature-mean distance.  Labels are used only for offline AUROC scoring.
For each scalar baseline we report the better of both score orientations; this
gives the generic baseline the benefit of a label-informed sign choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

from shiftroute.experiment import (
    FeatureBank,
    StreamPhase,
    build_stream,
    long_tailed_prior,
)
from shiftroute.router import Action


def parse_input(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must have the form LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("input label cannot be empty")
    return label.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_bank(path: Path, name: str) -> FeatureBank:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return FeatureBank(
        name=name,
        features=artifact["features"].float(),
        logits=artifact["logits"].float(),
        labels=artifact["labels"].long(),
    )


def source_mean(path: Path, cache: Dict[Path, torch.Tensor]) -> torch.Tensor:
    resolved = path.resolve()
    if resolved not in cache:
        artifact = torch.load(resolved, map_location="cpu", weights_only=False)
        features = F.normalize(artifact["features"].float(), dim=-1)
        cache[resolved] = F.normalize(features.mean(dim=0), dim=0)
    return cache[resolved]


def phases(classes: int, phase_windows: int, imbalance_ratio: float, protocol: str):
    uniform = torch.full((classes,), 1.0 / classes)
    forward = long_tailed_prior(classes, imbalance_ratio)
    backward = long_tailed_prior(classes, imbalance_ratio, reverse=True)
    if protocol == "prior_development":
        return [
            StreamPhase("clean_start", "clean", phase_windows, Action.FREEZE, uniform),
            StreamPhase("prior_forward", "clean", phase_windows, Action.PRIOR_ONLY, forward),
            StreamPhase("clean_middle", "clean", phase_windows, Action.FREEZE, uniform),
            StreamPhase("prior_backward", "clean", phase_windows, Action.PRIOR_ONLY, backward),
            StreamPhase("clean_end", "clean", phase_windows, Action.FREEZE, uniform),
        ]
    return [
        StreamPhase("clean_start", "clean", phase_windows, Action.FREEZE, uniform),
        StreamPhase("prior_forward", "clean", phase_windows, Action.PRIOR_ONLY, forward),
        StreamPhase("clean_recovery", "clean", phase_windows, Action.FREEZE, uniform),
        StreamPhase(
            "representation_a", "representation_a", phase_windows,
            Action.REPRESENTATION_ONLY, uniform,
        ),
        StreamPhase(
            "compound_forward", "representation_a", phase_windows,
            Action.JOINT, forward,
        ),
        StreamPhase(
            "representation_b", "representation_b", phase_windows,
            Action.REPRESENTATION_ONLY, uniform,
        ),
        StreamPhase(
            "compound_backward", "representation_b", phase_windows,
            Action.JOINT, backward,
        ),
        StreamPhase("clean_end", "clean", phase_windows, Action.FREEZE, uniform),
    ]


def oriented_auc(target: Sequence[int], score: Sequence[float]) -> float:
    """Return the better AUROC after granting the scalar an optimal sign."""

    auc = float(roc_auc_score(target, score))
    return max(auc, 1.0 - auc)


def mechanism_targets(actions: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Map recorded enum values to prior-bearing and representation-bearing labels."""

    prior = actions.isin([Action.PRIOR_ONLY.value, Action.JOINT.value]).astype(int)
    representation = actions.isin(
        [Action.REPRESENTATION_ONLY.value, Action.JOINT.value]
    ).astype(int)
    return prior, representation


def condition_id(relative: Path) -> str:
    rendered = relative.as_posix()
    return re.sub(r"([_-])?seed\d+", "", rendered).rstrip("_-")


def score_windows(windows, global_source_mean: torch.Tensor) -> pd.DataFrame:
    rows = []
    for window in windows:
        log_probabilities = window.logits.float().log_softmax(dim=-1)
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
        normalized = F.normalize(window.features.float(), dim=-1)
        window_mean = F.normalize(normalized.mean(dim=0), dim=0)
        distance = 1.0 - torch.dot(window_mean, global_source_mean)
        rows.append(
            {
                "window": window.window_index,
                "phase": window.phase_name,
                "true_action": window.true_action.value,
                "source_entropy": float(entropy),
                "raw_feature_distance": float(distance),
            }
        )
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    methods = [
        "ShiftRoute coordinate",
        "Frozen-source entropy",
        "Raw feature-mean distance",
    ]
    pivot = summary.pivot(
        index=["dataset", "axis", "condition_cells"],
        columns="method",
        values="mean_auroc",
    ).reset_index()
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Offline mechanism-discrimination audit. Generic scalar baselines are granted their label-informed optimal score orientation. Values are AUROC after seed averaging within each condition cell.}",
        r"\label{tab:diagnostic-baselines}",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Dataset & Mechanism & Cells & ShiftRoute & Entropy & Raw dist. \\",
        r"\midrule",
    ]
    for _, row in pivot.iterrows():
        values = [float(row[method]) for method in methods]
        best = int(np.argmax(values))
        rendered = [f"{value:.3f}" for value in values]
        rendered[best] = rf"\textbf{{{rendered[best]}}}"
        axis = str(row["axis"]).replace("-bearing shift", "")
        lines.append(
            f"{row['dataset']} & {axis} & {int(row['condition_cells'])} & "
            f"{rendered[0]} & {rendered[1]} & {rendered[2]} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    datasets = list(dict.fromkeys(summary.dataset.tolist()))
    methods = [
        "ShiftRoute coordinate",
        "Frozen-source entropy",
        "Raw feature-mean distance",
    ]
    labels = ["ShiftRoute", "Entropy", "Raw distance"]
    colors = ["#1764ab", "#f28e2b", "#59a14f"]
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(max(6.4, 4.6 * len(datasets)), 3.4),
        sharey=True, squeeze=False,
    )
    for axis_object, dataset in zip(axes[0], datasets):
        subset = summary[summary.dataset.eq(dataset)]
        x = np.arange(2, dtype=float)
        width = 0.24
        for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
            values = []
            for target in ("prior-bearing shift", "representation-bearing shift"):
                selected = subset[subset.axis.eq(target) & subset.method.eq(method)]
                values.append(float(selected.mean_auroc.iloc[0]))
            axis_object.bar(
                x + (index - 1) * width, values, width=width,
                label=label, color=color, edgecolor="white", linewidth=0.6,
            )
        axis_object.set_title(dataset)
        axis_object.set_xticks(x, ["Prior", "Representation"])
        axis_object.set_ylim(0.5, 1.01)
        axis_object.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis_object.set_axisbelow(True)
    axes[0, 0].set_ylabel("Orientation-favorable AUROC")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 1.04), frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "diagnostic_baseline_auroc.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "diagnostic_baseline_auroc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_result_sentence(comparisons: pd.DataFrame, path: Path) -> None:
    positive = comparisons.mean_auroc_difference > 0
    significant = comparisons.holm_p < 0.05
    lower = float(comparisons.mean_auroc_difference.min())
    upper = float(comparisons.mean_auroc_difference.max())
    sentence = (
        f"The mechanism-specific ShiftRoute coordinate exceeded the generic scalar "
        f"baseline in {int(positive.sum())}/{len(comparisons)} dataset--axis comparisons; "
        f"{int((positive & significant).sum())}/{len(comparisons)} remained significant "
        f"after Holm correction, with mean AUROC differences from {lower:.3f} to "
        f"{upper:.3f}. Baseline score direction was selected post hoc using labels for "
        f"this audit, while labels were never exposed to the controller."
    )
    path.write_text(sentence + "\n", encoding="utf-8")


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    specs = []
    for dataset, root in args.input:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for metadata_path in sorted(root.rglob("metadata.json")):
            metrics_path = metadata_path.with_name("window_metrics.csv")
            if not metrics_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            arguments = metadata.get("arguments", {})
            required = ("source", "clean", "representation_a", "representation_b")
            if any(not arguments.get(key) for key in required):
                continue
            specs.append(
                {
                    "dataset": dataset,
                    "root": root,
                    "metadata": metadata_path,
                    "metrics": metrics_path,
                    "relative": metadata_path.parent.relative_to(root),
                    "arguments": arguments,
                }
            )
    if not specs:
        raise RuntimeError("no formal cached experiments were found")

    groups: Dict[Tuple[object, ...], List[dict]] = {}
    for spec in specs:
        a = spec["arguments"]
        key = (
            a["source"], a["clean"], a["representation_a"], a["representation_b"],
            int(a["window_size"]), int(a["phase_windows"]),
            float(a["imbalance_ratio"]), a.get("stream_protocol", "full"),
        )
        groups.setdefault(key, []).append(spec)

    source_cache: Dict[Path, torch.Tensor] = {}
    run_rows: List[dict] = []
    for key, group in groups.items():
        source_path, clean_path, rep_a_path, rep_b_path, window_size, phase_count, ratio, protocol = key
        clean = load_bank(Path(clean_path), "clean")
        rep_a = load_bank(Path(rep_a_path), "representation_a")
        rep_b = load_bank(Path(rep_b_path), "representation_b")
        banks = {"clean": clean, "representation_a": rep_a, "representation_b": rep_b}
        stream_phases = phases(clean.classes, phase_count, ratio, protocol)
        global_mean = source_mean(Path(source_path), source_cache)
        for spec in group:
            seed = int(spec["arguments"]["seed"])
            windows = build_stream(banks, stream_phases, window_size=window_size, seed=seed)
            generic = score_windows(windows, global_mean)
            formal = pd.read_csv(spec["metrics"])
            formal = formal.loc[formal.method == "shiftroute", [
                "window", "phase", "true_action", "prior_score", "representation_score"
            ]].copy()
            merged = formal.merge(
                generic, on=["window", "phase", "true_action"], how="inner", validate="one_to_one"
            )
            if len(merged) != len(windows):
                raise RuntimeError(f"stream reconstruction mismatch: {spec['metrics']}")
            prior_target, rep_target = mechanism_targets(merged.true_action)
            scores = {
                "ShiftRoute coordinate": (
                    merged.prior_score.to_numpy(), merged.representation_score.to_numpy()
                ),
                "Frozen-source entropy": (
                    merged.source_entropy.to_numpy(), merged.source_entropy.to_numpy()
                ),
                "Raw feature-mean distance": (
                    merged.raw_feature_distance.to_numpy(),
                    merged.raw_feature_distance.to_numpy(),
                ),
            }
            for method, (prior_score, rep_score) in scores.items():
                run_rows.extend(
                    [
                        {
                            "dataset": spec["dataset"],
                            "condition": condition_id(spec["relative"]),
                            "seed": seed,
                            "axis": "prior-bearing shift",
                            "method": method,
                            "auroc": oriented_auc(prior_target, prior_score),
                        },
                        {
                            "dataset": spec["dataset"],
                            "condition": condition_id(spec["relative"]),
                            "seed": seed,
                            "axis": "representation-bearing shift",
                            "method": method,
                            "auroc": oriented_auc(rep_target, rep_score),
                        },
                    ]
                )

    run_frame = pd.DataFrame(run_rows)
    condition_frame = (
        run_frame.groupby(["dataset", "condition", "axis", "method"], as_index=False)
        .agg(mean_auroc=("auroc", "mean"), seeds=("seed", "nunique"))
    )
    summary = (
        condition_frame.groupby(["dataset", "axis", "method"], as_index=False)
        .agg(
            mean_auroc=("mean_auroc", "mean"),
            sd_across_conditions=("mean_auroc", "std"),
            condition_cells=("condition", "nunique"),
        )
    )

    comparisons = []
    for (dataset, axis), frame in condition_frame.groupby(["dataset", "axis"]):
        pivot = frame.pivot(index="condition", columns="method", values="mean_auroc").dropna()
        ours = pivot["ShiftRoute coordinate"]
        for baseline in ("Frozen-source entropy", "Raw feature-mean distance"):
            delta = ours - pivot[baseline]
            if len(delta) < 2 or np.allclose(delta.to_numpy(), 0.0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(delta.to_numpy(), alternative="two-sided").pvalue)
            comparisons.append(
                {
                    "dataset": dataset,
                    "axis": axis,
                    "baseline": baseline,
                    "condition_cells": len(delta),
                    "mean_auroc_difference": float(delta.mean()),
                    "wins": int((delta > 0).sum()),
                    "ties": int(np.isclose(delta, 0.0).sum()),
                    "raw_p": p_value,
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["holm_p"] = holm_adjust(comparison_frame.raw_p.tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_run_auc.csv": run_frame,
        "per_condition_auc.csv": condition_frame,
        "summary.csv": summary,
        "comparisons.csv": comparison_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
    write_latex_table(summary, args.output_dir / "diagnostic_baselines.tex")
    write_figure(summary, args.output_dir)
    write_result_sentence(
        comparison_frame, args.output_dir / "diagnostic_baseline_sentence.tex"
    )
    auxiliary_outputs = [
        "diagnostic_baselines.tex",
        "diagnostic_baseline_sentence.tex",
        "diagnostic_baseline_auroc.pdf",
        "diagnostic_baseline_auroc.png",
    ]
    manifest = {
        "inputs": [{"label": label, "path": str(path)} for label, path in args.input],
        "orientation_policy": "max(AUROC, 1-AUROC) for every scalar score",
        "labels_used_by_controller": False,
        "labels_used_for_offline_audit_only": True,
        "output_sha256": {
            filename: sha256(args.output_dir / filename) for filename in outputs
        },
    }
    manifest["output_sha256"].update(
        {filename: sha256(args.output_dir / filename) for filename in auxiliary_outputs}
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
