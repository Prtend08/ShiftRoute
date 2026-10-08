"""Select prior-expert stability controls using development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def numbers(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--phase-windows", type=int, default=6)
    parser.add_argument("--strengths", default="0,0.1,0.25,0.5")
    parser.add_argument("--momenta", default="0.5,0.8,0.9")
    parser.add_argument("--ratio-clips", default="2,5")
    parser.add_argument("--imbalance-ratios", default="10,50,100")
    parser.add_argument("--seeds", default="2026,2027")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strengths = numbers(args.strengths, float)
    momenta = numbers(args.momenta, float)
    ratio_clips = numbers(args.ratio_clips, float)
    ratios = numbers(args.imbalance_ratios, float)
    seeds = numbers(args.seeds, int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_cached_experiment.py")
    records = []
    for strength in strengths:
        for momentum in momenta:
            for ratio_clip in ratio_clips:
                for ratio in ratios:
                    for seed in seeds:
                        tag = (
                            f"s{strength:g}_m{momentum:g}_c{ratio_clip:g}"
                            f"_ir{ratio:g}_seed{seed}"
                        )
                        output = args.output_dir / tag
                        if not (output / "metadata.json").exists():
                            subprocess.run(
                                [
                                    sys.executable,
                                    runner,
                                    "--source", args.source,
                                    "--calibration", args.calibration,
                                    "--clean", args.development,
                                    "--representation-a", args.development,
                                    "--representation-b", args.development,
                                    "--output-dir", output,
                                    "--window-size", str(args.window_size),
                                    "--phase-windows", str(args.phase_windows),
                                    "--calibration-windows", "300",
                                    "--imbalance-ratio", str(ratio),
                                    "--seed", str(seed),
                                    "--event-log", "none",
                                    "--stream-protocol", "prior_development",
                                    "--prior-strength", str(strength),
                                    "--prior-momentum", str(momentum),
                                    "--prior-ratio-clip", str(ratio_clip),
                                ],
                                check=True,
                                stdout=subprocess.DEVNULL,
                            )
                        frame = pd.read_csv(output / "window_metrics.csv")
                        source = frame[frame.method == "source"].set_index("window")
                        routed = frame[frame.method == "shiftroute"].set_index("window")
                        prior_mask = routed.phase.str.startswith("prior")
                        clean_mask = routed.phase.str.startswith("clean")
                        prior_nll_delta = float(
                            (routed.loc[prior_mask, "nll"] - source.loc[prior_mask, "nll"]).mean()
                        )
                        clean_accuracy_loss = float(
                            (source.loc[clean_mask, "accuracy"] - routed.loc[clean_mask, "accuracy"])
                            .clip(lower=0.0)
                            .mean()
                        )
                        severe_rate = float(
                            (
                                routed.loc[clean_mask, "accuracy"]
                                - source.loc[clean_mask, "accuracy"]
                                < -0.02
                            ).mean()
                        )
                        records.append(
                            {
                                "strength": strength,
                                "momentum": momentum,
                                "ratio_clip": ratio_clip,
                                "imbalance_ratio": ratio,
                                "seed": seed,
                                "prior_nll_delta": prior_nll_delta,
                                "clean_accuracy_loss": clean_accuracy_loss,
                                "severe_clean_negative_transfer_rate": severe_rate,
                                "objective": prior_nll_delta
                                + 2.0 * clean_accuracy_loss
                                + 0.1 * severe_rate,
                            }
                        )
    details = pd.DataFrame(records)
    details.to_csv(args.output_dir / "development_grid.csv", index=False)
    aggregate = (
        details.groupby(["strength", "momentum", "ratio_clip"], as_index=False)
        .agg(
            objective=("objective", "mean"),
            prior_nll_delta=("prior_nll_delta", "mean"),
            clean_accuracy_loss=("clean_accuracy_loss", "mean"),
            severe_clean_negative_transfer_rate=(
                "severe_clean_negative_transfer_rate", "mean"
            ),
        )
        .sort_values("objective")
    )
    aggregate.to_csv(args.output_dir / "development_grid_aggregate.csv", index=False)
    chosen = aggregate.iloc[0].to_dict()
    (args.output_dir / "chosen.json").write_text(
        json.dumps(chosen, indent=2), encoding="utf-8"
    )
    print(json.dumps(chosen), flush=True)


if __name__ == "__main__":
    main()
