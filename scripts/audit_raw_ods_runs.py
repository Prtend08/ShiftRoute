"""Fail-closed audit for the ODS and exploratory ShiftRoute+ODS run matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_METHODS = {
    "source", "always_prior", "prequential_tent", "prequential_eta",
    "prequential_sar", "prequential_deyo", "prequential_ods",
    "prequential_joint", "shiftroute_tent", "oracle_tent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, action="append", required=True)
    parser.add_argument("--aux-root", type=Path, action="append", required=True)
    parser.add_argument("--expected-runs-per-root", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_root(root: Path, methods: set[str], expected_runs: int) -> dict:
    run_dirs = sorted(path.parent for path in root.glob("*/metadata.json"))
    if len(run_dirs) != expected_runs:
        raise RuntimeError(f"{root}: expected {expected_runs} runs, found {len(run_dirs)}")
    records = []
    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.json"
        summary_path = run_dir / "summary.csv"
        metrics_path = run_dir / "window_metrics.csv"
        if not summary_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"incomplete run: {run_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = pd.read_csv(summary_path)
        metrics = pd.read_csv(metrics_path)
        actual_methods = set(summary.method.astype(str))
        if actual_methods != methods or set(metrics.method.astype(str)) != methods:
            raise RuntimeError(
                f"{run_dir}: methods {sorted(actual_methods)} != {sorted(methods)}"
            )
        if metrics.window.nunique() != 48:
            raise RuntimeError(f"{run_dir}: expected 48 windows")
        numeric = metrics.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise RuntimeError(f"{run_dir}: non-finite window metric")
        if metadata.get("labels_used_for_controller_or_update") is not False:
            raise RuntimeError(f"{run_dir}: label-use audit failed")
        outputs = metadata.get("output_sha256", {})
        if outputs.get("summary") != sha256(summary_path):
            raise RuntimeError(f"{run_dir}: summary hash mismatch")
        if outputs.get("window_metrics") != sha256(metrics_path):
            raise RuntimeError(f"{run_dir}: window hash mismatch")
        for bank, check in metadata.get("cache_raw_consistency", {}).items():
            if check["max_absolute_logit_error"] > check["maximum_tolerance"]:
                raise RuntimeError(f"{run_dir}: cache/raw mismatch for {bank}")
        records.append(
            {
                "run": run_dir.name,
                "metadata_sha256": sha256(metadata_path),
                "summary_sha256": sha256(summary_path),
                "window_metrics_sha256": sha256(metrics_path),
            }
        )
    return {"root": str(root), "runs": len(records), "methods": sorted(methods), "records": records}


def main() -> None:
    args = parse_args()
    audit = {
        "base": [audit_root(root, BASE_METHODS, args.expected_runs_per_root) for root in args.base_root],
        "exploratory_auxiliary": [
            audit_root(root, {"shiftroute_ods"}, args.expected_runs_per_root)
            for root in args.aux_root
        ],
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "audited_roots": 4, "runs": 192}))


if __name__ == "__main__":
    main()
