"""Create identity-disjoint calibration/evaluation banks for corruption suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-name", default="source_train.pt")
    parser.add_argument("--development-name", default="calibration_train.pt")
    parser.add_argument("--condition-glob", default="*_s[1-5].pt")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def subset_artifact(artifact: dict, selection: torch.Tensor, partition: str) -> dict:
    samples = int(artifact["labels"].numel())
    output = dict(artifact)
    for key in ("features", "logits", "labels", "indices"):
        value = artifact.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == samples:
            output[key] = value[selection]
    if "indices" not in output:
        output["indices"] = selection.clone()
    output["identity_partition"] = partition
    output["parent_samples"] = samples
    return output


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must lie in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "identity_split_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(json.dumps({"skip_existing": str(manifest_path)}), flush=True)
        return

    clean_path = args.input_dir / "test_clean.pt"
    clean = torch.load(clean_path, map_location="cpu", weights_only=False)
    labels = clean["labels"].long()
    rng = np.random.default_rng(args.seed)
    calibration_indices = []
    evaluation_indices = []
    for class_index in torch.unique(labels, sorted=True).tolist():
        candidates = torch.nonzero(labels == int(class_index), as_tuple=True)[0].numpy()
        rng.shuffle(candidates)
        count = int(round(len(candidates) * args.calibration_fraction))
        calibration_indices.extend(candidates[:count].tolist())
        evaluation_indices.extend(candidates[count:].tolist())
    rng.shuffle(calibration_indices)
    rng.shuffle(evaluation_indices)
    calibration_indices = torch.tensor(calibration_indices, dtype=torch.long)
    evaluation_indices = torch.tensor(evaluation_indices, dtype=torch.long)

    overlap = set(calibration_indices.tolist()).intersection(evaluation_indices.tolist())
    if overlap or len(calibration_indices) + len(evaluation_indices) != len(labels):
        raise RuntimeError("identity partition is not exhaustive and disjoint")

    source_destination = args.output_dir / "source_train.pt"
    development_destination = args.output_dir / "development_train.pt"
    link_or_copy(args.input_dir / args.source_name, source_destination)
    link_or_copy(args.input_dir / args.development_name, development_destination)

    conformal_path = args.output_dir / "conformal_clean.pt"
    evaluation_clean_path = args.output_dir / "test_clean.pt"
    torch.save(subset_artifact(clean, calibration_indices, "conformal"), conformal_path)
    torch.save(subset_artifact(clean, evaluation_indices, "evaluation"), evaluation_clean_path)

    outputs = [
        source_destination,
        development_destination,
        conformal_path,
        evaluation_clean_path,
    ]
    for input_path in sorted(args.input_dir.glob(args.condition_glob)):
        artifact = torch.load(input_path, map_location="cpu", weights_only=False)
        if not torch.equal(artifact["labels"].long(), labels):
            raise ValueError(f"label order differs from clean bank: {input_path}")
        output_path = args.output_dir / input_path.name
        torch.save(subset_artifact(artifact, evaluation_indices, "evaluation"), output_path)
        outputs.append(output_path)

    split_record = {
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "parent_clean_sha256": sha256(clean_path),
        "calibration_samples": len(calibration_indices),
        "evaluation_samples": len(evaluation_indices),
        "identity_overlap": 0,
        "calibration_indices": calibration_indices.tolist(),
        "evaluation_indices": evaluation_indices.tolist(),
        "output_sha256": {path.name: sha256(path) for path in outputs},
    }
    manifest_path.write_text(json.dumps(split_record, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "calibration_samples": len(calibration_indices),
                "evaluation_samples": len(evaluation_indices),
                "conditions": len(outputs) - 4,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
