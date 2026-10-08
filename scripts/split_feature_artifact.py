"""Create disjoint stratified feature artifacts for development-only tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    parser.add_argument("--fraction-a", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def subset(artifact, indices: torch.Tensor, parent: Path):
    output = dict(artifact)
    for key in ("features", "logits", "labels", "indices"):
        if key in artifact and len(artifact[key]) == len(artifact["labels"]):
            output[key] = artifact[key][indices]
    output["parent_artifact"] = str(parent)
    return output


def main() -> None:
    args = parse_args()
    if not 0.0 < args.fraction_a < 1.0:
        raise ValueError("fraction-a must lie in (0, 1)")
    artifact = torch.load(args.input, map_location="cpu", weights_only=False)
    labels = artifact["labels"].long()
    rng = np.random.default_rng(args.seed)
    first = []
    second = []
    for class_index in labels.unique(sorted=True).tolist():
        candidates = torch.nonzero(labels == class_index, as_tuple=False).flatten().numpy()
        rng.shuffle(candidates)
        cut = max(1, min(len(candidates) - 1, int(round(len(candidates) * args.fraction_a))))
        first.extend(candidates[:cut].tolist())
        second.extend(candidates[cut:].tolist())
    index_a = torch.as_tensor(sorted(first), dtype=torch.long)
    index_b = torch.as_tensor(sorted(second), dtype=torch.long)
    args.output_a.parent.mkdir(parents=True, exist_ok=True)
    args.output_b.parent.mkdir(parents=True, exist_ok=True)
    torch.save(subset(artifact, index_a, args.input), args.output_a)
    torch.save(subset(artifact, index_b, args.input), args.output_b)
    print({"a": len(index_a), "b": len(index_b)}, flush=True)


if __name__ == "__main__":
    main()
