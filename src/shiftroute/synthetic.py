"""Small CPU smoke experiment for the ShiftRoute diagnostic."""

from __future__ import annotations

import json

import torch
from torch import Tensor
import torch.nn.functional as F

from .diagnostics import diagnose_window, fit_source_statistics


def make_prototypes(classes: int, dimension: int, *, seed: int = 0) -> Tensor:
    if dimension < classes:
        raise ValueError("dimension must be at least the number of classes")
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(dimension, classes, generator=generator)
    orthogonal, _ = torch.linalg.qr(basis)
    return orthogonal.T.contiguous()


def sample_features(
    prototypes: Tensor,
    class_prior: Tensor,
    samples: int,
    *,
    noise: float = 0.12,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.multinomial(class_prior, samples, replacement=True, generator=generator)
    perturbation = noise * torch.randn(
        samples,
        prototypes.shape[1],
        generator=generator,
    )
    features = F.normalize(prototypes[labels] + perturbation, dim=-1)
    return features, labels


def shifted_prototypes(prototypes: Tensor, strength: float, *, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(prototypes.shape, generator=generator)
    direction = direction - (direction * prototypes).sum(dim=-1, keepdim=True) * prototypes
    direction = F.normalize(direction, dim=-1)
    return F.normalize(prototypes + strength * direction, dim=-1)


def run_demo() -> dict[str, dict[str, object]]:
    classes, dimension = 4, 32
    prototypes = make_prototypes(classes, dimension)
    uniform = torch.full((classes,), 1.0 / classes)
    long_tail = torch.tensor([0.70, 0.15, 0.10, 0.05])

    source_features, source_labels = sample_features(
        prototypes,
        uniform,
        6000,
        seed=1,
    )
    source = fit_source_statistics(source_features, source_labels, kappa=45.0)
    representation_prototypes = shifted_prototypes(prototypes, 0.55, seed=2)

    scenarios = {
        "no_shift": sample_features(prototypes, uniform, 512, seed=10)[0],
        "prior_only": sample_features(prototypes, long_tail, 512, seed=11)[0],
        "representation_only": sample_features(
            representation_prototypes,
            uniform,
            512,
            seed=12,
        )[0],
        "compound": sample_features(
            representation_prototypes,
            long_tail,
            512,
            seed=13,
        )[0],
    }

    output: dict[str, dict[str, object]] = {}
    for name, features in scenarios.items():
        diagnosis = diagnose_window(features, source)
        output[name] = {
            "estimated_prior": [round(value, 4) for value in diagnosis.target_prior.tolist()],
            "prior_score": round(diagnosis.prior_score.item(), 4),
            "representation_z": round(diagnosis.representation_score.item(), 4),
            "confidence": round(diagnosis.mean_assignment_confidence.item(), 4),
        }
    return output


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))

