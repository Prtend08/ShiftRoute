"""Lightweight action experts used by the ShiftRoute controller."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


def _probability(value: Tensor, eps: float = 1e-8) -> Tensor:
    value = value.float().clamp_min(eps)
    return value / value.sum()


def prior_corrected_logits(
    logits: Tensor,
    target_prior: Tensor,
    source_prior: Tensor,
    *,
    strength: float = 1.0,
    ratio_clip: float | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Apply the standard label-shift posterior correction in log space."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [samples, classes]")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    if ratio_clip is not None and ratio_clip <= 1.0:
        raise ValueError("ratio_clip must be greater than one")
    target = _probability(target_prior, eps).to(logits.device)
    source = _probability(source_prior, eps).to(logits.device)
    if target.shape != source.shape or target.numel() != logits.shape[1]:
        raise ValueError("prior shapes must match the logits class dimension")
    log_ratio = target.log() - source.log()
    if ratio_clip is not None:
        bound = torch.as_tensor(
            ratio_clip, dtype=log_ratio.dtype, device=log_ratio.device
        ).log()
        log_ratio = log_ratio.clamp(-bound, bound)
    adjustment = strength * log_ratio
    return logits + adjustment.unsqueeze(0)


@dataclass
class PrototypeAdapterConfig:
    temperature: float = 0.07
    momentum: float = 0.9
    source_anchor: float = 0.2
    confidence_threshold: float = 0.8
    minimum_class_mass: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if not 0.0 <= self.source_anchor <= 1.0:
            raise ValueError("source_anchor must lie in [0, 1]")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")


@dataclass
class PriorAdapterConfig:
    """Stability controls for sequential label-prior correction."""

    strength: float = 0.5
    momentum: float = 0.5
    ratio_clip: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must lie in [0, 1]")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if self.ratio_clip <= 1.0:
            raise ValueError("ratio_clip must be greater than one")


class SourceAnchoredPrototypeAdapter:
    """Gradient-free representation expert with source-anchored prototypes.

    The adapter only consumes unlabeled normalized features and soft pseudo
    labels.  ShiftRoute decides whether this expert is allowed to update.
    """

    def __init__(
        self,
        source_prototypes: Tensor,
        config: PrototypeAdapterConfig | None = None,
    ) -> None:
        if source_prototypes.ndim != 2:
            raise ValueError("source_prototypes must have shape [classes, dim]")
        self.config = config or PrototypeAdapterConfig()
        self.source_prototypes = F.normalize(source_prototypes.float(), dim=-1)
        self.prototypes = self.source_prototypes.clone()
        self.update_steps = 0

    def to(self, device: torch.device | str) -> "SourceAnchoredPrototypeAdapter":
        self.source_prototypes = self.source_prototypes.to(device)
        self.prototypes = self.prototypes.to(device)
        return self

    def reset(self) -> None:
        self.prototypes = self.source_prototypes.clone()
        self.update_steps = 0

    def logits(self, features: Tensor) -> Tensor:
        features = F.normalize(features.float(), dim=-1)
        return features @ self.prototypes.T / self.config.temperature

    def delta_logits(self, features: Tensor) -> Tensor:
        """Prototype-change correction to add to an existing source classifier."""

        features = F.normalize(features.float(), dim=-1)
        adapted = features @ self.prototypes.T
        source = features @ self.source_prototypes.T
        return (adapted - source) / self.config.temperature

    @torch.no_grad()
    def update(self, features: Tensor, probabilities: Tensor | None = None) -> int:
        features = F.normalize(features.float(), dim=-1)
        if probabilities is None:
            probabilities = self.logits(features).softmax(dim=-1)
        if probabilities.shape != (features.shape[0], self.prototypes.shape[0]):
            raise ValueError("probabilities have an incompatible shape")

        confidence = probabilities.max(dim=-1).values
        keep = confidence >= self.config.confidence_threshold
        if not bool(keep.any()):
            return 0
        selected_features = features[keep]
        selected_probabilities = probabilities[keep]
        class_mass = selected_probabilities.sum(dim=0)
        eligible = class_mass >= self.config.minimum_class_mass
        if not bool(eligible.any()):
            return 0

        means = selected_probabilities.T @ selected_features
        means = means / class_mass.clamp_min(1e-8).unsqueeze(-1)
        means = F.normalize(means, dim=-1)
        proposal = F.normalize(
            self.config.momentum * self.prototypes
            + (1.0 - self.config.momentum) * means,
            dim=-1,
        )
        proposal = F.normalize(
            (1.0 - self.config.source_anchor) * proposal
            + self.config.source_anchor * self.source_prototypes,
            dim=-1,
        )
        self.prototypes[eligible] = proposal[eligible]
        self.update_steps += 1
        return int(eligible.sum().item())

    def drift(self) -> float:
        cosine = (self.prototypes * self.source_prototypes).sum(dim=-1)
        return float((1.0 - cosine).mean().item())
