"""Prior-projected diagnostics for normalized vision-language features.

The implementation intentionally stores no source images.  Test-time state consists of
class prototypes, a source prior, and per-class residual calibration statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


def _normalize_rows(value: Tensor) -> Tensor:
    if value.ndim != 2:
        raise ValueError(f"expected a rank-2 tensor, got shape {tuple(value.shape)}")
    return F.normalize(value.float(), dim=-1)


def _normalize_probability(value: Tensor, eps: float) -> Tensor:
    value = value.float().clamp_min(eps)
    return value / value.sum()


@dataclass(frozen=True)
class SourceStatistics:
    """Compact source summary used by the test-time diagnostic."""

    prototypes: Tensor
    source_prior: Tensor
    residual_mean: Tensor
    residual_var: Tensor
    kappa: float = 30.0

    def __post_init__(self) -> None:
        if self.prototypes.ndim != 2:
            raise ValueError("prototypes must have shape [classes, feature_dim]")
        classes = self.prototypes.shape[0]
        for name, value in (
            ("source_prior", self.source_prior),
            ("residual_mean", self.residual_mean),
            ("residual_var", self.residual_var),
        ):
            if value.shape != (classes,):
                raise ValueError(f"{name} must have shape [{classes}]")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")

    def to(self, device: torch.device | str) -> "SourceStatistics":
        return SourceStatistics(
            prototypes=self.prototypes.to(device),
            source_prior=self.source_prior.to(device),
            residual_mean=self.residual_mean.to(device),
            residual_var=self.residual_var.to(device),
            kappa=self.kappa,
        )


@dataclass(frozen=True)
class ShiftDiagnosis:
    """Diagnostics returned for one stream window."""

    target_prior: Tensor
    responsibilities: Tensor
    prior_score: Tensor
    representation_score: Tensor
    signed_representation_z: Tensor
    observed_residual: Tensor
    expected_residual: Tensor
    mixture_nll: Tensor
    mean_assignment_confidence: Tensor


def fit_source_statistics(
    features: Tensor,
    labels: Tensor,
    *,
    num_classes: int | None = None,
    kappa: float = 30.0,
    prior_smoothing: float = 1.0,
    variance_floor: float = 1e-5,
) -> SourceStatistics:
    """Fit prototypes and residual calibration from labeled source features.

    Args:
        features: Normalized or unnormalized source features, shape ``[N, D]``.
        labels: Integer labels, shape ``[N]``.
        num_classes: Total number of classes. Defaults to ``max(labels)+1``.
    """

    features = _normalize_rows(features)
    labels = labels.long().flatten()
    if labels.numel() != features.shape[0]:
        raise ValueError("features and labels must contain the same number of samples")
    if labels.numel() == 0:
        raise ValueError("cannot fit source statistics from an empty tensor")

    inferred_classes = int(labels.max().item()) + 1
    num_classes = inferred_classes if num_classes is None else num_classes
    if num_classes < inferred_classes:
        raise ValueError("num_classes is smaller than the largest label")

    prototypes = []
    counts = []
    for class_index in range(num_classes):
        mask = labels == class_index
        count = int(mask.sum().item())
        if count == 0:
            raise ValueError(f"class {class_index} has no source samples")
        prototypes.append(F.normalize(features[mask].mean(dim=0), dim=0))
        counts.append(count)
    prototype_tensor = torch.stack(prototypes)

    own_prototypes = prototype_tensor[labels]
    residuals = 1.0 - (features * own_prototypes).sum(dim=-1)
    residual_mean = torch.empty(num_classes, device=features.device)
    residual_var = torch.empty_like(residual_mean)
    for class_index in range(num_classes):
        class_residuals = residuals[labels == class_index]
        residual_mean[class_index] = class_residuals.mean()
        residual_var[class_index] = class_residuals.var(unbiased=False).clamp_min(variance_floor)

    count_tensor = torch.tensor(counts, dtype=features.dtype, device=features.device)
    source_prior = (count_tensor + prior_smoothing) / (
        count_tensor.sum() + prior_smoothing * num_classes
    )
    return SourceStatistics(
        prototypes=prototype_tensor,
        source_prior=source_prior,
        residual_mean=residual_mean,
        residual_var=residual_var,
        kappa=kappa,
    )


def estimate_target_prior(
    features: Tensor,
    source: SourceStatistics,
    *,
    max_steps: int = 20,
    tolerance: float = 1e-5,
    prior_strength: float = 1.0,
    damping: float = 0.25,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Estimate target mixture weights while keeping source components fixed.

    Returns the target prior, final responsibilities, and per-sample log mixture
    density up to the common vMF normalization constant.
    """

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1)")

    features = _normalize_rows(features)
    source = source.to(features.device)
    if features.shape[1] != source.prototypes.shape[1]:
        raise ValueError("feature dimension does not match source prototypes")

    component_logits = source.kappa * features @ source.prototypes.T
    target_prior = _normalize_probability(source.source_prior, eps)

    for _ in range(max_steps):
        log_joint = component_logits + target_prior.clamp_min(eps).log()
        responsibilities = log_joint.softmax(dim=-1)
        updated = responsibilities.sum(dim=0) + prior_strength * source.source_prior
        updated = _normalize_probability(updated, eps)
        updated = damping * target_prior + (1.0 - damping) * updated
        if torch.max(torch.abs(updated - target_prior)) <= tolerance:
            target_prior = updated
            break
        target_prior = updated

    log_joint = component_logits + target_prior.clamp_min(eps).log()
    responsibilities = log_joint.softmax(dim=-1)
    log_mixture = torch.logsumexp(log_joint, dim=-1)
    return target_prior, responsibilities, log_mixture


def jensen_shannon_divergence(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    p = _normalize_probability(p, eps)
    q = _normalize_probability(q, eps)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        torch.sum(p * (p.clamp_min(eps).log() - midpoint.clamp_min(eps).log()))
        + torch.sum(q * (q.clamp_min(eps).log() - midpoint.clamp_min(eps).log()))
    )


def diagnose_window(
    features: Tensor,
    source: SourceStatistics,
    *,
    max_em_steps: int = 20,
    prior_strength: float = 1.0,
    prior_project_representation: bool = True,
    include_between_class_variance: bool = True,
    eps: float = 1e-8,
) -> ShiftDiagnosis:
    """Estimate prior and representation evidence for one unlabeled window."""

    features = _normalize_rows(features)
    source = source.to(features.device)
    target_prior, responsibilities, log_mixture = estimate_target_prior(
        features,
        source,
        max_steps=max_em_steps,
        prior_strength=prior_strength,
        eps=eps,
    )

    cosine_distance = 1.0 - features @ source.prototypes.T
    observed_residual = (responsibilities * cosine_distance).sum(dim=-1).mean()
    representation_prior = (
        target_prior if prior_project_representation else source.source_prior
    )
    expected_residual = torch.sum(representation_prior * source.residual_mean)

    window_size = max(features.shape[0], 1)
    # Law of total variance.  The between-class term matters whenever classes
    # have different source residual means; omitting it makes the diagnostic
    # spuriously sensitive to label-prior changes.
    variance_terms = source.residual_var
    if include_between_class_variance:
        variance_terms = variance_terms + (
            source.residual_mean - expected_residual
        ).square()
    expected_variance = torch.sum(
        representation_prior * variance_terms
    ) / window_size
    signed_z = (observed_residual - expected_residual) / torch.sqrt(
        expected_variance.clamp_min(eps)
    )

    return ShiftDiagnosis(
        target_prior=target_prior,
        responsibilities=responsibilities,
        prior_score=jensen_shannon_divergence(target_prior, source.source_prior, eps),
        representation_score=signed_z.clamp_min(0.0),
        signed_representation_z=signed_z,
        observed_residual=observed_residual,
        expected_residual=expected_residual,
        mixture_nll=-log_mixture.mean(),
        mean_assignment_confidence=responsibilities.max(dim=-1).values.mean(),
    )
