"""Finite-sample calibration for the two ShiftRoute evidence axes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor

from .router import RouterConfig


def _as_finite_vector(values: Iterable[float]) -> Tensor:
    tensor = torch.as_tensor(list(values), dtype=torch.float64).flatten()
    if tensor.numel() == 0:
        raise ValueError("calibration values cannot be empty")
    if not torch.isfinite(tensor).all():
        raise ValueError("calibration values must be finite")
    return tensor


def conformal_upper_threshold(values: Iterable[float], tail_probability: float) -> float:
    """Return a split-conformal upper threshold.

    The ``ceil((m+1)(1-alpha))`` order statistic yields marginal upper-tail
    coverage under exchangeability.  If the requested tail is smaller than
    ``1/(m+1)``, finite calibration data cannot resolve it and ``+inf`` is
    returned rather than claiming unsupported precision.
    """

    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must lie in (0, 1)")
    tensor = _as_finite_vector(values).sort().values
    rank = int(math.ceil((tensor.numel() + 1) * (1.0 - tail_probability)))
    if rank > tensor.numel():
        return float("inf")
    return float(tensor[rank - 1].item())


def conformal_upper_pvalue(values: Iterable[float], observed: float) -> float:
    """One-sided conformal p-value for an unusually large evidence score."""

    tensor = _as_finite_vector(values)
    return float((1 + (tensor >= float(observed)).sum().item()) / (tensor.numel() + 1))


@dataclass(frozen=True)
class CalibratedThresholds:
    prior_enter: float
    prior_exit: float
    representation_enter: float
    representation_exit: float
    familywise_alpha: float
    calibration_windows: int
    calibration_method: str = "bonferroni_marginal"
    prior_scale: float = 1.0
    representation_scale: float = 1.0

    def to_router_config(
        self,
        *,
        confirmation_windows: int = 2,
        min_dwell_windows: int = 2,
        minimum_confidence: float = 0.0,
    ) -> RouterConfig:
        return RouterConfig(
            prior_enter=self.prior_enter,
            prior_exit=self.prior_exit,
            representation_enter=self.representation_enter,
            representation_exit=self.representation_exit,
            confirmation_windows=confirmation_windows,
            min_dwell_windows=min_dwell_windows,
            minimum_confidence=minimum_confidence,
        )


def calibrate_thresholds(
    clean_prior_scores: Iterable[float],
    clean_representation_scores: Iterable[float],
    *,
    familywise_alpha: float = 0.05,
    exit_quantile: float = 0.80,
) -> CalibratedThresholds:
    """Calibrate both axes with Bonferroni-controlled entry thresholds.

    The formal guarantee concerns a single source-like window.  Hysteresis and
    repeated confirmations are operational stabilizers and are not presented
    as an anytime-valid theorem.
    """

    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0, 1)")
    if not 0.0 < exit_quantile < 1.0:
        raise ValueError("exit_quantile must lie in (0, 1)")

    prior = _as_finite_vector(clean_prior_scores)
    representation = _as_finite_vector(clean_representation_scores)
    if prior.numel() != representation.numel():
        raise ValueError("both diagnostic axes need the same calibration windows")

    axis_tail = familywise_alpha / 2.0
    prior_enter = conformal_upper_threshold(prior.tolist(), axis_tail)
    representation_enter = conformal_upper_threshold(
        representation.tolist(), axis_tail
    )
    prior_exit = float(torch.quantile(prior, exit_quantile).item())
    representation_exit = float(torch.quantile(representation, exit_quantile).item())

    # RouterConfig requires strict hysteresis.  This only matters for degenerate
    # calibration vectors (for example, an exactly zero synthetic diagnostic).
    if math.isfinite(prior_enter):
        prior_margin = max(abs(prior_enter) * 1e-12, 1e-12)
        prior_exit = min(prior_exit, prior_enter - prior_margin)
    if math.isfinite(representation_enter):
        representation_margin = max(abs(representation_enter) * 1e-12, 1e-12)
        representation_exit = min(
            representation_exit,
            representation_enter - representation_margin,
        )
    return CalibratedThresholds(
        prior_enter=prior_enter,
        prior_exit=prior_exit,
        representation_enter=representation_enter,
        representation_exit=representation_exit,
        familywise_alpha=familywise_alpha,
        calibration_windows=int(prior.numel()),
        calibration_method="bonferroni_marginal",
    )


def development_scale(
    values: Iterable[float],
    *,
    quantile: float = 0.90,
    minimum: float = 1e-8,
) -> float:
    """Return a fixed positive axis scale learned before conformal calibration.

    The development sample is part of the fitted controller.  It must therefore
    be separate from the identity-disjoint conformal sample used to obtain the
    finite-sample threshold.
    """

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    if minimum <= 0.0:
        raise ValueError("minimum must be positive")
    tensor = _as_finite_vector(values)
    return max(float(torch.quantile(tensor, quantile).item()), float(minimum))


def calibrate_joint_max_thresholds(
    clean_prior_scores: Iterable[float],
    clean_representation_scores: Iterable[float],
    *,
    prior_scale: float,
    representation_scale: float,
    familywise_alpha: float = 0.05,
    exit_quantile: float = 0.80,
) -> CalibratedThresholds:
    """Calibrate a familywise threshold through one max-type conformity score.

    With fixed positive development scales ``s_p`` and ``s_r``, each clean
    window has score ``max(E_p / s_p, E_r / s_r)``.  A single conformal upper
    quantile of this maximum controls the event that either router coordinate
    crosses its entry threshold.  Unlike marginal Bonferroni calibration, this
    construction only needs enough windows to resolve the requested familywise
    tail probability itself.
    """

    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0, 1)")
    if not 0.0 < exit_quantile < 1.0:
        raise ValueError("exit_quantile must lie in (0, 1)")
    if prior_scale <= 0.0 or representation_scale <= 0.0:
        raise ValueError("development scales must be positive")

    prior = _as_finite_vector(clean_prior_scores)
    representation = _as_finite_vector(clean_representation_scores)
    if prior.numel() != representation.numel():
        raise ValueError("both diagnostic axes need the same calibration windows")

    maximum = torch.maximum(
        prior / float(prior_scale),
        representation / float(representation_scale),
    )
    joint_enter = conformal_upper_threshold(maximum.tolist(), familywise_alpha)
    prior_enter = float(joint_enter * prior_scale)
    representation_enter = float(joint_enter * representation_scale)
    prior_exit = float(torch.quantile(prior, exit_quantile).item())
    representation_exit = float(torch.quantile(representation, exit_quantile).item())

    if math.isfinite(prior_enter):
        prior_exit = min(
            prior_exit,
            prior_enter - max(abs(prior_enter) * 1e-12, 1e-12),
        )
    if math.isfinite(representation_enter):
        representation_exit = min(
            representation_exit,
            representation_enter
            - max(abs(representation_enter) * 1e-12, 1e-12),
        )

    return CalibratedThresholds(
        prior_enter=prior_enter,
        prior_exit=prior_exit,
        representation_enter=representation_enter,
        representation_exit=representation_exit,
        familywise_alpha=familywise_alpha,
        calibration_windows=int(prior.numel()),
        calibration_method="joint_max_conformal",
        prior_scale=float(prior_scale),
        representation_scale=float(representation_scale),
    )
