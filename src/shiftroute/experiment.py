"""Feature-cache stream construction and prequential ShiftRoute evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .adapters import (
    PriorAdapterConfig,
    PrototypeAdapterConfig,
    SourceAnchoredPrototypeAdapter,
    prior_corrected_logits,
)
from .calibration import (
    CalibratedThresholds,
    calibrate_joint_max_thresholds,
    calibrate_thresholds,
    development_scale,
)
from .diagnostics import SourceStatistics, diagnose_window, fit_source_statistics
from .router import Action, HysteresisRouter


@dataclass(frozen=True)
class FeatureBank:
    name: str
    features: Tensor
    logits: Tensor
    labels: Tensor

    def __post_init__(self) -> None:
        samples = self.labels.numel()
        if self.features.ndim != 2 or self.logits.ndim != 2:
            raise ValueError("feature and logit banks must be rank two")
        if self.features.shape[0] != samples or self.logits.shape[0] != samples:
            raise ValueError("all bank tensors need the same sample dimension")

    @property
    def classes(self) -> int:
        return int(self.logits.shape[1])


@dataclass(frozen=True)
class StreamPhase:
    name: str
    bank: str
    windows: int
    true_action: Action
    prior: Optional[Tensor] = None


@dataclass(frozen=True)
class StreamWindow:
    window_index: int
    phase_name: str
    true_action: Action
    features: Tensor
    logits: Tensor
    labels: Tensor
    bank_name: str = ""
    sample_indices: Optional[Tensor] = None


def long_tailed_prior(classes: int, imbalance_ratio: float, reverse: bool = False) -> Tensor:
    if classes <= 1 or imbalance_ratio < 1.0:
        raise ValueError("invalid class count or imbalance ratio")
    exponents = torch.linspace(0.0, 1.0, classes)
    values = imbalance_ratio ** (-exponents)
    if reverse:
        values = values.flip(0)
    return values / values.sum()


def _sample_indices(labels: Tensor, prior: Tensor, samples: int, rng: np.random.Generator) -> Tensor:
    labels_np = labels.cpu().numpy()
    prior_np = (prior / prior.sum()).cpu().numpy()
    requested_labels = rng.choice(len(prior_np), size=samples, replace=True, p=prior_np)
    output = np.empty(samples, dtype=np.int64)
    class_indices = [np.flatnonzero(labels_np == index) for index in range(len(prior_np))]
    for class_index, candidates in enumerate(class_indices):
        positions = np.flatnonzero(requested_labels == class_index)
        if positions.size == 0:
            continue
        if candidates.size == 0:
            raise ValueError("a requested class is absent from the feature bank")
        output[positions] = rng.choice(candidates, size=positions.size, replace=True)
    return torch.from_numpy(output)


def build_stream(
    banks: Dict[str, FeatureBank],
    phases: Sequence[StreamPhase],
    *,
    window_size: int,
    seed: int,
) -> List[StreamWindow]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    rng = np.random.default_rng(seed)
    windows: List[StreamWindow] = []
    window_index = 0
    for phase in phases:
        bank = banks[phase.bank]
        prior = phase.prior
        if prior is None:
            prior = torch.bincount(bank.labels.long(), minlength=bank.classes).float()
            prior = prior / prior.sum()
        for _ in range(phase.windows):
            indices = _sample_indices(bank.labels, prior, window_size, rng)
            windows.append(
                StreamWindow(
                    window_index=window_index,
                    phase_name=phase.name,
                    true_action=phase.true_action,
                    features=bank.features[indices].float(),
                    logits=bank.logits[indices].float(),
                    labels=bank.labels[indices].long(),
                    bank_name=phase.bank,
                    sample_indices=indices,
                )
            )
            window_index += 1
    return windows


def score_clean_windows(
    features: Tensor,
    labels: Tensor,
    source: SourceStatistics,
    *,
    window_size: int,
    windows: int | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
    prior_project_representation: bool = True,
    include_between_class_variance: bool = True,
    sampling: str = "disjoint",
    return_assignment_confidence: bool = False,
) -> tuple[list[float], list[float]] | tuple[list[float], list[float], list[float]]:
    """Compute clean diagnostic scores using an explicit identity policy.

    ``disjoint`` shuffles the clean identity pool once and partitions it into
    non-overlapping windows. ``bootstrap`` is retained only for development
    scale estimation and legacy pilot reproduction; it must not be used to
    claim split-conformal coverage.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(features) != len(labels):
        raise ValueError("features and labels must contain the same samples")
    if sampling not in {"disjoint", "bootstrap"}:
        raise ValueError("sampling must be 'disjoint' or 'bootstrap'")
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    features = features.to(device)
    source = source.to(device)
    prior_scores = []
    representation_scores = []
    assignment_confidences = []
    samples = len(labels)
    if sampling == "disjoint":
        available = samples // window_size
        requested = available if windows is None or windows <= 0 else windows
        if requested > available:
            raise ValueError(
                f"requested {requested} disjoint windows of size {window_size}, "
                f"but only {available} fit in {samples} clean identities"
            )
        permutation = rng.permutation(samples)[: requested * window_size]
        index_windows = permutation.reshape(requested, window_size)
    else:
        requested = 200 if windows is None or windows <= 0 else windows
        index_windows = [
            rng.choice(samples, size=window_size, replace=True)
            for _ in range(requested)
        ]

    for indices in index_windows:
        index_tensor = torch.from_numpy(indices).to(device)
        diagnosis = diagnose_window(
            features[index_tensor].float(),
            source,
            prior_project_representation=prior_project_representation,
            include_between_class_variance=include_between_class_variance,
        )
        prior_scores.append(float(diagnosis.prior_score))
        representation_scores.append(float(diagnosis.representation_score))
        assignment_confidences.append(float(diagnosis.mean_assignment_confidence))
    if return_assignment_confidence:
        return prior_scores, representation_scores, assignment_confidences
    return prior_scores, representation_scores


def calibrate_from_clean_windows(
    features: Tensor,
    labels: Tensor,
    source: SourceStatistics,
    *,
    window_size: int,
    windows: int | None = None,
    familywise_alpha: float = 0.05,
    seed: int = 0,
    device: torch.device | str = "cpu",
    prior_project_representation: bool = True,
    include_between_class_variance: bool = True,
    sampling: str = "bootstrap",
    prior_scale: float | None = None,
    representation_scale: float | None = None,
) -> CalibratedThresholds:
    prior_scores, representation_scores = score_clean_windows(
        features,
        labels,
        source,
        window_size=window_size,
        windows=windows,
        seed=seed,
        device=device,
        prior_project_representation=prior_project_representation,
        include_between_class_variance=include_between_class_variance,
        sampling=sampling,
    )
    if prior_scale is not None or representation_scale is not None:
        if prior_scale is None or representation_scale is None:
            raise ValueError("both development scales are required for joint calibration")
        return calibrate_joint_max_thresholds(
            prior_scores,
            representation_scores,
            prior_scale=prior_scale,
            representation_scale=representation_scale,
            familywise_alpha=familywise_alpha,
        )
    return calibrate_thresholds(
        prior_scores,
        representation_scores,
        familywise_alpha=familywise_alpha,
    )


def development_scales_from_clean_windows(
    features: Tensor,
    labels: Tensor,
    source: SourceStatistics,
    *,
    window_size: int,
    windows: int = 300,
    quantile: float = 0.90,
    seed: int = 0,
    device: torch.device | str = "cpu",
    prior_project_representation: bool = True,
    include_between_class_variance: bool = True,
) -> tuple[float, float]:
    prior_scores, representation_scores = score_clean_windows(
        features,
        labels,
        source,
        window_size=window_size,
        windows=windows,
        seed=seed,
        device=device,
        prior_project_representation=prior_project_representation,
        include_between_class_variance=include_between_class_variance,
        sampling="bootstrap",
    )
    return (
        development_scale(prior_scores, quantile=quantile),
        development_scale(representation_scores, quantile=quantile),
    )


def development_prior_reliability_from_clean_windows(
    features: Tensor,
    labels: Tensor,
    source: SourceStatistics,
    *,
    window_size: int,
    windows: int = 300,
    lower_quantile: float = 0.05,
    seed: int = 0,
    device: torch.device | str = "cpu",
    prior_project_representation: bool = True,
    include_between_class_variance: bool = True,
) -> float:
    """Fit a clean lower bound for when the estimated prior is trustworthy."""

    if not 0.0 < lower_quantile < 1.0:
        raise ValueError("lower_quantile must lie in (0, 1)")
    _, _, confidences = score_clean_windows(
        features,
        labels,
        source,
        window_size=window_size,
        windows=windows,
        seed=seed,
        device=device,
        prior_project_representation=prior_project_representation,
        include_between_class_variance=include_between_class_variance,
        sampling="bootstrap",
        return_assignment_confidence=True,
    )
    return float(torch.quantile(torch.tensor(confidences), lower_quantile).item())


@dataclass
class _MethodState:
    name: str
    action: Action
    adapter: SourceAnchoredPrototypeAdapter
    target_prior: Tensor
    prior_config: PriorAdapterConfig

    def predict(self, window: StreamWindow, source_prior: Tensor) -> Tensor:
        logits = window.logits.float() + self.adapter.delta_logits(window.features)
        if self.action in (Action.PRIOR_ONLY, Action.JOINT):
            logits = prior_corrected_logits(
                logits,
                self.target_prior,
                source_prior,
                strength=self.prior_config.strength,
                ratio_clip=self.prior_config.ratio_clip,
            )
        return logits

    def transition(self, action: Action, diagnosis, window: StreamWindow, source_prior: Tensor) -> None:
        if action in (Action.FREEZE, Action.PRIOR_ONLY):
            self.adapter.reset()
        if action in (Action.FREEZE, Action.REPRESENTATION_ONLY):
            self.target_prior = source_prior.clone()
        if action in (Action.PRIOR_ONLY, Action.JOINT):
            estimate = diagnosis.target_prior.detach()
            self.target_prior = (
                self.prior_config.momentum * self.target_prior
                + (1.0 - self.prior_config.momentum) * estimate
            )
            self.target_prior = self.target_prior / self.target_prior.sum()
        if action in (Action.REPRESENTATION_ONLY, Action.JOINT):
            update_logits = window.logits.float() + self.adapter.delta_logits(window.features)
            if action == Action.JOINT:
                update_logits = prior_corrected_logits(
                    update_logits,
                    self.target_prior,
                    source_prior,
                    strength=self.prior_config.strength,
                    ratio_clip=self.prior_config.ratio_clip,
                )
            probabilities = update_logits.softmax(dim=-1)
            self.adapter.update(window.features, probabilities)
        self.action = action


def expected_calibration_error(logits: Tensor, labels: Tensor, bins: int = 15) -> float:
    probabilities = logits.softmax(dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    correctness = predictions.eq(labels)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    output = logits.new_tensor(0.0)
    for index in range(bins):
        include = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if bool(include.any()):
            gap = confidence[include].mean() - correctness[include].float().mean()
            output += include.float().mean() * gap.abs()
    return float(output.item())


def _method_action(name: str, true_action: Action, routed_action: Action) -> Action:
    mapping = {
        "source": Action.FREEZE,
        "always_prior": Action.PRIOR_ONLY,
        "always_representation": Action.REPRESENTATION_ONLY,
        "always_joint": Action.JOINT,
        "oracle": true_action,
        "shiftroute": routed_action,
    }
    return mapping[name]


def apply_prior_reliability_guard(
    action: Action,
    *,
    assignment_confidence: float,
    threshold: float,
) -> tuple[Action, bool]:
    """Suppress prior-bearing actions when class assignments are unreliable.

    The guard can only remove the prior component.  It never creates an alarm,
    so familywise false-entry control from the joint conformal threshold is
    preserved.
    """

    reliable = float(assignment_confidence) >= float(threshold)
    if reliable:
        return action, True
    if action == Action.JOINT:
        return Action.REPRESENTATION_ONLY, False
    if action == Action.PRIOR_ONLY:
        return Action.FREEZE, False
    return action, False


def run_prequential_experiment(
    source_features: Tensor,
    source_labels: Tensor,
    calibration_features: Tensor,
    calibration_labels: Tensor,
    windows: Sequence[StreamWindow],
    thresholds: CalibratedThresholds,
    *,
    kappa: float = 30.0,
    adapter_config: Optional[PrototypeAdapterConfig] = None,
    prior_config: Optional[PriorAdapterConfig] = None,
    device: torch.device | str = "cpu",
    log_sample_events: bool = True,
    prior_project_representation: bool = True,
    include_between_class_variance: bool = True,
    confirmation_windows: int = 2,
    min_dwell_windows: int = 2,
    minimum_assignment_confidence: float = 0.0,
    prior_reliability_threshold: float = 0.0,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], SourceStatistics]:
    """Evaluate predictions before each window is used for diagnosis/update."""

    device = torch.device(device)
    source_features = source_features.to(device)
    source_labels = source_labels.to(device)
    source = fit_source_statistics(source_features, source_labels, kappa=kappa)
    prior_config = prior_config or PriorAdapterConfig()
    router = HysteresisRouter(
        thresholds.to_router_config(
            confirmation_windows=confirmation_windows,
            min_dwell_windows=min_dwell_windows,
            minimum_confidence=minimum_assignment_confidence,
        )
    )
    method_names = (
        "source",
        "always_prior",
        "always_representation",
        "always_joint",
        "shiftroute",
        "oracle",
    )
    states = {
        name: _MethodState(
            name=name,
            action=Action.FREEZE,
            adapter=SourceAnchoredPrototypeAdapter(source.prototypes, adapter_config).to(device),
            target_prior=source.source_prior.clone(),
            prior_config=prior_config,
        )
        for name in method_names
    }
    window_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []

    for window in windows:
        active_window = StreamWindow(
            window_index=window.window_index,
            phase_name=window.phase_name,
            true_action=window.true_action,
            features=window.features.to(device),
            logits=window.logits.to(device),
            labels=window.labels.to(device),
            bank_name=window.bank_name,
            sample_indices=window.sample_indices,
        )
        diagnosis = diagnose_window(
            active_window.features,
            source,
            prior_project_representation=prior_project_representation,
            include_between_class_variance=include_between_class_variance,
        )
        routed_action = router.update(
            float(diagnosis.prior_score),
            float(diagnosis.representation_score),
            confidence=float(diagnosis.mean_assignment_confidence),
        )
        routed_action, prior_reliable = apply_prior_reliability_guard(
            routed_action,
            assignment_confidence=float(diagnosis.mean_assignment_confidence),
            threshold=prior_reliability_threshold,
        )
        empirical_prior = torch.bincount(
            active_window.labels.long(), minlength=active_window.logits.shape[1]
        ).float()
        empirical_prior = empirical_prior / empirical_prior.sum().clamp_min(1.0)
        prior_total_variation = 0.5 * torch.abs(
            diagnosis.target_prior.cpu() - empirical_prior.cpu()
        ).sum()
        for name, state in states.items():
            logits = state.predict(active_window, source.source_prior)
            predictions = logits.argmax(dim=-1)
            accuracy = float(predictions.eq(active_window.labels).float().mean().item())
            nll = float(F.cross_entropy(logits, active_window.labels).item())
            row = {
                "window": window.window_index,
                "phase": window.phase_name,
                "true_action": window.true_action.value,
                "method": name,
                "action_applied": state.action.value,
                "action_selected_next": _method_action(
                    name, window.true_action, routed_action
                ).value,
                "accuracy": accuracy,
                "nll": nll,
                "ece": expected_calibration_error(logits, active_window.labels),
                "prior_score": float(diagnosis.prior_score),
                "representation_score": float(diagnosis.representation_score),
                "assignment_confidence": float(diagnosis.mean_assignment_confidence),
                "prior_reliable": bool(prior_reliable),
                # Offline audit metric only.  The empirical label histogram is
                # never passed to the diagnostic, router, or state transition.
                "prior_total_variation": float(prior_total_variation),
                "prototype_drift": state.adapter.drift(),
            }
            window_rows.append(row)
            if not log_sample_events:
                continue
            labels_cpu = active_window.labels.detach().cpu().tolist()
            predictions_cpu = predictions.detach().cpu().tolist()
            for offset, (label, prediction) in enumerate(
                zip(labels_cpu, predictions_cpu)
            ):
                event_rows.append(
                    {
                        "window": window.window_index,
                        "offset": offset,
                        "phase": window.phase_name,
                        "method": name,
                        "action_applied": state.action.value,
                        "label": int(label),
                        "prediction": int(prediction),
                        "correct": int(prediction == label),
                    }
                )

        # The current window is unlabeled to the controller.  Labels above are
        # used only for scoring and never enter diagnosis or transition.
        for name, state in states.items():
            next_action = _method_action(name, window.true_action, routed_action)
            state.transition(next_action, diagnosis, active_window, source.source_prior)

    return window_rows, event_rows, source
