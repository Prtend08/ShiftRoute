"""Core components for ShiftRoute."""

from .diagnostics import (
    ShiftDiagnosis,
    SourceStatistics,
    diagnose_window,
    estimate_target_prior,
    fit_source_statistics,
)
from .calibration import CalibratedThresholds, calibrate_thresholds
from .adapters import (
    PriorAdapterConfig,
    PrototypeAdapterConfig,
    SourceAnchoredPrototypeAdapter,
    prior_corrected_logits,
)
from .router import Action, HysteresisRouter, RouterConfig

__all__ = [
    "Action",
    "HysteresisRouter",
    "RouterConfig",
    "ShiftDiagnosis",
    "SourceStatistics",
    "diagnose_window",
    "estimate_target_prior",
    "fit_source_statistics",
    "CalibratedThresholds",
    "calibrate_thresholds",
    "PrototypeAdapterConfig",
    "PriorAdapterConfig",
    "SourceAnchoredPrototypeAdapter",
    "prior_corrected_logits",
]
