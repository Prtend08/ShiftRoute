"""Hysteretic four-action routing for ShiftRoute."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    FREEZE = "freeze"
    PRIOR_ONLY = "prior_only"
    REPRESENTATION_ONLY = "representation_only"
    JOINT = "joint"


@dataclass(frozen=True)
class RouterConfig:
    prior_enter: float = 0.05
    prior_exit: float = 0.025
    representation_enter: float = 3.0
    representation_exit: float = 1.5
    confirmation_windows: int = 2
    min_dwell_windows: int = 2
    minimum_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.prior_enter <= self.prior_exit:
            raise ValueError("prior_enter must be greater than prior_exit")
        if self.representation_enter <= self.representation_exit:
            raise ValueError("representation_enter must be greater than representation_exit")
        if self.confirmation_windows <= 0:
            raise ValueError("confirmation_windows must be positive")
        if self.min_dwell_windows < 0:
            raise ValueError("min_dwell_windows cannot be negative")


@dataclass
class _EvidenceLatch:
    active: bool = False
    enter_count: int = 0
    exit_count: int = 0

    def update(self, value: float, *, enter: float, exit: float, confirmations: int) -> None:
        if self.active:
            if value <= exit:
                self.exit_count += 1
            else:
                self.exit_count = 0
            if self.exit_count >= confirmations:
                self.active = False
                self.exit_count = 0
                self.enter_count = 0
        else:
            if value >= enter:
                self.enter_count += 1
            else:
                self.enter_count = 0
            if self.enter_count >= confirmations:
                self.active = True
                self.enter_count = 0
                self.exit_count = 0


class HysteresisRouter:
    """Stateful router that avoids reacting to one noisy stream window."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()
        self._prior = _EvidenceLatch()
        self._representation = _EvidenceLatch()
        self.action = Action.FREEZE
        self._dwell = 0

    @property
    def dwell_windows(self) -> int:
        return self._dwell

    def reset(self) -> None:
        self._prior = _EvidenceLatch()
        self._representation = _EvidenceLatch()
        self.action = Action.FREEZE
        self._dwell = 0

    def update(
        self,
        prior_score: float,
        representation_score: float,
        *,
        confidence: float = 1.0,
        force_freeze: bool = False,
    ) -> Action:
        config = self.config
        self._prior.update(
            prior_score,
            enter=config.prior_enter,
            exit=config.prior_exit,
            confirmations=config.confirmation_windows,
        )
        self._representation.update(
            representation_score,
            enter=config.representation_enter,
            exit=config.representation_exit,
            confirmations=config.confirmation_windows,
        )

        if force_freeze or confidence < config.minimum_confidence:
            candidate = Action.FREEZE
        elif self._prior.active and self._representation.active:
            candidate = Action.JOINT
        elif self._prior.active:
            candidate = Action.PRIOR_ONLY
        elif self._representation.active:
            candidate = Action.REPRESENTATION_ONLY
        else:
            candidate = Action.FREEZE

        if candidate != self.action and self._dwell < config.min_dwell_windows:
            self._dwell += 1
            return self.action
        if candidate != self.action:
            self.action = candidate
            self._dwell = 1
        else:
            self._dwell += 1
        return self.action

