import pandas as pd

from scripts.evaluate_diagnostic_baselines import (
    holm_adjust,
    mechanism_targets,
    oriented_auc,
)


def test_oriented_auc_grants_optimal_direction():
    target = [0, 0, 1, 1]
    assert oriented_auc(target, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert oriented_auc(target, [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_holm_adjust_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


def test_mechanism_targets_match_serialized_action_values():
    actions = pd.Series(["freeze", "prior_only", "representation_only", "joint"])
    prior, representation = mechanism_targets(actions)
    assert prior.tolist() == [0, 1, 0, 1]
    assert representation.tolist() == [0, 0, 1, 1]
