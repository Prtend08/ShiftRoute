from scripts.analyze_ablation_variants import holm_adjust, safe_wilcoxon

import pandas as pd


def test_ablation_holm_adjustment():
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_safe_wilcoxon_handles_identical_effects():
    assert safe_wilcoxon(pd.Series([0.0, 0.0, 0.0])) == 1.0
