import unittest

import torch

from shiftroute.calibration import CalibratedThresholds
from shiftroute.diagnostics import fit_source_statistics
from shiftroute.experiment import (
    FeatureBank,
    StreamPhase,
    StreamWindow,
    apply_prior_reliability_guard,
    build_stream,
    score_clean_windows,
    long_tailed_prior,
    run_prequential_experiment,
)
from shiftroute.router import Action
from shiftroute.synthetic import make_prototypes, sample_features


class ExperimentTests(unittest.TestCase):
    def test_prior_reliability_guard_only_removes_prior_component(self):
        guarded, reliable = apply_prior_reliability_guard(
            Action.JOINT, assignment_confidence=0.7, threshold=0.9
        )
        self.assertEqual(guarded, Action.REPRESENTATION_ONLY)
        self.assertFalse(reliable)
        guarded, reliable = apply_prior_reliability_guard(
            Action.PRIOR_ONLY, assignment_confidence=0.95, threshold=0.9
        )
        self.assertEqual(guarded, Action.PRIOR_ONLY)
        self.assertTrue(reliable)

    def test_strict_calibration_rejects_overlapping_request(self):
        prototypes = make_prototypes(3, 8)
        prior = torch.full((3,), 1 / 3)
        features, labels = sample_features(prototypes, prior, 100, seed=1)
        source = fit_source_statistics(features, labels)
        with self.assertRaisesRegex(ValueError, "disjoint windows"):
            score_clean_windows(
                features,
                labels,
                source,
                window_size=30,
                windows=4,
                sampling="disjoint",
            )

    def test_stream_respects_requested_prior(self):
        labels = torch.arange(3).repeat_interleave(20)
        bank = FeatureBank(
            name="clean",
            features=torch.randn(60, 4),
            logits=torch.randn(60, 3),
            labels=labels,
        )
        prior = long_tailed_prior(3, 100.0)
        windows = build_stream(
            {"clean": bank},
            [StreamPhase("prior", "clean", 2, Action.PRIOR_ONLY, prior)],
            window_size=1000,
            seed=2,
        )
        observed = torch.bincount(windows[0].labels, minlength=3).float()
        observed /= observed.sum()
        self.assertLess(torch.max(torch.abs(observed - prior)).item(), 0.06)

    def test_non_oracle_predictions_do_not_depend_on_evaluation_labels(self):
        prototypes = make_prototypes(3, 8)
        prior = torch.full((3,), 1 / 3)
        source_features, source_labels = sample_features(prototypes, prior, 600, seed=1)
        calibration_features, calibration_labels = sample_features(prototypes, prior, 300, seed=2)
        features, labels = sample_features(prototypes, prior, 60, seed=3)
        logits = 12.0 * features @ prototypes.T
        thresholds = CalibratedThresholds(0.05, 0.02, 3.0, 1.0, 0.05, 200)
        window = StreamWindow(0, "clean", Action.FREEZE, features, logits, labels)
        _, events_a, _ = run_prequential_experiment(
            source_features,
            source_labels,
            calibration_features,
            calibration_labels,
            [window],
            thresholds,
        )
        permuted = labels[torch.randperm(labels.numel(), generator=torch.Generator().manual_seed(9))]
        relabeled_window = StreamWindow(0, "clean", Action.FREEZE, features, logits, permuted)
        _, events_b, _ = run_prequential_experiment(
            source_features,
            source_labels,
            calibration_features,
            calibration_labels,
            [relabeled_window],
            thresholds,
        )
        predictions_a = [row["prediction"] for row in events_a if row["method"] == "shiftroute"]
        predictions_b = [row["prediction"] for row in events_b if row["method"] == "shiftroute"]
        self.assertEqual(predictions_a, predictions_b)


if __name__ == "__main__":
    unittest.main()
    apply_prior_reliability_guard,
