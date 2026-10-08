import unittest

import torch

from shiftroute.diagnostics import diagnose_window, fit_source_statistics
from shiftroute.synthetic import make_prototypes, sample_features, shifted_prototypes


def _fixture():
    prototypes = make_prototypes(4, 32, seed=0)
    uniform = torch.full((4,), 0.25)
    source_features, source_labels = sample_features(
        prototypes,
        uniform,
        8000,
        noise=0.10,
        seed=1,
    )
    source = fit_source_statistics(source_features, source_labels, kappa=55.0)
    return prototypes, uniform, source


class DiagnosticTests(unittest.TestCase):
    def test_source_statistics_are_well_formed(self):
        _, _, source = _fixture()
        self.assertEqual(source.prototypes.shape, (4, 32))
        self.assertTrue(
            torch.allclose(source.prototypes.norm(dim=-1), torch.ones(4), atol=1e-5)
        )
        self.assertTrue(torch.isclose(source.source_prior.sum(), torch.tensor(1.0)))
        self.assertTrue(bool(torch.all(source.residual_var > 0)))

    def test_prior_projection_separates_controlled_shifts(self):
        prototypes, uniform, source = _fixture()
        long_tail = torch.tensor([0.70, 0.15, 0.10, 0.05])
        shifted = shifted_prototypes(prototypes, 0.55, seed=3)

        no_shift = diagnose_window(
            sample_features(prototypes, uniform, 1024, noise=0.10, seed=4)[0],
            source,
        )
        prior_only = diagnose_window(
            sample_features(prototypes, long_tail, 1024, noise=0.10, seed=5)[0],
            source,
        )
        representation_only = diagnose_window(
            sample_features(shifted, uniform, 1024, noise=0.10, seed=6)[0],
            source,
        )

        self.assertGreater(
            prior_only.prior_score.item(),
            no_shift.prior_score.item() + 0.02,
        )
        self.assertEqual(torch.argmax(prior_only.target_prior).item(), 0)
        self.assertGreater(
            representation_only.representation_score.item(),
            prior_only.representation_score.item() + 3.0,
        )
        self.assertGreater(
            representation_only.representation_score.item(),
            no_shift.representation_score.item() + 3.0,
        )

    def test_empty_source_class_is_rejected(self):
        features = torch.randn(8, 4)
        labels = torch.zeros(8, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "no source samples"):
            fit_source_statistics(features, labels, num_classes=2)


if __name__ == "__main__":
    unittest.main()
