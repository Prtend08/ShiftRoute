import unittest

import torch

from shiftroute.adapters import (
    PrototypeAdapterConfig,
    SourceAnchoredPrototypeAdapter,
    prior_corrected_logits,
)


class AdapterTests(unittest.TestCase):
    def test_prior_correction_favors_more_common_target_class(self):
        logits = torch.zeros(2, 2)
        corrected = prior_corrected_logits(
            logits,
            torch.tensor([0.8, 0.2]),
            torch.tensor([0.5, 0.5]),
        )
        self.assertTrue(torch.all(corrected[:, 0] > corrected[:, 1]))

    def test_prototype_update_respects_confidence_gate(self):
        adapter = SourceAnchoredPrototypeAdapter(
            torch.eye(2),
            PrototypeAdapterConfig(confidence_threshold=0.9),
        )
        updated = adapter.update(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.full((2, 2), 0.5),
        )
        self.assertEqual(updated, 0)
        self.assertEqual(adapter.drift(), 0.0)


if __name__ == "__main__":
    unittest.main()

