import math
import unittest

from shiftroute.calibration import (
    calibrate_joint_max_thresholds,
    calibrate_thresholds,
    conformal_upper_pvalue,
    conformal_upper_threshold,
    development_scale,
)


class CalibrationTests(unittest.TestCase):
    def test_conformal_order_statistic(self):
        values = list(range(99))
        self.assertEqual(conformal_upper_threshold(values, 0.05), 94.0)

    def test_unresolvable_tail_returns_infinity(self):
        self.assertTrue(math.isinf(conformal_upper_threshold([0.0, 1.0], 0.01)))

    def test_upper_pvalue(self):
        self.assertAlmostEqual(conformal_upper_pvalue([1.0, 2.0, 3.0], 2.5), 0.5)

    def test_joint_calibration_builds_router_config(self):
        scores = [index / 1000.0 for index in range(200)]
        calibrated = calibrate_thresholds(scores, scores, familywise_alpha=0.1)
        config = calibrated.to_router_config()
        self.assertGreater(config.prior_enter, config.prior_exit)
        self.assertGreater(config.representation_enter, config.representation_exit)

    def test_joint_max_controls_union_with_one_conformal_score(self):
        prior = [float(index + 1) for index in range(19)]
        representation = [float(19 - index) * 10.0 for index in range(19)]
        calibrated = calibrate_joint_max_thresholds(
            prior,
            representation,
            prior_scale=1.0,
            representation_scale=10.0,
            familywise_alpha=0.05,
        )
        self.assertEqual(calibrated.calibration_method, "joint_max_conformal")
        self.assertEqual(calibrated.prior_enter, 19.0)
        self.assertEqual(calibrated.representation_enter, 190.0)

    def test_development_scale_is_positive_for_degenerate_axis(self):
        self.assertEqual(development_scale([0.0, 0.0], minimum=1e-6), 1e-6)


if __name__ == "__main__":
    unittest.main()
