import unittest

from shiftroute.router import Action, HysteresisRouter, RouterConfig


class RouterTests(unittest.TestCase):
    def test_router_requires_confirmation_and_honors_hysteresis(self):
        router = HysteresisRouter(
            RouterConfig(
                prior_enter=0.05,
                prior_exit=0.02,
                representation_enter=3.0,
                representation_exit=1.0,
                confirmation_windows=2,
                min_dwell_windows=0,
            )
        )
        self.assertEqual(router.update(0.10, 0.0), Action.FREEZE)
        self.assertEqual(router.update(0.10, 0.0), Action.PRIOR_ONLY)
        self.assertEqual(router.update(0.03, 0.0), Action.PRIOR_ONLY)
        self.assertEqual(router.update(0.00, 0.0), Action.PRIOR_ONLY)
        self.assertEqual(router.update(0.00, 0.0), Action.FREEZE)

    def test_router_selects_all_four_actions(self):
        router = HysteresisRouter(
            RouterConfig(confirmation_windows=1, min_dwell_windows=0)
        )
        self.assertEqual(router.update(0.0, 0.0), Action.FREEZE)
        self.assertEqual(router.update(0.10, 0.0), Action.PRIOR_ONLY)
        self.assertEqual(router.update(0.10, 5.0), Action.JOINT)

        router.reset()
        self.assertEqual(router.update(0.0, 5.0), Action.REPRESENTATION_ONLY)

    def test_low_confidence_forces_freeze(self):
        router = HysteresisRouter(
            RouterConfig(
                confirmation_windows=1,
                min_dwell_windows=0,
                minimum_confidence=0.6,
            )
        )
        self.assertEqual(
            router.update(0.1, 5.0, confidence=0.2),
            Action.FREEZE,
        )


if __name__ == "__main__":
    unittest.main()
