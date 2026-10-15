from __future__ import annotations

import unittest

from iprune.experiments.tasks_vqe import _window_converged


class WindowConvergenceTests(unittest.TestCase):
    def test_rejects_before_minimum_steps(self) -> None:
        converged, spread = _window_converged(
            [2.0, 1.0, 1.0, 1.0], min_steps=4, window=3, tolerance=1e-6
        )
        self.assertFalse(converged)
        self.assertEqual(spread, float("inf"))

    def test_accepts_flat_final_window(self) -> None:
        converged, spread = _window_converged(
            [2.0, 1.0, 0.5, 0.500004, 0.500002],
            min_steps=4,
            window=3,
            tolerance=5e-6,
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(spread, 4e-6, places=12)

    def test_rejects_drifting_final_window(self) -> None:
        converged, spread = _window_converged(
            [2.0, 1.0, 0.7, 0.6, 0.5],
            min_steps=4,
            window=3,
            tolerance=1e-3,
        )
        self.assertFalse(converged)
        self.assertAlmostEqual(spread, 0.2, places=12)


if __name__ == "__main__":
    unittest.main()
