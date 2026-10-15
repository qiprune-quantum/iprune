from pathlib import Path
import unittest

import numpy as np

from iprune.context_states import compute_prefix_states
from iprune.gates import build_hea_gates
from iprune.experiments.protocol import build_random_baseline_circuit
from scripts.experiments.run_q_sensitivity_selection_diagnostic import (
    DEFAULT_SOURCE_CHECKPOINTS,
    _boundary_metrics,
    _jaccard,
    _compute_prefix_states_fast,
    _rank_correlations,
    _rank_positions,
    load_source_configurations,
)


class QSensitivitySelectionDiagnosticTests(unittest.TestCase):
    def test_rank_helpers_identical_and_reversed(self) -> None:
        reference = [10, 20, 30, 40]
        self.assertEqual(_rank_correlations(reference, reference), (1.0, 1.0))
        spearman, kendall = _rank_correlations(reference, list(reversed(reference)))
        self.assertEqual(spearman, -1.0)
        self.assertEqual(kendall, -1.0)

    def test_selection_helpers_and_boundaries(self) -> None:
        scores = {7: 0.3, 2: 0.1, 5: 0.2}
        order = _rank_positions(scores)
        self.assertEqual(order, [2, 5, 7])
        self.assertEqual(_jaccard([], []), 1.0)
        self.assertAlmostEqual(_jaccard([2, 5], [5, 7]), 1.0 / 3.0)
        metrics = _boundary_metrics(scores, order, 2)
        self.assertEqual(metrics["matched_k_score"], 0.2)
        self.assertEqual(metrics["matched_k_next_score"], 0.3)
        self.assertAlmostEqual(metrics["matched_k_boundary_margin"], 0.1)

    def test_historical_source_contains_160_complete_pairs(self) -> None:
        paths = [Path(path) for path in DEFAULT_SOURCE_CHECKPOINTS]
        configs = load_source_configurations(paths)
        self.assertEqual(len(configs), 160)
        self.assertTrue(
            all(
                len(config["source_qiprune_positions"])
                == config["source_selected_k"]
                for config in configs
            )
        )
        self.assertTrue(
            all(
                sorted(config["source_qiprune_positions"])
                == sorted(config["source_q1_positions"])
                for config in configs
            )
        )

    def test_fast_prefix_propagation_matches_qnode_reference(self) -> None:
        rng = np.random.default_rng(20260810)
        states = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        all_gates = build_hea_gates(
            2,
            2,
            seed=42,
            rot_pool_k=2,
            rot_pool_sigma=0.003,
            include_cnot=True,
        )
        circuit = build_random_baseline_circuit(all_gates, seed=1041)
        expected = compute_prefix_states(circuit, states, 2)
        actual = _compute_prefix_states_fast(circuit, states, 2)
        self.assertEqual(len(actual), len(expected))
        for actual_prefix, expected_prefix in zip(actual, expected):
            np.testing.assert_allclose(actual_prefix, expected_prefix, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
