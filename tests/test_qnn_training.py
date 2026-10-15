from __future__ import annotations

import unittest

import numpy as np

from iprune.experiments.qnn_training import (
    stratified_calibration_test_split,
    stratified_subsample,
    train_qnn_circuit_with_diagnostics,
)
from iprune.gates import build_hea_gates_from_parameters


class QNNTrainingTests(unittest.TestCase):
    def test_calibration_and_test_are_disjoint_with_imbalanced_classes(self) -> None:
        states = np.eye(10, dtype=np.complex128)
        labels = np.array([0] * 7 + [1] * 3)
        cal_x, cal_y, test_x, test_y = stratified_calibration_test_split(
            states, labels, n_calibration=4, n_test=4, seed=42
        )
        self.assertEqual(len(cal_y), 4)
        self.assertEqual(len(test_y), 4)
        cal_ids = set(np.argmax(cal_x, axis=1).tolist())
        test_ids = set(np.argmax(test_x, axis=1).tolist())
        self.assertTrue(cal_ids.isdisjoint(test_ids))

    def test_stratified_subsample_preserves_both_classes(self) -> None:
        states = np.eye(10, dtype=np.complex128)
        labels = np.array([0] * 8 + [1] * 2)
        _, sampled = stratified_subsample(states, labels, n=4, seed=1)
        self.assertEqual(set(sampled.tolist()), {0, 1})

    def test_fixed_monitor_convergence_diagnostic_is_recorded(self) -> None:
        params = np.zeros((1, 2, 3), dtype=np.float64)
        _, circuit = build_hea_gates_from_parameters(
            params, include_cnot=True, include_identity_rot=False
        )
        states = np.eye(4, dtype=np.complex128)
        labels = np.array([0, 1, 0, 1], dtype=np.float64)
        result = train_qnn_circuit_with_diagnostics(
            circuit=circuit,
            train_states=states,
            train_labels=labels,
            monitor_states=states[:2],
            monitor_labels=labels[:2],
            n_qubits=2,
            max_steps=4,
            min_steps=4,
            lr=0.01,
            batch_size=2,
            seed=7,
            monitor_interval=1,
            monitor_window=2,
            convergence_tol=1e6,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.steps_run, 4)
        self.assertEqual(len(result.monitor_loss_history), 4)
        self.assertGreaterEqual(result.final_monitor_window_range, 0.0)


if __name__ == "__main__":
    unittest.main()
