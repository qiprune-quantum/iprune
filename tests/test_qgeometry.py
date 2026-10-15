from __future__ import annotations

import unittest

import numpy as np

from iprune.qgeometry import (
    build_Gq_diag_popcount,
    dq_task_conditioned,
    epsilon_from_delta,
    task_conditioned_output_angles,
)


class QGeometryTests(unittest.TestCase):
    def test_identical_unitaries_have_zero_angle(self) -> None:
        identity = np.eye(2, dtype=np.complex128)
        states = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.complex128)
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        Gq, _ = build_Gq_diag_popcount(q=1.2, n_qubits=1)
        q_angles, physical_angles = task_conditioned_output_angles(
            identity, identity, states, Gq
        )
        np.testing.assert_allclose(q_angles, 0.0, atol=1e-7)
        np.testing.assert_allclose(physical_angles, 0.0, atol=1e-7)

    def test_q_one_is_the_physical_fubini_study_angle(self) -> None:
        identity = np.eye(2, dtype=np.complex128)
        z_gate = np.diag([1.0, -1.0]).astype(np.complex128)
        states = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.complex128)
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        Gq, _ = build_Gq_diag_popcount(q=1.0, n_qubits=1)
        q_angles, physical_angles = task_conditioned_output_angles(
            identity, z_gate, states, Gq
        )
        np.testing.assert_allclose(q_angles, physical_angles, atol=1e-12)

    def test_max_reduction_is_not_mean_reduction(self) -> None:
        identity = np.eye(2, dtype=np.complex128)
        z_gate = np.diag([1.0, -1.0]).astype(np.complex128)
        plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        states = np.stack([np.array([1.0, 0.0]), plus])
        Gq = np.ones(2, dtype=np.float64)
        mean_angle = dq_task_conditioned(identity, z_gate, states, Gq, reduction="mean")
        max_angle = dq_task_conditioned(identity, z_gate, states, Gq, reduction="max")
        self.assertAlmostEqual(mean_angle, np.pi / 4.0, places=7)
        self.assertAlmostEqual(max_angle, np.pi / 2.0, places=7)

    def test_weighted_overlap_is_well_defined_for_non_g_unitary(self) -> None:
        identity = np.eye(2, dtype=np.complex128)
        z_gate = np.diag([1.0, -1.0]).astype(np.complex128)
        plus = np.array([[1.0, 1.0]], dtype=np.complex128) / np.sqrt(2.0)
        Gq = np.array([0.4, 1.0], dtype=np.float64)
        q_angles, physical_angles = task_conditioned_output_angles(
            identity, z_gate, plus, Gq
        )
        self.assertTrue(np.isfinite(q_angles[0]))
        self.assertTrue(0.0 <= q_angles[0] <= np.pi / 2.0)
        self.assertAlmostEqual(physical_angles[0], np.pi / 2.0, places=7)

    def test_epsilon_is_monotone_in_delta(self) -> None:
        self.assertLess(epsilon_from_delta(0.01, 1.0), epsilon_from_delta(0.02, 1.0))


if __name__ == "__main__":
    unittest.main()
