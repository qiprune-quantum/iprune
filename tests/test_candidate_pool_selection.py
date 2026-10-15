from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from iprune.experiments.protocol import build_random_baseline_circuit
from iprune.pruning_positionaware import q_iPrune_positionaware_candidate_pools


def ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def fake_gate(name: str, theta: float, *, identity_params: bool = False):
    def apply():
        return None

    apply.matrix = ry(theta)
    return SimpleNamespace(
        name=name,
        gate_type="Rot",
        layer=0,
        pos_id=0,
        wires=(0,),
        params=(0.0, 0.0, 0.0) if identity_params else (0.0, theta, 0.0),
        apply=apply,
    )


class CandidatePoolSelectionTests(unittest.TestCase):
    def test_random_baseline_excludes_identity_candidate(self) -> None:
        candidate = fake_gate("candidate", 0.4)
        identity = fake_gate("identity", 0.0, identity_params=True)
        for seed in range(10):
            baseline = build_random_baseline_circuit([candidate, identity], seed=seed)
            self.assertIs(baseline[0], candidate)

    def test_selected_circuit_maps_the_executed_baseline_candidate(self) -> None:
        reference = fake_gate("reference", 0.0)
        close_baseline = fake_gate("close", 0.02)
        states = np.array([[1.0, 0.0]], dtype=np.complex128)

        with patch(
            "iprune.pruning_positionaware.unitary_of_single_gate",
            side_effect=lambda apply, **_: apply.matrix,
        ):
            result = q_iPrune_positionaware_candidate_pools(
                [reference, close_baseline],
                n_qubits=1,
                device_name="unused",
                prefix_states_list=[states],
                Gq_diag=np.ones(2),
                epsilon_q=0.1,
                baseline_circuit=[close_baseline],
            )

        self.assertIs(result.selected_circuit[0], reference)
        self.assertEqual(result.replaced_count, 1)
        self.assertEqual(result.selected_replacement_count, 1)

    def test_nonredundant_baseline_candidate_is_not_silently_replaced(self) -> None:
        reference = fake_gate("reference", 0.0)
        far_baseline = fake_gate("far", 1.0)
        states = np.array([[1.0, 0.0]], dtype=np.complex128)

        with patch(
            "iprune.pruning_positionaware.unitary_of_single_gate",
            side_effect=lambda apply, **_: apply.matrix,
        ):
            result = q_iPrune_positionaware_candidate_pools(
                [reference, far_baseline],
                n_qubits=1,
                device_name="unused",
                prefix_states_list=[states],
                Gq_diag=np.ones(2),
                epsilon_q=0.1,
                baseline_circuit=[far_baseline],
            )

        self.assertIs(result.selected_circuit[0], far_baseline)
        self.assertEqual(result.replaced_count, 0)
        self.assertEqual(result.selected_replacement_count, 0)


if __name__ == "__main__":
    unittest.main()
