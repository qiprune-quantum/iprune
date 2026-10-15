from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class FakeGate:
    name: str
    gate_type: str
    pos_id: int
    apply: np.ndarray
    identity: bool = False


def load_pruning_module():
    fake_gates = types.ModuleType("gates")
    fake_gates.GateOp = FakeGate
    fake_gates.is_identity_rot = lambda gate, tol=0.0: bool(gate.identity)

    fake_unitary = types.ModuleType("pennylane_unitary")
    fake_unitary.unitary_of_single_gate = (
        lambda gate_apply, n_qubits, device_name: np.asarray(gate_apply, dtype=np.complex128)
    )

    source = PROJECT_ROOT / "iprune" / "pruning_true.py"
    spec = importlib.util.spec_from_file_location("pruning_true_without_pennylane", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "iprune.gates": fake_gates,
            "iprune.pennylane_unitary": fake_unitary,
            "pruning_true_without_pennylane": module,
        },
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class TruePruningTests(unittest.TestCase):
    def test_trace_distance_treats_machine_precision_identical_paths_as_zero(self) -> None:
        module = load_pruning_module()
        # This normalized state has an overlap with itself slightly below one
        # in ordinary double precision, which used to yield a false O(1e-8)
        # trace distance after the square root.
        state = np.array([[0.1 + 0.2j, 0.3 - 0.4j, 0.5 + 0.6j]], dtype=np.complex128)
        state /= np.linalg.norm(state)
        distance = module._pure_state_trace_distances(state, state)
        self.assertEqual(float(distance[0]), 0.0)

    def test_L_counts_only_actual_forward_deletions(self) -> None:
        module = load_pruning_module()
        identity = np.eye(2, dtype=np.complex128)
        z_gate = np.diag([1.0, -1.0]).astype(np.complex128)

        baseline = [
            FakeGate("near_identity", "Rot", 0, identity, identity=False),
            FakeGate("important_z", "Rot", 1, z_gate, identity=False),
        ]
        identity_candidates = [
            FakeGate("id0", "Rot", 0, identity, identity=True),
            FakeGate("id1", "Rot", 1, identity, identity=True),
        ]
        plus = np.array([[1.0, 1.0]], dtype=np.complex128) / np.sqrt(2.0)

        result = module.q_iPrune_true_gate_pruning_by_identity(
            all_gates=baseline + identity_candidates,
            baseline_circuit=baseline,
            n_qubits=1,
            device_name="fake",
            task_states=plus,
            Gq_diag=np.ones(2),
            epsilon_q=0.1,
        )

        self.assertEqual(result.pruned_positions, 1)
        self.assertEqual(result.gate_count_before, 2)
        self.assertEqual(result.gate_count_after, 1)
        self.assertEqual(result.compact_circuit[0].name, "important_z")
        self.assertEqual(result.dq_violations_over_replaced, 0)
        self.assertEqual(result.physical_violations_over_replaced, 0)

    def test_fixed_k_q_geometry_can_change_ranking_without_changing_k(self) -> None:
        module = load_pruning_module()
        identity = np.eye(2, dtype=np.complex128)
        state = np.array(
            [[
                0.055166690350535716 - 0.14202478651373726j,
                0.5227471996839491 - 0.8387615641611654j,
            ]],
            dtype=np.complex128,
        )
        u0 = np.array(
            [
                [
                    -0.2159679667445502 + 0.8141390737483815j,
                    0.36973972511449194 + 0.3922090534505961j,
                ],
                [
                    0.3415113424548875 - 0.41701967448912175j,
                    0.27217332365836344 + 0.7971112067662875j,
                ],
            ],
            dtype=np.complex128,
        )
        u1 = np.array(
            [
                [
                    0.8244297833348513 - 0.0876173705808686j,
                    0.489972148412316 + 0.2693808131682066j,
                ],
                [
                    -0.3704930851122086 + 0.418776315718765j,
                    0.8057144333383498 - 0.19541116443877182j,
                ],
            ],
            dtype=np.complex128,
        )
        baseline = [
            FakeGate("u0", "Rot", 0, u0),
            FakeGate("u1", "Rot", 1, u1),
        ]
        identity_candidates = [
            FakeGate("id0", "Rot", 0, identity, identity=True),
            FakeGate("id1", "Rot", 1, identity, identity=True),
        ]

        q1 = module.q_iPrune_fixed_k_identity(
            all_gates=baseline + identity_candidates,
            baseline_circuit=baseline,
            n_qubits=1,
            device_name="fake",
            task_states=state,
            Gq_diag=np.ones(2),
            physical_epsilon=0.6,
            deletion_budget=1,
        )
        q3 = module.q_iPrune_fixed_k_identity(
            all_gates=baseline + identity_candidates,
            baseline_circuit=baseline,
            n_qubits=1,
            device_name="fake",
            task_states=state,
            Gq_diag=np.array([1.0 / 3.0, 1.0]),
            physical_epsilon=0.6,
            deletion_budget=1,
        )

        self.assertEqual(q1.selected_positions, [1])
        self.assertEqual(q3.selected_positions, [0])
        self.assertEqual(len(q1.selected_positions), 1)
        self.assertEqual(len(q3.selected_positions), 1)
        self.assertEqual(q1.physical_feasible_positions, [0, 1])
        self.assertEqual(q3.physical_feasible_positions, [0, 1])
        self.assertEqual(q1.budget_shortfall, 0)
        self.assertEqual(q3.budget_shortfall, 0)

    def test_fixed_k_rejects_an_infeasible_budget(self) -> None:
        module = load_pruning_module()
        identity = np.eye(2, dtype=np.complex128)
        baseline = [FakeGate("id_like", "Rot", 0, identity)]
        candidates = [FakeGate("id0", "Rot", 0, identity, identity=True)]
        state = np.array([[1.0, 0.0]], dtype=np.complex128)

        with self.assertRaisesRegex(ValueError, "fewer than deletion_budget"):
            module.q_iPrune_fixed_k_identity(
                all_gates=baseline + candidates,
                baseline_circuit=baseline,
                n_qubits=1,
                device_name="fake",
                task_states=state,
                Gq_diag=np.ones(2),
                physical_epsilon=0.1,
                deletion_budget=2,
            )

    def test_frozen_path_audit_bounds_independent_states(self) -> None:
        module = load_pruning_module()
        identity = np.eye(2, dtype=np.complex128)
        z_gate = np.diag([1.0, -1.0]).astype(np.complex128)
        baseline = [FakeGate("z", "Rot", 0, z_gate)]
        candidates = [FakeGate("id0", "Rot", 0, identity, identity=True)]
        states = np.array(
            [[1.0, 0.0], [1.0, 1.0]], dtype=np.complex128
        )
        states[1] /= np.linalg.norm(states[1])

        audit = module.audit_fixed_identity_deletion_path(
            all_gates=baseline + candidates,
            baseline_circuit=baseline,
            selected_positions=[0],
            n_qubits=1,
            device_name="fake",
            task_states=states,
            Gq_diag=np.ones(2),
            physical_epsilon=np.pi / 2,
        )

        self.assertEqual(audit.selected_positions, [0])
        self.assertAlmostEqual(audit.actual_trace_distance_max, 1.0, places=12)
        self.assertAlmostEqual(audit.path_trace_sum_max, 1.0, places=12)
        self.assertEqual(audit.physical_threshold_violations, 0)
        self.assertEqual(audit.physical_state_position_violations, 0)
        self.assertEqual(audit.path_bound_violations, 0)

    def test_iterative_rule_records_path_and_respects_optional_budget(self) -> None:
        module = load_pruning_module()
        identity = np.eye(2, dtype=np.complex128)
        z_gate = np.diag([1.0, -1.0]).astype(np.complex128)
        baseline = [
            FakeGate("z0", "Rot", 0, z_gate),
            FakeGate("z1", "Rot", 1, z_gate),
        ]
        candidates = [
            FakeGate("id0", "Rot", 0, identity, identity=True),
            FakeGate("id1", "Rot", 1, identity, identity=True),
        ]
        plus = np.array([[1.0, 1.0]], dtype=np.complex128) / np.sqrt(2.0)

        result = module.q_iPrune_iterative_identity_audited(
            all_gates=baseline + candidates,
            baseline_circuit=baseline,
            n_qubits=1,
            device_name="fake",
            task_states=plus,
            Gq_diag=np.ones(2),
            q_epsilon=np.pi / 2,
            physical_epsilon=np.pi / 2,
            path_budget=1.0,
        )

        self.assertEqual(result.accepted_positions, [0])
        self.assertEqual(result.rejected_by_path_budget, [1])
        self.assertAlmostEqual(result.path_trace_sum_max, 1.0, places=12)
        self.assertAlmostEqual(result.actual_trace_distance_max, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
