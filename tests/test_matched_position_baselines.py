import unittest

import numpy as np

from iprune.context_states import compute_prefix_states
from iprune.experiments.matched_position_baselines import build_matched_position_baselines
from iprune.experiments.protocol import build_random_baseline_circuit
from iprune.gates import build_hea_gates


class MatchedPositionBaselineTests(unittest.TestCase):
    def test_every_method_uses_the_same_forward_replacement_budget(self):
        n_qubits = 2
        all_gates = build_hea_gates(
            n_qubits,
            depth=2,
            seed=17,
            rot_pool_k=4,
            rot_pool_sigma=0.01,
            include_cnot=True,
        )
        baseline = build_random_baseline_circuit(all_gates, seed=101)
        task_states = np.zeros((3, 2**n_qubits), dtype=np.complex128)
        task_states[:, 0] = 1.0
        prefixes = compute_prefix_states(
            realizable_circuit=baseline,
            task_states=task_states,
            n_qubits=n_qubits,
            device_name="default.qubit",
        )
        methods = build_matched_position_baselines(
            all_gates=all_gates,
            baseline_circuit=baseline,
            prefix_states_list=prefixes,
            task_states=task_states,
            Gq_diag=np.ones(2**n_qubits),
            replacement_count=2,
            n_qubits=n_qubits,
            seed=203,
        )
        self.assertEqual(len(methods), 5)
        for method in methods:
            changed = sum(
                selected is not original
                for selected, original in zip(method.selected_circuit, baseline)
            )
            self.assertEqual(method.replacement_count, 2)
            self.assertEqual(changed, 2)
            self.assertEqual(len(method.replaced_positions), 2)


if __name__ == "__main__":
    unittest.main()
