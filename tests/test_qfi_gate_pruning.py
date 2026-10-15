import numpy as np

from iprune.experiments.qfi_gate_pruning import diagonal_qfi_gate_saliency_scores
from iprune.gates import build_hea_gates_from_parameters


def test_diagonal_qfi_saliency_is_nonnegative_and_deterministic():
    params = np.array([[[0.3, -0.2, 0.1], [0.4, 0.5, -0.1]]])
    _, circuit = build_hea_gates_from_parameters(params, include_cnot=True)
    states = np.zeros((2, 4), dtype=np.complex128)
    states[0, 0] = 1.0
    states[1, 1] = 1.0
    first = diagonal_qfi_gate_saliency_scores(
        circuit=circuit, task_states=states, n_qubits=2
    )
    second = diagonal_qfi_gate_saliency_scores(
        circuit=circuit, task_states=states, n_qubits=2
    )
    ranking, scores, diagonals, _ = first
    assert ranking == second[0]
    assert ranking == sorted(scores, key=lambda position: (scores[position], position))
    assert all(value >= 0.0 for value in scores.values())
    assert all(len(value) == 3 and all(x >= 0.0 for x in value) for value in diagonals.values())
