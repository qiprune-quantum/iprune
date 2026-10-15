"""Static diagonal-QFI gate saliency for post-training deletion controls.

This module intentionally implements a *derived* QFI baseline rather than
claiming to reproduce a published gate-deletion algorithm.  For each trained
``Rot(a,b,c)`` location and every calibration state, it estimates the diagonal
pure-state QFI of the three Euler coordinates immediately after that location.
The suffix is omitted because a common unitary suffix preserves the QFI.  The
group score ``sum_a F_aa * wrap(theta_a)^2`` is the local quadratic estimate of
the state displacement incurred by sending that gate to identity.

It is a static, post-training, matched-count comparator.  It therefore has a
different decision rule from the dynamic hybrid-prefix q-iPrune path and does
not receive a q-iPrune certificate.
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from iprune.experiments.protocol import simulate_outputs
from iprune.experiments.vqe_finetune import rebuild_circuit_with_rot_params
from iprune.gates import GateOp


def _wrapped_angles(params: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(params, dtype=np.float64)))


def diagonal_qfi_gate_saliency_scores(
    *,
    circuit: Sequence[GateOp],
    task_states: np.ndarray,
    n_qubits: int,
    device: str = "default.qubit",
    finite_difference_step: float = 1e-4,
) -> Tuple[List[int], Dict[int, float], Dict[int, List[float]], float]:
    """Return a deterministic low-to-high static QFI gate ranking.

    The QFI convention is ``F_aa=4 Re(<d_a|d_a>-|<psi|d_a>|^2)``.  Derivatives
    use central finite differences of statevectors.  Scores are averaged over
    the supplied task ensemble and use a stable position-id tie break.
    """
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive")
    template = list(circuit)
    rot_indices = [index for index, gate in enumerate(template) if gate.gate_type == "Rot"]
    if not rot_indices:
        return [], {}, {}, 0.0
    params = np.asarray(
        [gate.params for gate in template if gate.gate_type == "Rot"], dtype=np.float64
    ).reshape(-1, 3)
    task_states = np.asarray(task_states, dtype=np.complex128)
    if task_states.ndim != 2 or task_states.shape[1] != 2**n_qubits:
        raise ValueError("task_states must have shape (M, 2**n_qubits)")
    if len(task_states) == 0:
        raise ValueError("QFI saliency requires at least one task state")

    scores: Dict[int, float] = {}
    diagonal_by_position: Dict[int, List[float]] = {}
    started = perf_counter()
    for rot_index, gate_index in enumerate(rot_indices):
        gate = template[gate_index]
        position = int(gate.pos_id)
        # QFI is invariant under the untouched suffix, so only simulate the
        # physical prefix ending at the candidate rotation.
        prefix = template[: gate_index + 1]
        diagonals = np.zeros((len(task_states), 3), dtype=np.float64)
        for coordinate in range(3):
            plus = params.copy()
            minus = params.copy()
            plus[rot_index, coordinate] += finite_difference_step
            minus[rot_index, coordinate] -= finite_difference_step
            psi_plus = simulate_outputs(
                rebuild_circuit_with_rot_params(prefix, plus[: rot_index + 1]),
                task_states,
                n_qubits,
                device,
            )
            psi_minus = simulate_outputs(
                rebuild_circuit_with_rot_params(prefix, minus[: rot_index + 1]),
                task_states,
                n_qubits,
                device,
            )
            derivative = (psi_plus - psi_minus) / (2.0 * finite_difference_step)
            # The central state estimate at a unitary circuit point is the
            # normalized midpoint to numerical precision.
            psi = 0.5 * (psi_plus + psi_minus)
            derivative_norm = np.sum(np.conj(derivative) * derivative, axis=1).real
            parallel = np.sum(np.conj(psi) * derivative, axis=1)
            diagonals[:, coordinate] = np.maximum(
                0.0, 4.0 * (derivative_norm - np.abs(parallel) ** 2)
            )
        mean_diagonal = np.mean(diagonals, axis=0)
        angles = _wrapped_angles(params[rot_index])
        scores[position] = float(np.dot(mean_diagonal, angles * angles))
        diagonal_by_position[position] = [float(value) for value in mean_diagonal]

    ranking = sorted(scores, key=lambda position: (scores[position], position))
    return ranking, scores, diagonal_by_position, (perf_counter() - started) * 1000.0
