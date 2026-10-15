from __future__ import annotations
import numpy as np
import pennylane as qml
from typing import List
from iprune.gates import GateOp


def compute_prefix_states(
    realizable_circuit: List[GateOp],
    task_states: np.ndarray,   # (M, dim)
    n_qubits: int,
    device_name: str = "default.qubit",
) -> List[np.ndarray]:
    """
    Strict prefix-state generator aligned with paper:

      For position p (pos_id == p), define prefix unitary U_{<p}
      as the product of circuit gates before that position (in realizable_circuit order).

    We return prefix_states_list where prefix_states_list[p] has shape (M, dim):
      phi_k^{(p)} = U_{<p} |psi_k>

    NOTE:
      - realizable_circuit MUST have exactly one gate per position,
        ordered by increasing pos_id (as constructed by select_first_candidate_per_position()).
      - Let P = number of positions. We return a list of length P+1:
          prefix_states_list[p] for p=0..P (P is full prefix = whole circuit).
    """
    dev = qml.device(device_name, wires=n_qubits)
    M, dim = task_states.shape
    P = len(realizable_circuit)

    # sanity: positions should be consecutive 0..P-1
    for idx, g in enumerate(realizable_circuit):
        if g.pos_id != idx:
            raise ValueError(
                f"Realizable circuit must be ordered by pos_id consecutive. "
                f"Found gate with pos_id={g.pos_id} at index={idx}."
            )

    def _apply_prefix(upto_pos: int):
        for p in range(upto_pos):
            realizable_circuit[p].apply()

    @qml.qnode(dev)
    def run_prefix(psi: np.ndarray, upto_pos: int):
        qml.StatePrep(psi, wires=list(range(n_qubits)))
        _apply_prefix(upto_pos)
        return qml.state()

    prefix_states_list: List[np.ndarray] = []
    for p in range(P + 1):
        arr = np.zeros((M, dim), dtype=np.complex128)
        for k in range(M):
            arr[k] = np.asarray(run_prefix(task_states[k], p), dtype=np.complex128)
        prefix_states_list.append(arr)

    return prefix_states_list
