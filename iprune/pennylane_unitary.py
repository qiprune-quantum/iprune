from __future__ import annotations
import numpy as np
import pennylane as qml
from typing import Callable


def unitary_of_single_gate(
    gate_apply: Callable[[], None],
    n_qubits: int,
    device_name: str = "default.qubit",
) -> np.ndarray:
    """
    得到该 gate 在 n_qubits 上的 full unitary 矩阵 (dim x dim)
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def apply_to_basis(basis_index: int):
        bits = np.array(list(map(int, np.binary_repr(basis_index, width=n_qubits))), dtype=int)
        qml.BasisState(bits, wires=list(range(n_qubits)))
        gate_apply()
        return qml.state()

    dim = 2 ** n_qubits
    U = np.zeros((dim, dim), dtype=np.complex128)
    for b in range(dim):
        out = apply_to_basis(b)
        U[:, b] = np.asarray(out, dtype=np.complex128)
    return U
