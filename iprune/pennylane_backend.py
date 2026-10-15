from __future__ import annotations
import numpy as np
import pennylane as qml
from typing import Callable


def unitary_from_block(
    block_fn: Callable[[], None],
    n_qubits: int,
    device_name: str = "default.qubit",
) -> np.ndarray:
    """
    Build the full unitary matrix of a block by applying it to computational basis states.
    Complexity: O(dim * circuit_cost). Suitable for n_qubits ~ <= 10 in research prototype.
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def apply_to_basis(basis_index: int):
        # Prepare |basis_index>
        qml.BasisState(np.array(list(map(int, np.binary_repr(basis_index, width=n_qubits))), dtype=int),
                       wires=list(range(n_qubits)))
        block_fn()
        return qml.state()

    dim = 2 ** n_qubits
    U = np.zeros((dim, dim), dtype=np.complex128)
    for b in range(dim):
        out = apply_to_basis(b)
        U[:, b] = np.asarray(out, dtype=np.complex128)
    return U


def unitary_of_full_circuit(
    blocks: list[Callable[[], None]],
    n_qubits: int,
    device_name: str = "default.qubit",
) -> np.ndarray:
    """
    Multiply unitaries of blocks to get full circuit unitary:
    U_full = U_last ... U_2 U_1 (right action on ket).
    """
    dim = 2 ** n_qubits
    U_full = np.eye(dim, dtype=np.complex128)
    for blk in blocks:
        U_blk = unitary_from_block(blk, n_qubits, device_name=device_name)
        U_full = U_blk @ U_full
    return U_full
