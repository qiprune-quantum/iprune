from __future__ import annotations
import numpy as np
import pennylane as qml
from typing import Callable, List, Tuple


def build_hea_blocks(
    n_qubits: int,
    depth: int,
    seed: int = 42,
) -> Tuple[List[Callable[[], None]], np.ndarray]:
    """
    Return (blocks, params) where blocks is a list of callables that apply one layer.
    Each block captures a structured unit (layer).
    """
    rng = np.random.default_rng(seed)
    # params: (depth, n_qubits, 3) -> for Rot(phi, theta, omega)
    params = rng.uniform(0, 2 * np.pi, size=(depth, n_qubits, 3))

    blocks: List[Callable[[], None]] = []

    for d in range(depth):
        layer_params = params[d].copy()

        def make_block(lp=layer_params):
            def block():
                # single-qubit rotations
                for q in range(n_qubits):
                    a, b, c = lp[q]
                    qml.Rot(a, b, c, wires=q)
                # ring entanglers
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
            return block

        blocks.append(make_block())

    return blocks, params
