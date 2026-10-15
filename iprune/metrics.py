from __future__ import annotations
import numpy as np


def average_overlap_fidelity(U: np.ndarray, V: np.ndarray, states: np.ndarray) -> float:
    """
    F = 1/M sum_k | <psi_k | U^† V | psi_k> |
    """
    W = U.conj().T @ V
    acc = 0.0
    for psi in states:
        acc += float(np.abs(np.vdot(psi, W @ psi)))
    return acc / float(states.shape[0])


def pruning_ratio(n_orig: int, n_kept: int) -> float:
    return 1.0 - (n_kept / float(n_orig))


def bound_rhs(epsilon: float, Mq: float, L: int) -> float:
    return float((2.0 * L / max(Mq, 1e-12)) * np.sin(epsilon))
