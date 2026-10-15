from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict

from .qgeometry import dq_task_conditioned
from .pennylane_backend import unitary_from_block


@dataclass
class PruneResult:
    kept_blocks: List[Callable[[], None]]
    pruned_indices: List[int]
    L: int
    dq_values: List[float]


def choose_reference_first(n: int) -> int:
    return 0


def choose_reference_medoid(
    U_list: List[np.ndarray],
    states: np.ndarray,
    Gq_diag: np.ndarray,
) -> int:
    n = len(U_list)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = dq_task_conditioned(U_list[i], U_list[j], states, Gq_diag)
            D[i, j] = D[j, i] = d
    return int(np.argmin(D.sum(axis=1)))


def q_iPrune_blocks(
    blocks: List[Callable[[], None]],
    *,
    n_qubits: int,
    device_name: str,
    states: np.ndarray,
    Gq_diag: np.ndarray,
    epsilon_q: float,
    reference: str = "first",
    max_pairs: Optional[int] = None,
) -> PruneResult:
    """
    Structured pruning at block/layer level.
    Keep one representative and remove blocks that are epsilon-redundant to it.
    """
    # Precompute unitaries for blocks
    U_list = [unitary_from_block(b, n_qubits, device_name=device_name) for b in blocks]

    if reference == "medoid" and len(blocks) <= 80:
        ref_idx = choose_reference_medoid(U_list, states, Gq_diag)
    else:
        ref_idx = choose_reference_first(len(blocks))

    U_ref = U_list[ref_idx]
    kept = [blocks[ref_idx]]
    pruned = []
    dq_vals = []
    L = 0
    compared = 0

    for i, (blk, U) in enumerate(zip(blocks, U_list)):
        if i == ref_idx:
            continue
        if max_pairs is not None and compared >= max_pairs:
            # conservative: keep remaining
            kept.append(blk)
            continue

        d = dq_task_conditioned(U_ref, U, states, Gq_diag)
        dq_vals.append(d)
        compared += 1

        if d <= epsilon_q:
            pruned.append(i)
            L += 1
        else:
            kept.append(blk)

    return PruneResult(kept_blocks=kept, pruned_indices=pruned, L=L, dq_values=dq_vals)
