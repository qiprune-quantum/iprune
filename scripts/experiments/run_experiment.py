from __future__ import annotations
import numpy as np

from .config import ExperimentConfig
from .task_ensemble import TaskEnsemble
from .qgeometry import (
    lambda_from_gamma, q_from_lambda,
    build_Gq_diag_popcount, epsilon_from_delta
)
from .circuits import build_hea_blocks
from .pennylane_backend import unitary_of_full_circuit
from .pruning import q_iPrune_blocks
from .metrics import average_overlap_fidelity, pruning_ratio, bound_rhs


def main(cfg: ExperimentConfig):
    dim = 2 ** cfg.n_qubits

    lam = lambda_from_gamma(cfg.gamma, cfg.alpha)
    q = q_from_lambda(lam, cfg.qbeta)

    Gq_diag, Mq = build_Gq_diag_popcount(q=q, n_qubits=cfg.n_qubits)
    eps_q = epsilon_from_delta(cfg.pruning.delta, Mq)

    D = TaskEnsemble.from_random(dim=dim, M=cfg.M, seed=cfg.seed)

    blocks, params = build_hea_blocks(cfg.n_qubits, cfg.depth, seed=cfg.seed)

    # Full unitary original
    U_full = unitary_of_full_circuit(blocks, cfg.n_qubits, device_name=cfg.device)

    # Prune (block-level)
    res = q_iPrune_blocks(
        blocks,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
        states=D.states,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
        reference=cfg.pruning.reference,
        max_pairs=cfg.pruning.max_pairs_per_group,
    )

    # Full unitary pruned
    U_pruned = unitary_of_full_circuit(res.kept_blocks, cfg.n_qubits, device_name=cfg.device)

    F = average_overlap_fidelity(U_full, U_pruned, D.states)
    pr = pruning_ratio(len(blocks), len(res.kept_blocks))
    bound = bound_rhs(eps_q, Mq, res.L)

    print("=== PennyLane q-iPrune (Block-level) ===")
    print(f"device={cfg.device}")
    print(f"n_qubits={cfg.n_qubits}, depth={cfg.depth}, M={cfg.M}, seed={cfg.seed}")
    print(f"gamma={cfg.gamma:.4f}, alpha={cfg.alpha:.3f} -> lambda={lam:.4f}")
    print(f"qbeta={cfg.qbeta:.3f} -> q={q:.6f}")
    print(f"Mq={Mq:.4f}, delta={cfg.pruning.delta:.4f} -> epsilon_q={eps_q:.6f}")
    print(f"orig_blocks={len(blocks)}, kept={len(res.kept_blocks)}, pruned={len(res.pruned_indices)}, pruning_ratio={pr:.2%}")
    print(f"L(replaced)={res.L}")
    print(f"Avg overlap fidelity F={F:.6f}")
    print(f"Theory RHS (2L/Mq*sin(eps))={bound:.6f}")
    if res.dq_values:
        print(f"dq_mean={float(np.mean(res.dq_values)):.4f}, dq_max={float(np.max(res.dq_values)):.4f}")


if __name__ == "__main__":
    cfg = ExperimentConfig()
    main(cfg)
