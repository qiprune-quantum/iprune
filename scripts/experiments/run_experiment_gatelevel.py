from __future__ import annotations
import numpy as np

from .config import ExperimentConfig
from .task_ensemble import TaskEnsemble
from .qgeometry import (
    lambda_from_gamma, q_from_lambda,
    build_Gq_diag_popcount, epsilon_from_delta
)
from .gates import build_hea_gates
from .pruning_gatelevel import q_iPrune_gates


def main(cfg: ExperimentConfig):
    dim = 2 ** cfg.n_qubits

    # lambda, q
    lam = lambda_from_gamma(cfg.gamma, cfg.alpha)
    q = q_from_lambda(lam, cfg.qbeta)

    # Gq and Mq
    Gq_diag, Mq = build_Gq_diag_popcount(q=q, n_qubits=cfg.n_qubits)

    # epsilon from delta
    eps_q = epsilon_from_delta(cfg.pruning.delta, Mq)

    # Task ensemble D (random for now; you can swap to data-encoding later)
    D = TaskEnsemble.from_random(dim=dim, M=cfg.M, seed=cfg.seed)

    # Build gate-level HEA list
    # 注意：为了让你“马上看到剪枝”，建议 tie_prob>0
    gates = build_hea_gates(
        cfg.n_qubits, cfg.depth, seed=cfg.seed,
        tie_prob=0.35,     # <-- 建议先用 0.2~0.5
        tie_sigma=0.01     # <-- 让它变成“近似冗余”
    )

    # Run q-iPrune gate-level
    res = q_iPrune_gates(
        gates,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
        states=D.states,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
        reference=cfg.pruning.reference,
        max_pairs_per_group=cfg.pruning.max_pairs_per_group,
        cache_unitaries=True,
    )

    orig = len(gates)
    kept = len(res.kept)
    pruned = len(res.pruned)
    ratio = 1.0 - kept / float(orig)

    rhs = (2.0 * res.L / max(Mq, 1e-12)) * np.sin(eps_q)

    print("=== PennyLane q-iPrune (Gate-level) ===")
    print(f"device={cfg.device}")
    print(f"n_qubits={cfg.n_qubits}, depth={cfg.depth}, M={cfg.M}, seed={cfg.seed}")
    print(f"gamma={cfg.gamma:.4f}, alpha={cfg.alpha:.3f} -> lambda={lam:.4f}")
    print(f"qbeta={cfg.qbeta:.3f} -> q={q:.6f}")
    print(f"Mq={Mq:.4f}, delta={cfg.pruning.delta:.4f} -> epsilon_q={eps_q:.6f}")
    print(f"orig_gates={orig}, kept={kept}, pruned={pruned}, pruning_ratio={ratio:.2%}")
    print(f"L(replaced)={res.L}")
    print(f"Theory RHS (2L/Mq*sin(eps))={rhs:.6f}")

    print("Group stats (by wires):")
    for wires, st in sorted(res.group_stats.items(), key=lambda kv: (len(kv[0]), kv[0])):
        print(
            f"  wires={wires} size={int(st['group_size'])} kept={int(st['kept'])} pruned={int(st['pruned'])} "
            f"dq_mean={st['dq_mean']:.4f} dq_max={st['dq_max']:.4f}"
        )


if __name__ == "__main__":
    cfg = ExperimentConfig()
    main(cfg)
