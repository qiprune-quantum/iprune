from __future__ import annotations
import numpy as np
import pennylane as qml

from iprune.config import ExperimentConfig
from iprune.task_ensemble import TaskEnsemble
from iprune.qgeometry import (
    lambda_from_gamma, q_from_lambda,
    build_Gq_diag_popcount, epsilon_from_delta
)
from iprune.gates import build_hea_gates
from iprune.context_states import compute_prefix_states
from iprune.pruning_positionaware import q_iPrune_positionaware_candidate_pools
from iprune.metrics import bound_rhs


def simulate_outputs(
    circuit_gates,
    task_states: np.ndarray,
    n_qubits: int,
    device_name: str = "default.qubit",
) -> np.ndarray:
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def run(psi: np.ndarray):
        qml.StatePrep(psi, wires=list(range(n_qubits)))
        for g in circuit_gates:
            g.apply()
        return qml.state()

    M, dim = task_states.shape
    out = np.zeros((M, dim), dtype=np.complex128)
    for k in range(M):
        out[k] = np.asarray(run(task_states[k]), dtype=np.complex128)
    return out


def trace_distance_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    # For pure states: D(psi,phi) = sqrt(1 - |<psi|phi>|^2)
    ov = np.abs(np.vdot(psi, phi))
    ov = float(np.clip(ov, 0.0, 1.0))
    return float(np.sqrt(1.0 - ov * ov))


def build_random_baseline_circuit(all_gates, *, seed: int) -> list:
    """
    Random baseline: for each pos_id, randomly choose ONE candidate gate among its pool.
    This makes baseline differ from the reference-choice circuit, enabling drift measurement.

    Assumes pos_id are consecutive starting from 0, and candidates appear in all_gates in
    pos_id-major order (as built by build_hea_gates).
    """
    rng = np.random.default_rng(seed)

    # collect candidates per pos_id
    pools = {}
    for g in all_gates:
        pools.setdefault(g.pos_id, []).append(g)

    P = len(pools)
    baseline = []
    for pos in range(P):
        cand_list = pools[pos]
        k = int(rng.integers(0, len(cand_list)))
        baseline.append(cand_list[k])

    # sanity: baseline must be ordered by pos_id consecutive
    for idx, g in enumerate(baseline):
        if g.pos_id != idx:
            raise ValueError(
                f"Random baseline must be ordered by pos_id consecutive. "
                f"Found gate with pos_id={g.pos_id} at index={idx}."
            )
    return baseline


def main(cfg: ExperimentConfig):
    dim = 2 ** cfg.n_qubits

    # lambda, q
    lam = lambda_from_gamma(cfg.gamma, cfg.alpha)
    q = q_from_lambda(lam, cfg.qbeta)

    # Gq, Mq, epsilon
    Gq_diag, Mq = build_Gq_diag_popcount(q=q, n_qubits=cfg.n_qubits)
    eps_q = epsilon_from_delta(cfg.pruning.delta, Mq)

    # Task ensemble D
    D = TaskEnsemble.from_random(dim=dim, M=cfg.M, seed=cfg.seed)

    # Build candidate pools
    all_gates = build_hea_gates(
        cfg.n_qubits, cfg.depth, seed=cfg.seed,
        rot_pool_k=4,
        rot_pool_sigma=0.003,
        include_cnot=True,
    )

    # Random baseline: choose one candidate per pos_id
    baseline_circuit = build_random_baseline_circuit(all_gates, seed=cfg.seed + 999)

    # Prefix states computed on the baseline circuit positions (paper-aligned)
    prefix_states_list = compute_prefix_states(
        realizable_circuit=baseline_circuit,
        task_states=D.states,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
    )

    # Prune: replace candidates that are close to the reference candidate (idxs[0]) at each position
    res = q_iPrune_positionaware_candidate_pools(
        all_gates,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
        prefix_states_list=prefix_states_list,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
        baseline_circuit=baseline_circuit,
    )

    # Empirical verification: baseline vs pruned realizable circuit
    out_base = simulate_outputs(baseline_circuit, D.states, cfg.n_qubits, device_name=cfg.device)
    out_prun = simulate_outputs(res.selected_circuit, D.states, cfg.n_qubits, device_name=cfg.device)

    dists = np.array([trace_distance_pure(out_base[k], out_prun[k]) for k in range(cfg.M)], dtype=np.float64)
    d_max = float(np.max(dists))
    d_mean = float(np.mean(dists))

    # Theory RHS (use min(1, RHS) to respect trace-distance upper bound)
    rhs_raw = bound_rhs(eps_q, Mq, res.replaced_count)
    rhs = float(min(1.0, rhs_raw))

    print("=== PennyLane q-iPrune (Position-aware, candidate pools; RANDOM baseline) ===")
    print(f"device={cfg.device}")
    print(f"n_qubits={cfg.n_qubits}, depth={cfg.depth}, M={cfg.M}, seed={cfg.seed}")
    print(f"gamma={cfg.gamma:.4f}, alpha={cfg.alpha:.3f} -> lambda={lam:.4f}")
    print(f"qbeta={cfg.qbeta:.3f} -> q={q:.6f}")
    print(f"Mq={Mq:.4f}, delta={cfg.pruning.delta:.4f} -> epsilon_q={eps_q:.6f}")
    print(f"all_candidates={len(all_gates)}  positions={len(baseline_circuit)}")
    print(f"replaced(L)={res.replaced_count}  replace_ratio_over_candidates={res.replaced_count/len(all_gates):.2%}")
    print(f"Empirical trace distance (baseline vs pruned): mean={d_mean:.6f}, max={d_max:.6f}")
    print(f"Theory RHS raw={rhs_raw:.6f}  clipped(min(1,RHS))={rhs:.6f}")

    print("Group stats (key=(pos_id, type, wires)) [only pools]:")
    for key, st in sorted(res.group_stats.items(), key=lambda kv: kv[0]):
        if st["group_size"] > 1:
            print(
                f"  {key}: size={int(st['group_size'])}, replaced={int(st['replaced'])}, "
                f"dq_mean={st['dq_mean']:.6f}, dq_max={st['dq_max']:.6f}"
            )


if __name__ == "__main__":
    cfg = ExperimentConfig()
    main(cfg)
