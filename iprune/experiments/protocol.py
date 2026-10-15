from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Callable
import numpy as np
import pennylane as qml

from iprune.qgeometry import (
    lambda_from_gamma, q_from_lambda,
    build_Gq_diag_popcount, epsilon_from_delta
)
from iprune.context_states import compute_prefix_states
from iprune.pruning_positionaware import q_iPrune_positionaware_candidate_pools
from iprune.metrics import bound_rhs
from iprune.gates import is_identity_rot


@dataclass
class PaperProtocolConfig:
    device: str = "default.qubit"
    n_qubits: int = 4
    depth: int = 12

    # Task-ensemble sampling
    M: int = 50
    seed: int = 42

    # Paper parameters
    gamma: float = 0.05
    alpha: float = 0.6
    qbeta: float = 0.0

    # Deviation tolerance in the paper: delta in {0.01, 0.02}
    delta: float = 0.02

    # Candidate pool (for controlled redundancy experiments)
    rot_pool_k: int = 4
    rot_pool_sigma: float = 0.003
    include_cnot: bool = True


def trace_distance_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    """Pure-state trace distance: sqrt(1 - |<psi|phi>|^2)."""
    ov = np.abs(np.vdot(psi, phi))
    ov = float(np.clip(ov, 0.0, 1.0))
    return float(np.sqrt(1.0 - ov * ov))


import numpy as np

def _trace_distance_pure_states(psi: np.ndarray, phi: np.ndarray) -> float:
    # For pure states: D( |psi><psi|, |phi><phi| ) = sqrt(1 - |<psi|phi>|^2 )
    norm_product = float(np.linalg.norm(psi) * np.linalg.norm(phi))
    if norm_product == 0.0:
        raise ValueError("Trace distance is undefined for a zero state vector.")
    overlap = float(np.clip(np.abs(np.vdot(psi, phi)) / norm_product, 0.0, 1.0))
    infidelity = 1.0 - overlap * overlap
    # Independent simulator calls can leave an O(machine-epsilon) normalization
    # residue even for the exact same circuit.  Do not report that as physical
    # drift (sqrt(eps) would otherwise appear as an artificial ~1e-8 distance).
    if infidelity <= 32.0 * np.finfo(np.float64).eps:
        return 0.0
    return float(np.sqrt(infidelity))

def trace_distance_stats(states_a: np.ndarray, states_b: np.ndarray) -> tuple[float, float]:
    ds = [_trace_distance_pure_states(states_a[i], states_b[i]) for i in range(states_a.shape[0])]
    return float(np.mean(ds)), float(np.max(ds))



def simulate_outputs(
    circuit_gates,
    task_states: np.ndarray,
    n_qubits: int,
    device_name: str,
) -> np.ndarray:
    """Apply circuit to each |psi_k> and return final states."""
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


def build_random_baseline_circuit(
    all_gates,
    *,
    seed: int,
    exclude_identity_rot: bool = True,
) -> list:
    """
    Paper-aligned *random* baseline selection:
    - For each position pos_id, choose ONE candidate uniformly.
    - Identity is an optional deletion/reference candidate, not a random
      baseline architecture choice, and is excluded by default.
    - This makes the baseline realizable and allows meaningful drift verification.
    """
    rng = np.random.default_rng(seed)
    pools = {}
    for g in all_gates:
        pools.setdefault(g.pos_id, []).append(g)

    P = len(pools)
    baseline = []
    for pos in range(P):
        cand_list = pools[pos]
        if exclude_identity_rot:
            non_identity = [g for g in cand_list if not is_identity_rot(g)]
            if non_identity:
                cand_list = non_identity
        k = int(rng.integers(0, len(cand_list)))
        baseline.append(cand_list[k])

    for idx, g in enumerate(baseline):
        if g.pos_id != idx:
            raise ValueError(f"Baseline circuit must be ordered by pos_id. got pos_id={g.pos_id} at idx={idx}")
    return baseline


def summarize_group_stats(group_stats: dict) -> dict:
    dq_max_list = []
    dq_mean_list = []
    pools = 0
    for _key, st in group_stats.items():
        if st.get("group_size", 0) > 1:
            pools += 1
            dq_mean_list.append(st.get("dq_mean", 0.0))
            dq_max_list.append(st.get("dq_max", 0.0))
    return {
        "num_pools": int(pools),
        "dq_mean_avg": float(np.mean(dq_mean_list)) if dq_mean_list else 0.0,
        "dq_max_avg": float(np.mean(dq_max_list)) if dq_max_list else 0.0,
        "dq_max_global": float(np.max(dq_max_list)) if dq_max_list else 0.0,
    }


def make_q_geometry(cfg: PaperProtocolConfig) -> Tuple[float, float, np.ndarray, float, float]:
    """
    Returns:
      lambda, q, Gq_diag, Mq, epsilon_q
    """
    lam = lambda_from_gamma(cfg.gamma, cfg.alpha)
    q = q_from_lambda(lam, cfg.qbeta)
    Gq_diag, Mq = build_Gq_diag_popcount(q=q, n_qubits=cfg.n_qubits)
    # Paper: epsilon_q = arcsin(delta * Mq / 2)
    eps_q = epsilon_from_delta(cfg.delta, Mq)
    return float(lam), float(q), Gq_diag, float(Mq), float(eps_q)


def run_positionaware_paper_protocol(
    *,
    cfg: PaperProtocolConfig,
    all_gates_builder: Callable[[PaperProtocolConfig], List[Any]],
    task_states: np.ndarray,
    return_artifacts: bool = False,
) -> Any:
    """
    Core paper-aligned verification loop:
      1) Construct candidate pools (all_gates).
      2) Random baseline circuit (one gate per pos_id).
      3) Prefix states from baseline circuit => D_pos for each position.
      4) Position-aware pruning on candidate pools using dq and epsilon_q.
      5) Evaluate drift between baseline and pruned circuits on task ensemble.
      6) Compare with theorem bound RHS = min(1, 2L/Mq * sin(eps_q)).
    """
    lam, q, Gq_diag, Mq, eps_q = make_q_geometry(cfg)

    all_gates = all_gates_builder(cfg)
    baseline_circuit = build_random_baseline_circuit(all_gates, seed=cfg.seed + 999)

    prefix_states_list = compute_prefix_states(
        realizable_circuit=baseline_circuit,
        task_states=task_states,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
    )

    res = q_iPrune_positionaware_candidate_pools(
        all_gates,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
        prefix_states_list=prefix_states_list,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
        baseline_circuit=baseline_circuit,
    )

    out_base = simulate_outputs(baseline_circuit, task_states, cfg.n_qubits, cfg.device)
    out_prun = simulate_outputs(res.selected_circuit, task_states, cfg.n_qubits, cfg.device)

    dists = np.array(
        [_trace_distance_pure_states(out_base[k], out_prun[k]) for k in range(cfg.M)],
        dtype=np.float64,
    )
    d_mean = float(np.mean(dists))
    d_max = float(np.max(dists))

    rhs_raw = float(bound_rhs(eps_q, Mq, res.replaced_count))
    rhs_clip = float(min(1.0, rhs_raw))

    gsum = summarize_group_stats(res.group_stats)
    prunable_candidates = sum(g.gate_type == "Rot" for g in all_gates)
    identity_candidates = sum(is_identity_rot(g) for g in all_gates)
    redundant_identity_candidates = sum(
        is_identity_rot(all_gates[index]) for index in res.replaced_indices
    )
    redundant_ordinary_candidates = int(res.replaced_count) - int(
        redundant_identity_candidates
    )

    pools = {}
    for gate in all_gates:
        key = (gate.pos_id, gate.gate_type, tuple(sorted(gate.wires)))
        pools.setdefault(key, []).append(gate)
    baseline_candidate_indices = []
    for gate in baseline_circuit:
        key = (gate.pos_id, gate.gate_type, tuple(sorted(gate.wires)))
        pool = pools[key]
        if len(pool) > 1:
            baseline_candidate_indices.append(
                next(index for index, candidate in enumerate(pool) if candidate is gate)
            )

    report = {
        # paper params
        "device": cfg.device,
        "n_qubits": cfg.n_qubits,
        "depth": cfg.depth,
        "M": cfg.M,
        "seed": cfg.seed,
        "gamma": cfg.gamma,
        "alpha": cfg.alpha,
        "lambda": lam,
        "qbeta": cfg.qbeta,
        "q": q,
        "Mq": Mq,
        "delta": cfg.delta,
        "epsilon_q": eps_q,

        # counts & ratios
        "all_candidates": int(len(all_gates)),
        "prunable_candidates": int(prunable_candidates),
        "positions": int(len(baseline_circuit)),
        "L": int(res.replaced_count),
        "candidate_replacements": int(res.replaced_count),
        "selected_replacement_count": int(res.selected_replacement_count),
        "replace_ratio_over_candidates": float(
            res.replaced_count / max(1, prunable_candidates)
        ),
        "ordinary_rot_candidates": int(prunable_candidates - identity_candidates),
        "identity_candidates": int(identity_candidates),
        "redundant_ordinary_candidates": int(redundant_ordinary_candidates),
        "redundant_identity_candidates": int(redundant_identity_candidates),
        "reference_candidate_policy": "first_candidate_index_0",
        "baseline_candidate_indices": baseline_candidate_indices,

        # verification metrics
        "trace_dist_mean": d_mean,
        "trace_dist_max": d_max,
        "rhs_raw": rhs_raw,
        "rhs_clipped": rhs_clip,

        # dq summaries
        **gsum,
    }
    if return_artifacts:
        # Attach read-only run artifacts so comparison runners can reuse the
        # exact candidate objects and prefix ensemble without rebuilding a
        # second, merely equivalent random experiment.
        res.all_gates = all_gates
        res.prefix_states_list = prefix_states_list
        return report, baseline_circuit, res
    return report
