from __future__ import annotations
import os
import csv
import math
import argparse
import numpy as np
import pennylane as qml

from iprune.task_ensemble import TaskEnsemble
from iprune.qgeometry import (
    lambda_from_gamma, q_from_lambda,
    build_Gq_diag_popcount, epsilon_from_delta
)
from iprune.gates import build_hea_gates
from iprune.context_states import compute_prefix_states
from iprune.pruning_positionaware import q_iPrune_positionaware_candidate_pools
from iprune.metrics import bound_rhs


def trace_distance_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    ov = np.abs(np.vdot(psi, phi))
    ov = float(np.clip(ov, 0.0, 1.0))
    return float(np.sqrt(1.0 - ov * ov))


def simulate_outputs(circuit_gates, task_states: np.ndarray, n_qubits: int, device_name: str) -> np.ndarray:
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


def build_random_baseline_circuit(all_gates, *, seed: int) -> list:
    rng = np.random.default_rng(seed)
    pools = {}
    for g in all_gates:
        pools.setdefault(g.pos_id, []).append(g)

    P = len(pools)
    baseline = []
    for pos in range(P):
        cand_list = pools[pos]
        k = int(rng.integers(0, len(cand_list)))
        baseline.append(cand_list[k])

    for idx, g in enumerate(baseline):
        if g.pos_id != idx:
            raise ValueError(f"Random baseline pos_id mismatch: pos_id={g.pos_id} at index={idx}")
    return baseline


def summarize_group_stats(group_stats: dict) -> dict:
    """Summarize dq stats over all pools (size>1)."""
    dq_max_list = []
    dq_mean_list = []
    pool_sizes = []
    replaced_list = []

    for _key, st in group_stats.items():
        if st.get("group_size", 0) > 1:
            pool_sizes.append(st["group_size"])
            replaced_list.append(st["replaced"])
            dq_mean_list.append(st.get("dq_mean", 0.0))
            dq_max_list.append(st.get("dq_max", 0.0))

    if not dq_max_list:
        return {
            "num_pools": 0,
            "dq_mean_avg": 0.0,
            "dq_max_avg": 0.0,
            "dq_max_global": 0.0,
        }

    return {
        "num_pools": int(len(dq_max_list)),
        "dq_mean_avg": float(np.mean(dq_mean_list)),
        "dq_max_avg": float(np.mean(dq_max_list)),
        "dq_max_global": float(np.max(dq_max_list)),
    }


def run_one_setting(
    *,
    device_name: str,
    n_qubits: int,
    depth: int,
    M: int,
    seed: int,
    gamma: float,
    alpha: float,
    qbeta: float,
    delta: float,
    rot_pool_k: int,
    rot_pool_sigma: float,
    include_cnot: bool = True,
    baseline_seed_offset: int = 999,
) -> dict:
    dim = 2 ** n_qubits

    lam = lambda_from_gamma(gamma, alpha)
    q = q_from_lambda(lam, qbeta)
    Gq_diag, Mq = build_Gq_diag_popcount(q=q, n_qubits=n_qubits)
    eps_q = epsilon_from_delta(delta, Mq)

    D = TaskEnsemble.from_random(dim=dim, M=M, seed=seed)

    all_gates = build_hea_gates(
        n_qubits, depth, seed=seed,
        rot_pool_k=rot_pool_k,
        rot_pool_sigma=rot_pool_sigma,
        include_cnot=include_cnot,
    )

    baseline_circuit = build_random_baseline_circuit(all_gates, seed=seed + baseline_seed_offset)

    prefix_states_list = compute_prefix_states(
        realizable_circuit=baseline_circuit,
        task_states=D.states,
        n_qubits=n_qubits,
        device_name=device_name,
    )

    res = q_iPrune_positionaware_candidate_pools(
        all_gates,
        n_qubits=n_qubits,
        device_name=device_name,
        prefix_states_list=prefix_states_list,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
        baseline_circuit=baseline_circuit,
    )

    out_base = simulate_outputs(baseline_circuit, D.states, n_qubits, device_name=device_name)
    out_prun = simulate_outputs(res.selected_circuit, D.states, n_qubits, device_name=device_name)

    dists = np.array([trace_distance_pure(out_base[k], out_prun[k]) for k in range(M)], dtype=np.float64)
    d_mean = float(np.mean(dists))
    d_max = float(np.max(dists))

    rhs_raw = float(bound_rhs(eps_q, Mq, res.replaced_count))
    rhs_clip = float(min(1.0, rhs_raw))

    gsum = summarize_group_stats(res.group_stats)

    return {
        "device": device_name,
        "n_qubits": n_qubits,
        "depth": depth,
        "M": M,
        "seed": seed,
        "gamma": gamma,
        "alpha": alpha,
        "lambda": float(lam),
        "qbeta": qbeta,
        "q": float(q),
        "Mq": float(Mq),
        "delta": delta,
        "epsilon_q": float(eps_q),
        "rot_pool_k": rot_pool_k,
        "rot_pool_sigma": rot_pool_sigma,
        "include_cnot": int(include_cnot),
        "all_candidates": int(len(all_gates)),
        "positions": int(len(baseline_circuit)),
        "L": int(res.replaced_count),
        "replace_ratio_over_candidates": float(res.replaced_count / max(1, len(all_gates))),
        "trace_dist_mean": d_mean,
        "trace_dist_max": d_max,
        "rhs_raw": rhs_raw,
        "rhs_clipped": rhs_clip,
        **gsum,
    }


def write_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_list_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser("q-iPrune paper-aligned ablation runner (PennyLane)")
    ap.add_argument("--device", type=str, default="default.qubit")
    ap.add_argument("--n_qubits", type=int, default=4)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    # theory params
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--qbeta", type=float, default=0.0)

    # baseline setting
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--rot_pool_k", type=int, default=4)
    ap.add_argument("--rot_pool_sigma", type=float, default=0.003)

    # sweeps
    ap.add_argument("--sweep", type=str, choices=["sigma", "delta", "k", "all"], default="all")
    ap.add_argument("--sigmas", type=str, default="0.001,0.002,0.003,0.005,0.008,0.01")
    ap.add_argument("--deltas", type=str, default="0.005,0.01,0.02,0.05")
    ap.add_argument("--ks", type=str, default="2,4,8")

    ap.add_argument("--out_dir", type=str, default="runs_ablation")
    ap.add_argument("--tag", type=str, default="paper_aligned")
    ap.add_argument("--include_cnot", action="store_true")

    args = ap.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Creating {args.out_dir}")


    rows = []
    out_csv = os.path.join(args.out_dir, f"ablation_{args.tag}.csv")

    sigmas = parse_list_floats(args.sigmas)
    deltas = parse_list_floats(args.deltas)
    ks = parse_list_ints(args.ks)

    def run_and_append(delta, k, sigma, sweep_name):
        r = run_one_setting(
            device_name=args.device,
            n_qubits=args.n_qubits,
            depth=args.depth,
            M=args.M,
            seed=args.seed,
            gamma=args.gamma,
            alpha=args.alpha,
            qbeta=args.qbeta,
            delta=delta,
            rot_pool_k=k,
            rot_pool_sigma=sigma,
            include_cnot=args.include_cnot,
        )
        r["sweep"] = sweep_name
        rows.append(r)
        print(
            f"[{sweep_name}] delta={delta} k={k} sigma={sigma} "
            f"=> L={r['L']} maxD={r['trace_dist_max']:.6f} rhs={r['rhs_clipped']:.3f}"
        )

    if args.sweep in ("sigma", "all"):
        for sigma in sigmas:
            run_and_append(args.delta, args.rot_pool_k, sigma, "sigma")

    if args.sweep in ("delta", "all"):
        for delta in deltas:
            run_and_append(delta, args.rot_pool_k, args.rot_pool_sigma, "delta")

    if args.sweep in ("k", "all"):
        for k in ks:
            run_and_append(args.delta, k, args.rot_pool_sigma, "k")

    write_csv(out_csv, rows)
    print(f"\nSaved CSV -> {out_csv}")


if __name__ == "__main__":
    main()
