from __future__ import annotations

import os, json, time
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np

from iprune.gates import GateOp, is_identity_rot
from iprune.gates import build_hea_gates

from iprune.experiments.protocol import (
    PaperProtocolConfig,
    simulate_outputs,
    make_q_geometry,
)

from iprune.experiments.tasks_classification import (
    ToyVisionConfig,
    build_mnist49_splits,
    build_fashion_sandal_boot_splits,
    build_bas_splits,
    sample_task_ensemble_from_validation,
)
from iprune.experiments.tasks_vqe import VQETwitterConfig, sample_intermediate_states_from_vqe

from iprune.experiments.task_metrics import (
    evaluate_classification_accuracy_drop,
    evaluate_vqe_energy_deviation,
)

from iprune.context_states import compute_prefix_states
from iprune.pruning_true import q_iPrune_true_gate_pruning_by_identity


# -------------------------
# Gate builders / baselines
# -------------------------
def all_gates_builder(cfg: PaperProtocolConfig):
    # IMPORTANT: make sure gates.py provides identity candidate in each Rot pool,
    # otherwise "true pruning by identity" cannot actually prune.
    return build_hea_gates(
        cfg.n_qubits, cfg.depth, seed=cfg.seed,
        rot_pool_k=cfg.rot_pool_k,
        rot_pool_sigma=cfg.rot_pool_sigma,
        include_cnot=cfg.include_cnot,
    )


def build_random_baseline_circuit_no_identity(all_gates: List[GateOp], seed: int) -> List[GateOp]:
    """Random baseline circuit, but never select identity Rot as baseline gate."""
    rng = np.random.default_rng(seed)
    by_pos: Dict[int, List[GateOp]] = {}
    max_pos = -1
    for g in all_gates:
        max_pos = max(max_pos, g.pos_id)
        if is_identity_rot(g):
            continue
        by_pos.setdefault(g.pos_id, []).append(g)

    P = max_pos + 1
    baseline: List[GateOp] = []
    for pos in range(P):
        cands = by_pos.get(pos, [])
        if not cands:
            raise RuntimeError(f"No non-identity candidates for pos_id={pos}.")
        baseline.append(cands[int(rng.integers(len(cands)))])

    # sanity: positions should be consecutive
    for idx, g in enumerate(baseline):
        if g.pos_id != idx:
            raise RuntimeError(f"Baseline circuit pos_id must be consecutive. Found pos_id={g.pos_id} at index={idx}.")
    return baseline


def _attach_generation_params(out: Dict[str, Any], cfg: PaperProtocolConfig) -> None:
    out["rot_pool_k"] = int(cfg.rot_pool_k)
    out["rot_pool_sigma"] = float(cfg.rot_pool_sigma)
    out["include_cnot"] = bool(cfg.include_cnot)


# -------------------------
# Trace distance utilities
# -------------------------
def _trace_distance_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    """
    Trace distance between pure states |psi>,|phi>:
      D = sqrt(1 - |<psi|phi>|^2)
    """
    psi = np.asarray(psi).reshape(-1)
    phi = np.asarray(phi).reshape(-1)
    ov = np.vdot(psi, phi)
    fid = float(np.clip(np.abs(ov) ** 2, 0.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - fid)))


def _trace_distance_mixed(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Trace distance for density matrices:
      D = 0.5 * || rho - sigma ||_1
    """
    A = np.asarray(rho) - np.asarray(sigma)
    # Hermitianize for numerical stability
    A = 0.5 * (A + A.conj().T)
    w = np.linalg.eigvalsh(A)
    return float(0.5 * np.sum(np.abs(w)))


def trace_distance_stats(out_base: np.ndarray, out_prun: np.ndarray) -> Tuple[float, float]:
    """
    out_* could be:
      - (B, dim) state vectors
      - (B, dim, dim) density matrices
      - (dim,) single vector
      - (dim,dim) single density matrix
    """
    A = np.asarray(out_base)
    B = np.asarray(out_prun)

    if A.ndim == 1 and B.ndim == 1:
        d = _trace_distance_pure(A, B)
        return d, d

    if A.ndim == 2 and B.ndim == 2:
        # assume batch of vectors (B, dim)
        ds = [_trace_distance_pure(A[i], B[i]) for i in range(A.shape[0])]
        return float(np.mean(ds)), float(np.max(ds))

    if A.ndim == 3 and B.ndim == 3:
        # assume batch of density matrices (B, dim, dim)
        ds = [_trace_distance_mixed(A[i], B[i]) for i in range(A.shape[0])]
        return float(np.mean(ds)), float(np.max(ds))

    if A.ndim == 2 and B.ndim == 2 and A.shape[0] == A.shape[1]:
        d = _trace_distance_mixed(A, B)
        return d, d

    raise ValueError(f"Unsupported output shapes: base={A.shape}, prun={B.shape}")


# -------------------------
# Single-run wrappers
# -------------------------
def run_one_classification(
    *,
    dataset: str,
    cfg: PaperProtocolConfig,
    data_dir: str,
    download: bool,
    n_train_pool: int,
    n_val_pool: int,
    bas_h: int,
    bas_w: int,
) -> Dict[str, Any]:
    # build splits
    tcfg = ToyVisionConfig(
        n_qubits=cfg.n_qubits,
        data_dir=data_dir,
        seed=cfg.seed,
        download=download,
        n_train_pool=n_train_pool,
        n_val_pool=n_val_pool,
    )
    if dataset == "mnist49":
        tr_states, tr_y, va_states, va_y = build_mnist49_splits(tcfg)
    elif dataset == "fashion_sb":
        tr_states, tr_y, va_states, va_y = build_fashion_sandal_boot_splits(tcfg)
    elif dataset == "bas":
        tr_states, tr_y, va_states, va_y = build_bas_splits(tcfg, H=bas_h, W=bas_w)
    else:
        raise ValueError(dataset)

    # Paper: D from validation (uniform), size M
    task_states, _ = sample_task_ensemble_from_validation(va_states, va_y, M=cfg.M, seed=cfg.seed)

    # q-geometry params (paper)
    lam, q, Gq_diag, Mq, eps_q = make_q_geometry(cfg)

    # candidates + baseline
    all_gates = all_gates_builder(cfg)
    baseline_circuit = build_random_baseline_circuit_no_identity(all_gates, seed=cfg.seed + 999)

    # prefix states for each position (paper D_pos)
    prefix_states = compute_prefix_states(
        realizable_circuit=baseline_circuit,
        task_states=task_states,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
    )

    # TRUE pruning: replace baseline gates by identity if dq <= eps, then compact (remove identity)
    true_res = q_iPrune_true_gate_pruning_by_identity(
        all_gates=all_gates,
        baseline_circuit=baseline_circuit,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
        prefix_states_list=prefix_states,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
    )
    pruned_circuit = true_res.compact_circuit

    # empirical drift on D (paper): compare baseline vs pruned outputs on task_states
    out_task_base = simulate_outputs(baseline_circuit, task_states, cfg.n_qubits, cfg.device)
    out_task_prun = simulate_outputs(pruned_circuit, task_states, cfg.n_qubits, cfg.device)
    td_mean, td_max = trace_distance_stats(out_task_base, out_task_prun)

    # theory RHS uses TRUE L (pruned positions) + eps_q
    L_true = int(true_res.pruned_positions)
    rhs_raw = float((2.0 * L_true / float(Mq)) * np.sin(float(eps_q)))
    rhs_clipped = float(min(1.0, max(0.0, rhs_raw)))

    # task metric: accuracy drop (train readout on baseline; eval val baseline vs pruned)
    train_out_base = simulate_outputs(baseline_circuit, tr_states, cfg.n_qubits, cfg.device)
    val_out_base = simulate_outputs(baseline_circuit, va_states, cfg.n_qubits, cfg.device)
    val_out_prun = simulate_outputs(pruned_circuit, va_states, cfg.n_qubits, cfg.device)

    acc_out = evaluate_classification_accuracy_drop(
        n_qubits=cfg.n_qubits,
        train_out_states_base=train_out_base,
        train_labels=tr_y,
        val_out_states_base=val_out_base,
        val_out_states_prun=val_out_prun,
        val_labels=va_y,
        seed=cfg.seed,
    )

    positions = len(baseline_circuit)
    out: Dict[str, Any] = {
        "device": cfg.device,
        "n_qubits": cfg.n_qubits,
        "depth": cfg.depth,
        "M": cfg.M,
        "seed": cfg.seed,

        "gamma": cfg.gamma,
        "alpha": cfg.alpha,
        "lambda": float(lam),
        "qbeta": cfg.qbeta,
        "q": float(q),
        "Mq": float(Mq),
        "delta": cfg.delta,
        "epsilon_q": float(eps_q),

        "all_candidates": int(len(all_gates)),
        "positions": int(positions),

        # TRUE pruning counts
        "L": int(L_true),
        "replace_ratio_over_candidates": float(L_true / max(1, positions)),  # now means ratio over positions
        "gate_count_before": int(true_res.gate_count_before),
        "gate_count_after": int(true_res.gate_count_after),
        "pruned_positions": int(true_res.pruned_positions),
        "kept_positions": int(true_res.kept_positions),

        # empirical drift on D
        "trace_dist_mean": float(td_mean),
        "trace_dist_max": float(td_max),

        # theory bound
        "rhs_raw": float(rhs_raw),
        "rhs_clipped": float(rhs_clipped),

        # dq stats over ACTUALLY pruned/replaced positions
        "dq_max_global": float(true_res.dq_max_over_replaced),
        "dq_mean_avg": float(true_res.dq_mean_over_replaced),
        "dq_max_over_replaced": float(true_res.dq_max_over_replaced),
        "dq_mean_over_replaced": float(true_res.dq_mean_over_replaced),
        "dq_violations_over_replaced": int(true_res.dq_violations_over_replaced),

        "task": "classification",
        "dataset": dataset,
        "baseline_type": "paper",
    }

    out.update(acc_out)
    _attach_generation_params(out, cfg)

    # vqe fields exist for schema compatibility
    out["energy_base_mean"] = None
    out["energy_prun_mean"] = None
    out["dE_mean"] = None
    out["dE_max"] = None
    return out


def run_one_vqe(
    *,
    cfg: PaperProtocolConfig,
    vqe_depth: int,
    vqe_steps: int,
    vqe_lr: float,
    J: float,
    h: float,
) -> Dict[str, Any]:
    vcfg = VQETwitterConfig(
        n_qubits=cfg.n_qubits,
        depth=vqe_depth,
        steps=vqe_steps,
        lr=vqe_lr,
        seed=cfg.seed,
        J=J,
        h=h,
    )
    task_states = sample_intermediate_states_from_vqe(vcfg, M=cfg.M, device=cfg.device)

    lam, q, Gq_diag, Mq, eps_q = make_q_geometry(cfg)

    all_gates = all_gates_builder(cfg)
    baseline_circuit = build_random_baseline_circuit_no_identity(all_gates, seed=cfg.seed + 999)

    prefix_states = compute_prefix_states(
        realizable_circuit=baseline_circuit,
        task_states=task_states,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
    )

    true_res = q_iPrune_true_gate_pruning_by_identity(
        all_gates=all_gates,
        baseline_circuit=baseline_circuit,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
        prefix_states_list=prefix_states,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
    )
    pruned_circuit = true_res.compact_circuit

    out_task_base = simulate_outputs(baseline_circuit, task_states, cfg.n_qubits, cfg.device)
    out_task_prun = simulate_outputs(pruned_circuit, task_states, cfg.n_qubits, cfg.device)
    td_mean, td_max = trace_distance_stats(out_task_base, out_task_prun)

    L_true = int(true_res.pruned_positions)
    rhs_raw = float((2.0 * L_true / float(Mq)) * np.sin(float(eps_q)))
    rhs_clipped = float(min(1.0, max(0.0, rhs_raw)))

    # energy deviation metric
    eout = evaluate_vqe_energy_deviation(
        n_qubits=cfg.n_qubits,
        states_base=out_task_base,
        states_prun=out_task_prun,
        J=J,
        h=h,
    )

    positions = len(baseline_circuit)
    out: Dict[str, Any] = {
        "device": cfg.device,
        "n_qubits": cfg.n_qubits,
        "depth": cfg.depth,
        "M": cfg.M,
        "seed": cfg.seed,

        "gamma": cfg.gamma,
        "alpha": cfg.alpha,
        "lambda": float(lam),
        "qbeta": cfg.qbeta,
        "q": float(q),
        "Mq": float(Mq),
        "delta": cfg.delta,
        "epsilon_q": float(eps_q),

        "all_candidates": int(len(all_gates)),
        "positions": int(positions),

        "L": int(L_true),
        "replace_ratio_over_candidates": float(L_true / max(1, positions)),
        "gate_count_before": int(true_res.gate_count_before),
        "gate_count_after": int(true_res.gate_count_after),
        "pruned_positions": int(true_res.pruned_positions),
        "kept_positions": int(true_res.kept_positions),

        "trace_dist_mean": float(td_mean),
        "trace_dist_max": float(td_max),

        "rhs_raw": float(rhs_raw),
        "rhs_clipped": float(rhs_clipped),

        "dq_max_global": float(true_res.dq_max_over_replaced),
        "dq_mean_avg": float(true_res.dq_mean_over_replaced),
        "dq_max_over_replaced": float(true_res.dq_max_over_replaced),
        "dq_mean_over_replaced": float(true_res.dq_mean_over_replaced),
        "dq_violations_over_replaced": int(true_res.dq_violations_over_replaced),

        "task": "vqe",
        "dataset": "tfim",
        "baseline_type": "paper",
    }

    out.update(eout)
    _attach_generation_params(out, cfg)

    # classification fields exist for schema compatibility
    out["acc_base"] = None
    out["acc_prun"] = None
    out["acc_drop"] = None
    return out


# -------------------------
# Averaging over seeds
# -------------------------
def average_results_over_seeds(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in results:
        key = (
            r.get("task"),
            r.get("dataset"),
            r.get("baseline_type", "paper"),
            float(r.get("delta")),
            float(r.get("rot_pool_sigma")),
            int(r.get("n_qubits")),
        )
        groups[key].append(r)

    numeric_keys = [
        "L", "replace_ratio_over_candidates",
        "trace_dist_mean", "trace_dist_max",
        "rhs_raw", "rhs_clipped",
        "dq_mean_avg", "dq_max_global",
        "dq_max_over_replaced", "dq_mean_over_replaced", "dq_violations_over_replaced",
        "acc_base", "acc_prun", "acc_drop",
        "energy_base_mean", "energy_prun_mean", "dE_mean", "dE_max",
        "gate_count_before", "gate_count_after", "pruned_positions", "kept_positions",
        "all_candidates", "positions",
    ]

    int_fields = {
        "L", "dq_violations_over_replaced",
        "gate_count_before", "gate_count_after", "pruned_positions", "kept_positions",
        "all_candidates", "positions",
    }

    averaged: List[Dict[str, Any]] = []
    for _, group in groups.items():
        base = dict(group[0])
        for k in numeric_keys:
            vals = []
            for r in group:
                v = r.get(k, None)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            base[k] = (sum(vals) / len(vals)) if vals else None

        for k in int_fields:
            if base.get(k) is not None:
                base[k] = int(round(float(base[k])))

        base["num_seeds_averaged"] = len(group)
        base["seeds_used"] = [r.get("seed") for r in group]

        averaged.append(base)

    return averaged


# -------------------------
# Main sweep
# -------------------------
def main():
    deltas = [0.01, 0.02]
    sigmas = [0.001, 0.003, 0.006, 0.01]
    seeds = [42, 43]

    device = "default.qubit"
    depth = 12
    M = 50

    gamma = 0.05
    alpha = 0.6
    qbeta = 0.0

    rot_pool_k = 4
    include_cnot = True

    data_dir = "./data"
    download = True
    n_train_pool = 4000
    n_val_pool = 2000
    bas_h, bas_w = 4, 4

    vqe_depth = 6
    vqe_steps = 40
    vqe_lr = 0.1
    J, h = 1.0, 1.0

    results: List[Dict[str, Any]] = []
    meta = {
        "grid": {"deltas": deltas, "sigmas": sigmas, "seeds": seeds},
        "defaults": {
            "device": device, "depth": depth, "M": M,
            "gamma": gamma, "alpha": alpha, "qbeta": qbeta,
            "rot_pool_k": rot_pool_k, "include_cnot": include_cnot,
            "n_train_pool": n_train_pool, "n_val_pool": n_val_pool,
            "vqe_depth": vqe_depth, "vqe_steps": vqe_steps, "vqe_lr": vqe_lr, "J": J, "h": h,
        }
    }

    t0 = time.time()

    for seed in seeds:
        for delta in deltas:
            for sigma in sigmas:
                # MNIST49, Fashion use 8 qubits
                for ds in ["mnist49", "fashion_sb"]:
                    cfg = PaperProtocolConfig(
                        device=device,
                        n_qubits=8,
                        depth=depth,
                        M=M,
                        seed=seed,
                        gamma=gamma,
                        alpha=alpha,
                        qbeta=qbeta,
                        delta=delta,
                        rot_pool_k=rot_pool_k,
                        rot_pool_sigma=sigma,
                        include_cnot=include_cnot,
                    )
                    out = run_one_classification(
                        dataset=ds,
                        cfg=cfg,
                        data_dir=data_dir,
                        download=download,
                        n_train_pool=n_train_pool,
                        n_val_pool=n_val_pool,
                        bas_h=bas_h,
                        bas_w=bas_w,
                    )
                    results.append(out)
                    print(f"[OK] cls {ds} seed={seed} delta={delta} sigma={sigma} nQ=8 "
                          f"gates {out['gate_count_before']}->{out['gate_count_after']} "
                          f"L={out['L']} drift_max={out['trace_dist_max']:.6f} drop={out['acc_drop']:.4f}")

                # BAS and VQE use 4 qubits
                cfg4 = PaperProtocolConfig(
                    device=device,
                    n_qubits=4,
                    depth=depth,
                    M=M,
                    seed=seed,
                    gamma=gamma,
                    alpha=alpha,
                    qbeta=qbeta,
                    delta=delta,
                    rot_pool_k=rot_pool_k,
                    rot_pool_sigma=sigma,
                    include_cnot=include_cnot,
                )

                out = run_one_classification(
                    dataset="bas",
                    cfg=cfg4,
                    data_dir=data_dir,
                    download=download,
                    n_train_pool=n_train_pool,
                    n_val_pool=n_val_pool,
                    bas_h=bas_h,
                    bas_w=bas_w,
                )
                results.append(out)
                print(f"[OK] cls bas seed={seed} delta={delta} sigma={sigma} nQ=4 "
                      f"gates {out['gate_count_before']}->{out['gate_count_after']} "
                      f"L={out['L']} drift_max={out['trace_dist_max']:.6f} drop={out['acc_drop']:.4f}")

                out = run_one_vqe(
                    cfg=cfg4,
                    vqe_depth=vqe_depth,
                    vqe_steps=vqe_steps,
                    vqe_lr=vqe_lr,
                    J=J,
                    h=h,
                )
                results.append(out)
                print(f"[OK] vqe tfim seed={seed} delta={delta} sigma={sigma} nQ=4 "
                      f"gates {out['gate_count_before']}->{out['gate_count_after']} "
                      f"L={out['L']} drift_max={out['trace_dist_max']:.6f} dE_max={out['dE_max']:.6f}")

    print("\n" + "=" * 70)
    print("Averaging results over seeds...")
    print("=" * 70)
    averaged_results = average_results_over_seeds(results)

    payload = {"meta": meta, "results": averaged_results, "elapsed_sec": time.time() - t0}

    os.makedirs("./results", exist_ok=True)
    out_path = "./results/qiprune_sweep_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    print(f"Original results: {len(results)}  Averaged: {len(averaged_results)}")
    print(f"Elapsed: {time.time() - t0:.2f} sec")


if __name__ == "__main__":
    main()
