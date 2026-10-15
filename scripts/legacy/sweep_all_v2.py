from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from iprune.gates import GateOp, build_hea_gates, build_hea_gates_from_parameters, is_identity_rot

from iprune.experiments.protocol import (
    PaperProtocolConfig,
    simulate_outputs,
    make_q_geometry,
    trace_distance_stats,
)

from iprune.experiments.tasks_classification import (
    ToyVisionConfig,
    build_mnist49_splits,
    build_fashion_sandal_boot_splits,
    build_bas_splits,
    sample_task_ensemble_from_validation,
)
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters
from iprune.experiments.task_metrics import (
    evaluate_classification_accuracy_drop,
    evaluate_vqe_energy_deviation,
)

from iprune.pruning_true import TruePruneResult, q_iPrune_true_gate_pruning_by_identity
from iprune.result_validation import validate_result_record

def all_gates_builder(cfg: PaperProtocolConfig):
    return build_hea_gates(
        cfg.n_qubits, cfg.depth, seed=cfg.seed,
        rot_pool_k=cfg.rot_pool_k,
        rot_pool_sigma=cfg.rot_pool_sigma,
        include_cnot=cfg.include_cnot,
    )


def build_random_baseline_circuit_no_identity(all_gates: List[GateOp], seed: int) -> List[GateOp]:
    """Select one non-identity candidate per position for classification."""
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

    for index, gate in enumerate(baseline):
        if gate.pos_id != index:
            raise RuntimeError(
                "Baseline positions must be consecutive; "
                f"found pos_id={gate.pos_id} at index={index}."
            )
    return baseline



def _attach_generation_params(out: Dict[str, Any], cfg: PaperProtocolConfig) -> None:
    # Make every result row self-contained (no need to infer from meta/order)
    out["rot_pool_k"] = int(cfg.rot_pool_k)
    out["rot_pool_sigma"] = float(cfg.rot_pool_sigma)
    out["include_cnot"] = bool(cfg.include_cnot)


def _circuit_signature(circuit: List[GateOp]) -> str:
    """Stable short signature for checking that all metrics use the same circuit."""
    pieces = []
    for gate in circuit:
        params = "" if gate.params is None else ",".join(f"{value:.17g}" for value in gate.params)
        pieces.append(
            f"{gate.pos_id}|{gate.gate_type}|{gate.wires}|{params}"
        )
    payload = "\n".join(pieces).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _logical_resource_metrics(circuit: List[GateOp], n_qubits: int) -> Dict[str, int]:
    """Gate counts and a backend-independent greedy logical depth."""
    wire_depths = [0] * n_qubits
    one_qubit = 0
    two_qubit = 0
    for gate in circuit:
        if len(gate.wires) == 1:
            one_qubit += 1
        elif len(gate.wires) == 2:
            two_qubit += 1
        start = 1 + max(wire_depths[wire] for wire in gate.wires)
        for wire in gate.wires:
            wire_depths[wire] = start
    return {
        "total": len(circuit),
        "one_qubit": one_qubit,
        "two_qubit": two_qubit,
        "logical_depth": max(wire_depths, default=0),
    }


def _build_true_pruning_record(
    *,
    cfg: PaperProtocolConfig,
    all_gates: List[GateOp],
    baseline_circuit: List[GateOp],
    prune_result: TruePruneResult,
    lambda_value: float,
    q_value: float,
    Mq: float,
    epsilon_q: float,
    task_states_base: np.ndarray,
    task_states_pruned: np.ndarray,
    task: str,
    dataset: str,
    model_training: str,
) -> Dict[str, Any]:
    """Create one self-consistent record from one baseline/pruned circuit pair."""
    trace_mean, trace_max = trace_distance_stats(task_states_base, task_states_pruned)
    actual_L = int(prune_result.pruned_positions)
    positions = len(baseline_circuit)
    prune_ratio = float(actual_L / max(1, positions))
    # Every accepted deletion also passes the physical FS-angle threshold.
    # Therefore D <= L*sin(epsilon_q) and ||rho-sigma||_1 <= 2L*sin(epsilon_q).
    trace_norm_rhs_raw = float(2.0 * actual_L * np.sin(epsilon_q))
    trace_distance_rhs_raw = 0.5 * trace_norm_rhs_raw
    resources_before = _logical_resource_metrics(baseline_circuit, cfg.n_qubits)
    resources_after = _logical_resource_metrics(
        prune_result.compact_circuit, cfg.n_qubits
    )

    record: Dict[str, Any] = {
        "schema_version": 2,
        "task": task,
        "dataset": dataset,
        "baseline_type": "actual_circuit",
        "model_training": model_training,
        "pruning_semantics": "identity_gate_deletion",
        "candidate_pool_applies": task == "classification",
        "L_definition": "actual_deleted_forward_positions",
        "ratio_denominator": "realizable_circuit_positions",
        "device": cfg.device,
        "n_qubits": cfg.n_qubits,
        "depth": cfg.depth,
        "M": int(np.asarray(task_states_base).shape[0]),
        "seed": cfg.seed,
        "gamma": cfg.gamma,
        "alpha": cfg.alpha,
        "lambda": float(lambda_value),
        "qbeta": cfg.qbeta,
        "q": float(q_value),
        "Mq": float(Mq),
        "delta": cfg.delta,
        "epsilon_q": float(epsilon_q),
        "all_candidates": int(len(all_gates)),
        "positions": int(positions),
        "candidate_replacements": None,
        "L": actual_L,
        "actual_gate_prune_ratio": prune_ratio,
        # Legacy plotting key; its denominator is explicitly recorded above.
        "replace_ratio_over_candidates": prune_ratio,
        "gate_count_before": int(prune_result.gate_count_before),
        "gate_count_after": int(prune_result.gate_count_after),
        "one_qubit_gate_count_before": resources_before["one_qubit"],
        "one_qubit_gate_count_after": resources_after["one_qubit"],
        "two_qubit_gate_count_before": resources_before["two_qubit"],
        "two_qubit_gate_count_after": resources_after["two_qubit"],
        "logical_depth_before": resources_before["logical_depth"],
        "logical_depth_after": resources_after["logical_depth"],
        "transpiled_depth_before": None,
        "transpiled_depth_after": None,
        "transpiled_depth_status": "backend_not_configured",
        "pruned_positions": actual_L,
        "kept_positions": int(prune_result.kept_positions),
        "baseline_circuit_signature": _circuit_signature(baseline_circuit),
        "selected_circuit_signature": _circuit_signature(prune_result.compact_circuit),
        "trace_dist_mean": float(trace_mean),
        "trace_dist_max": float(trace_max),
        "rhs_raw": trace_distance_rhs_raw,
        "rhs_clipped": float(min(1.0, max(0.0, trace_distance_rhs_raw))),
        "rhs_quantity": "trace_distance_upper_bound",
        "trace_norm_rhs_raw": trace_norm_rhs_raw,
        "trace_norm_rhs_clipped": float(min(2.0, max(0.0, trace_norm_rhs_raw))),
        "dq_mean_avg": float(prune_result.dq_mean_over_replaced),
        "dq_max_global": float(prune_result.dq_max_over_replaced),
        "dq_max_over_replaced": float(prune_result.dq_max_over_replaced),
        "dq_mean_over_replaced": float(prune_result.dq_mean_over_replaced),
        "dq_violations_over_replaced": int(prune_result.dq_violations_over_replaced),
        "physical_angle_max_over_replaced": float(
            prune_result.physical_angle_max_over_replaced
        ),
        "physical_trace_dist_max_over_replaced": float(
            prune_result.physical_trace_dist_max_over_replaced
        ),
        "physical_violations_over_replaced": int(
            prune_result.physical_violations_over_replaced
        ),
    }
    _attach_generation_params(record, cfg)
    return record


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
    """Prune one fixed quantum feature map and evaluate one trained readout."""
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

    task_states, _ = sample_task_ensemble_from_validation(
        va_states, va_y, M=cfg.M, seed=cfg.seed
    )
    lam, q, Gq_diag, Mq, eps_q = make_q_geometry(cfg)
    all_gates = all_gates_builder(cfg)
    baseline_circuit = build_random_baseline_circuit_no_identity(all_gates, seed=cfg.seed + 999)
    true_res = q_iPrune_true_gate_pruning_by_identity(
        all_gates=all_gates,
        baseline_circuit=baseline_circuit,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
        task_states=task_states,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
    )
    pruned_circuit = true_res.compact_circuit

    task_out_base = simulate_outputs(baseline_circuit, task_states, cfg.n_qubits, cfg.device)
    task_out_pruned = simulate_outputs(pruned_circuit, task_states, cfg.n_qubits, cfg.device)
    train_out_base = simulate_outputs(baseline_circuit, tr_states, cfg.n_qubits, cfg.device)
    val_out_base = simulate_outputs(baseline_circuit, va_states, cfg.n_qubits, cfg.device)
    val_out_pruned = simulate_outputs(pruned_circuit, va_states, cfg.n_qubits, cfg.device)

    accuracy_metrics = evaluate_classification_accuracy_drop(
        n_qubits=cfg.n_qubits,
        train_out_states_base=train_out_base,
        train_labels=tr_y,
        val_out_states_base=val_out_base,
        val_out_states_prun=val_out_pruned,
        val_labels=va_y,
        seed=cfg.seed,
    )

    record = _build_true_pruning_record(
        cfg=cfg,
        all_gates=all_gates,
        baseline_circuit=baseline_circuit,
        prune_result=true_res,
        lambda_value=lam,
        q_value=q,
        Mq=Mq,
        epsilon_q=eps_q,
        task_states_base=task_out_base,
        task_states_pruned=task_out_pruned,
        task="classification",
        dataset=dataset,
        model_training="fixed_quantum_feature_map_with_trained_logistic_readout",
    )
    record.update(accuracy_metrics)
    record.update(
        {
            "energy_base_mean": None,
            "energy_prun_mean": None,
            "dE_mean": None,
            "dE_max": None,
        }
    )
    validate_result_record(record)
    return record


def run_one_vqe(
        *,
        cfg: PaperProtocolConfig,
        vqe_depth: int,
        vqe_steps: int,
        vqe_lr: float,
        J: float,
        h: float,
) -> Dict[str, Any]:
    """Optimize one TFIM ansatz, then prune that exact optimized circuit."""
    vcfg = VQEConfig(
        n_qubits=cfg.n_qubits,
        depth=vqe_depth,
        steps=vqe_steps,
        lr=vqe_lr,
        seed=cfg.seed,
        J=J,
        h=h,
    )
    optimized_params, energy_history = optimize_vqe_parameters(vcfg, device=cfg.device)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        optimized_params,
        include_cnot=cfg.include_cnot,
        include_identity_rot=True,
    )
    task_states = np.zeros((1, 2**cfg.n_qubits), dtype=np.complex128)
    task_states[0, 0] = 1.0
    lam, q, Gq_diag, Mq, eps_q = make_q_geometry(cfg)
    true_res = q_iPrune_true_gate_pruning_by_identity(
        all_gates=all_gates,
        baseline_circuit=baseline_circuit,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
        task_states=task_states,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
    )
    pruned_circuit = true_res.compact_circuit
    out_base = simulate_outputs(baseline_circuit, task_states, cfg.n_qubits, cfg.device)
    out_pruned = simulate_outputs(pruned_circuit, task_states, cfg.n_qubits, cfg.device)
    energy_metrics = evaluate_vqe_energy_deviation(
        n_qubits=cfg.n_qubits,
        states_base=out_base,
        states_prun=out_pruned,
        J=J,
        h=h,
    )

    record = _build_true_pruning_record(
        cfg=cfg,
        all_gates=all_gates,
        baseline_circuit=baseline_circuit,
        prune_result=true_res,
        lambda_value=lam,
        q_value=q,
        Mq=Mq,
        epsilon_q=eps_q,
        task_states_base=out_base,
        task_states_pruned=out_pruned,
        task="vqe",
        dataset="tfim",
        model_training="optimized_tfim_vqe_ansatz",
    )
    record.update(energy_metrics)
    record.update(
        {
            "vqe_initial_energy": float(energy_history[0]),
            "vqe_optimized_energy": float(energy_history[-1]),
            "vqe_optimization_steps": int(vqe_steps),
            "acc_base": None,
            "acc_prun": None,
            "acc_drop": None,
        }
    )
    validate_result_record(record)
    return record


def average_results_over_seeds(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Average results over seeds for each unique (task, dataset, delta, sigma, n_qubits) combination.
    """
    from collections import defaultdict

    # Group by (task, dataset, delta, sigma, n_qubits)
    groups = defaultdict(list)
    for r in results:
        key = (
            r.get("task"),
            r.get("dataset"),
            r.get("delta"),
            r.get("rot_pool_sigma"),
            r.get("n_qubits"),
        )
        groups[key].append(r)

    averaged = []

    # Metrics to average (numeric fields)
    numeric_keys = [
        "L", "actual_gate_prune_ratio", "replace_ratio_over_candidates",
        "gate_count_before", "gate_count_after", "pruned_positions", "kept_positions",
        "all_candidates", "positions",
        "trace_dist_mean", "trace_dist_max",
        "rhs_raw", "rhs_clipped", "trace_norm_rhs_raw", "trace_norm_rhs_clipped",
        "dq_mean_avg", "dq_max_avg", "dq_max_global",
        "dq_max_over_replaced", "dq_mean_over_replaced", "dq_violations_over_replaced",
        "physical_angle_max_over_replaced", "physical_trace_dist_max_over_replaced",
        "physical_violations_over_replaced",
        "acc_base", "acc_prun", "acc_drop",  # classification
        "energy_base_mean", "energy_prun_mean", "dE_mean", "dE_max",  # vqe
        "vqe_initial_energy", "vqe_optimized_energy",
    ]

    for key, group in groups.items():
        if not group:
            continue

        # Start with first result as template
        avg_result = dict(group[0])

        # Average numeric fields
        for field in numeric_keys:
            values = []
            for r in group:
                val = r.get(field)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass

            if values:
                avg_result[field] = sum(values) / len(values)
                avg_result[f"{field}_std"] = float(np.std(values))
            else:
                avg_result[field] = None
                avg_result[f"{field}_std"] = None

        # Add metadata about averaging
        avg_result["num_seeds_averaged"] = len(group)
        avg_result["seeds_used"] = [r.get("seed") for r in group]
        avg_result["aggregation"] = "mean_over_seed_specific_circuit_pairs"
        avg_result["per_seed_circuit_signatures"] = [
            {
                "seed": r.get("seed"),
                "baseline": r.get("baseline_circuit_signature"),
                "selected": r.get("selected_circuit_signature"),
            }
            for r in group
        ]
        avg_result["baseline_circuit_signature"] = None
        avg_result["selected_circuit_signature"] = None

        averaged.append(avg_result)

    return averaged


def main():
    # ---- sweep grid ----
    deltas = [0.01, 0.02]
    sigmas = [0.001, 0.003, 0.006, 0.01]
    seeds = [42, 43]  # 2 seeds for averaging

    # ---- fixed protocol defaults (paper-aligned) ----
    device = "default.qubit"
    depth = 12
    M = 50

    gamma = 0.05
    alpha = 0.6
    qbeta = 0.0

    rot_pool_k = 4
    include_cnot = True

    # classification pools
    data_dir = str(Path(__file__).resolve().parents[2] / "data")
    download = True
    n_train_pool = 4000
    n_val_pool = 2000
    bas_h, bas_w = 4, 4

    # VQE config
    vqe_depth = 6
    vqe_steps = 40
    vqe_lr = 0.1
    J, h = 1.0, 1.0

    results: List[Dict[str, Any]] = []
    meta = {
        "schema_version": 2,
        "grid": {
            "classification": {"deltas": deltas, "sigmas": sigmas, "seeds": seeds},
            "vqe": {"deltas": deltas, "sigmas": "not_applicable", "seeds": seeds},
        },
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
                for dataset in ["mnist49", "fashion_sb"]:
                    cfg8 = PaperProtocolConfig(
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
                        dataset=dataset,
                        cfg=cfg8,
                        data_dir=data_dir,
                        download=download,
                        n_train_pool=n_train_pool,
                        n_val_pool=n_val_pool,
                        bas_h=bas_h,
                        bas_w=bas_w,
                    )
                    results.append(out)
                    print(
                        f"[OK] cls {dataset} seed={seed} delta={delta} sigma={sigma} nQ=8 "
                        f"gates={out['gate_count_before']}->{out['gate_count_after']} "
                        f"drift_max={out['trace_dist_max']:.6f} drop={out['acc_drop']:.4f}"
                    )

                cfg4_classification = PaperProtocolConfig(
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
                    cfg=cfg4_classification,
                    data_dir=data_dir,
                    download=download,
                    n_train_pool=n_train_pool,
                    n_val_pool=n_val_pool,
                    bas_h=bas_h,
                    bas_w=bas_w,
                )
                results.append(out)
                print(
                    f"[OK] cls bas seed={seed} delta={delta} sigma={sigma} nQ=4 "
                    f"gates={out['gate_count_before']}->{out['gate_count_after']} "
                    f"drift_max={out['trace_dist_max']:.6f} drop={out['acc_drop']:.4f}"
                )

            # VQE has no sampled candidate-pool sigma.  Run it once per seed/delta.
            cfg4_vqe = PaperProtocolConfig(
                device=device,
                n_qubits=4,
                depth=vqe_depth,
                M=1,
                seed=seed,
                gamma=gamma,
                alpha=alpha,
                qbeta=qbeta,
                delta=delta,
                rot_pool_k=1,
                rot_pool_sigma=0.0,
                include_cnot=include_cnot,
            )
            out = run_one_vqe(
                cfg=cfg4_vqe,
                vqe_depth=vqe_depth,
                vqe_steps=vqe_steps,
                vqe_lr=vqe_lr,
                J=J,
                h=h,
            )
            results.append(out)
            print(
                f"[OK] vqe tfim seed={seed} delta={delta} nQ=4 "
                f"gates={out['gate_count_before']}->{out['gate_count_after']} "
                f"drift_max={out['trace_dist_max']:.6f} dE_max={out['dE_max']:.6f}"
            )

    # Average over seeds
    print("\n" + "=" * 70)
    print("Averaging results over seeds...")
    print("=" * 70)
    averaged_results = average_results_over_seeds(results)

    # Print averaged results summary
    print(f"\nOriginal results: {len(results)} (from {len(seeds)} seeds)")
    print(f"Averaged results: {len(averaged_results)} (unique configurations)")

    payload = {
        "meta": meta,
        "runs": results,
        "results": averaged_results,
        "elapsed_sec": time.time() - t0,
    }

    results_dir = Path(__file__).resolve().parents[2] / "results"
    os.makedirs(results_dir, exist_ok=True)
    out_path = results_dir / "qiprune_sweep_v2_fixed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    print(f"Total configurations: {len(averaged_results)}")
    print(f"Each averaged over {len(seeds)} seed(s)")
    print(f"Time elapsed: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    main()
