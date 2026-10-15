from __future__ import annotations

import os, json, time
from typing import Any, Dict, List, Tuple

import numpy as np

from iprune.experiments.random_baseline import build_random_pruned_circuit_matched_L
from iprune.experiments.protocol import (
    PaperProtocolConfig,
    run_positionaware_paper_protocol,
    simulate_outputs,
    build_random_baseline_circuit,
    make_q_geometry,
)

from iprune.gates import build_hea_gates, GateOp
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
from iprune.pruning_positionaware import q_iPrune_positionaware_candidate_pools


# -----------------------------
# Trace distance utils (pure states)
# -----------------------------
def _trace_distance_pure_states(psi: np.ndarray, phi: np.ndarray) -> float:
    # D(|psi><psi|,|phi><phi|) = sqrt(1 - |<psi|phi>|^2)
    ov = np.vdot(psi, phi)
    val = 1.0 - (np.abs(ov) ** 2)
    return float(np.sqrt(max(0.0, min(1.0, val))))


def trace_distance_stats(states_a: np.ndarray, states_b: np.ndarray) -> Tuple[float, float]:
    ds = [_trace_distance_pure_states(states_a[i], states_b[i]) for i in range(states_a.shape[0])]
    return float(np.mean(ds)), float(np.max(ds))


def all_gates_builder(cfg: PaperProtocolConfig):
    return build_hea_gates(
        cfg.n_qubits, cfg.depth, seed=cfg.seed,
        rot_pool_k=cfg.rot_pool_k,
        rot_pool_sigma=cfg.rot_pool_sigma,
        include_cnot=cfg.include_cnot,
    )


def _attach_generation_params(out: Dict[str, Any], cfg: PaperProtocolConfig) -> None:
    # Make every result row self-contained (no need to infer from meta/order)
    out["rot_pool_k"] = int(cfg.rot_pool_k)
    out["rot_pool_sigma"] = float(cfg.rot_pool_sigma)
    out["include_cnot"] = bool(cfg.include_cnot)


def _extract_dq_stats(prune_res: Any) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    for k in ["dq_max_over_replaced", "dq_mean_over_replaced", "dq_violations_over_replaced"]:
        if hasattr(prune_res, k):
            extra[k] = getattr(prune_res, k)
        elif isinstance(prune_res, dict) and k in prune_res:
            extra[k] = prune_res[k]
    return extra


def debug_print_one_rot_pool(all_gates: List[GateOp], pos_id: int, wires: Tuple[int, ...] = (0,)) -> None:
    # pool key 和你 pruning 里一致：(pos_id, gate_type, wires)
    pool = [g for g in all_gates
            if g.gate_type == "Rot" and g.pos_id == pos_id and tuple(sorted(g.wires)) == tuple(sorted(wires))]

    if not pool:
        print(f"[SANITY] No Rot pool found for pos_id={pos_id}, wires={wires}")
        return

    params = np.array([g.params for g in pool], dtype=float)  # shape: (k, n_params)
    mu = params.mean(axis=0)
    std = params.std(axis=0)

    print(f"[SANITY] Rot pool pos_id={pos_id}, wires={wires}, k={len(pool)}")
    print(f"         params mean = {mu}")
    print(f"         params std  = {std}")


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
) -> List[Dict[str, Any]]:
    """
    Return TWO records:
      - baseline_type="paper"            (dq-thresholded replacement)
      - baseline_type="random_matchedL"  (random replacement with the same L as paper)
    """
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
    task_states, _task_labels = sample_task_ensemble_from_validation(va_states, va_y, M=cfg.M, seed=cfg.seed)

    # Run protocol (paper-aligned drift/bound metrics) => scalars
    out_proto = run_positionaware_paper_protocol(cfg=cfg, all_gates_builder=all_gates_builder, task_states=task_states)

    # Reconstruct baseline circuit & pruned circuit (paper-aligned) to evaluate task metric + get dq stats
    lam, q, Gq_diag, Mq, eps_q = make_q_geometry(cfg)
    all_gates = all_gates_builder(cfg)
    baseline_circuit = build_random_baseline_circuit(all_gates, seed=cfg.seed + 999)

    prefix_states = compute_prefix_states(
        realizable_circuit=baseline_circuit,
        task_states=task_states,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
    )
    prune_res = q_iPrune_positionaware_candidate_pools(
        all_gates,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
        prefix_states_list=prefix_states,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
        baseline_circuit=baseline_circuit,
    )
    pruned_circuit_paper = prune_res.selected_circuit

    # task metric: train readout on baseline outputs, eval on val outputs baseline vs pruned
    train_out_base = simulate_outputs(baseline_circuit, tr_states, cfg.n_qubits, cfg.device)
    val_out_base = simulate_outputs(baseline_circuit, va_states, cfg.n_qubits, cfg.device)
    val_out_paper = simulate_outputs(pruned_circuit_paper, va_states, cfg.n_qubits, cfg.device)

    acc_paper = evaluate_classification_accuracy_drop(
        n_qubits=cfg.n_qubits,
        train_out_states_base=train_out_base,
        train_labels=tr_y,
        val_out_states_base=val_out_base,
        val_out_states_prun=val_out_paper,
        val_labels=va_y,
        seed=cfg.seed,
    )

    # paper record
    out_paper = dict(out_proto)
    out_paper.update(acc_paper)
    out_paper.update(_extract_dq_stats(prune_res))
    out_paper["task"] = "classification"
    out_paper["dataset"] = dataset
    out_paper["baseline_type"] = "paper"
    _attach_generation_params(out_paper, cfg)

    # -----------------------------
    # RANDOM baseline: matched-L
    # -----------------------------
    # match the SAME replacement budget (L) as paper
    L_target = int(getattr(prune_res, "replaced_count", out_paper.get("L", 0)))

    rb = build_random_pruned_circuit_matched_L(
        all_gates=all_gates,
        selected_circuit_paper=pruned_circuit_paper,
        L_target=L_target,
        seed=cfg.seed + 2025,
    )
    pruned_circuit_rand = rb.selected_circuit

    # drift computed on the SAME paper task ensemble D
    base_out_D = simulate_outputs(baseline_circuit, task_states, cfg.n_qubits, cfg.device)
    rand_out_D = simulate_outputs(pruned_circuit_rand, task_states, cfg.n_qubits, cfg.device)
    td_mean, td_max = trace_distance_stats(base_out_D, rand_out_D)

    # task metric on full validation pool
    val_out_rand = simulate_outputs(pruned_circuit_rand, va_states, cfg.n_qubits, cfg.device)
    acc_rand = evaluate_classification_accuracy_drop(
        n_qubits=cfg.n_qubits,
        train_out_states_base=train_out_base,
        train_labels=tr_y,
        val_out_states_base=val_out_base,
        val_out_states_prun=val_out_rand,
        val_labels=va_y,
        seed=cfg.seed,
    )

    out_rand = dict(out_proto)
    out_rand.update(acc_rand)
    out_rand["task"] = "classification"
    out_rand["dataset"] = dataset
    out_rand["baseline_type"] = "random_matchedL"
    _attach_generation_params(out_rand, cfg)

    # overwrite drift with random drift (paper protocol drift is for paper-aligned pruning)
    out_rand["trace_dist_mean"] = float(td_mean)
    out_rand["trace_dist_max"] = float(td_max)
    # overwrite L/ratio to matched-L numbers
    out_rand["L"] = int(rb.L)
    out_rand["replace_ratio_over_candidates"] = float(rb.replace_ratio_over_candidates)
    # random baseline does NOT satisfy dq constraint; leave these as None
    out_rand["dq_max_over_replaced"] = None
    out_rand["dq_mean_over_replaced"] = None
    out_rand["dq_violations_over_replaced"] = None

    return [out_paper, out_rand]


def run_one_vqe(
        *,
        cfg: PaperProtocolConfig,
        vqe_depth: int,
        vqe_steps: int,
        vqe_lr: float,
        J: float,
        h: float,
) -> List[Dict[str, Any]]:
    """
    Return TWO records:
      - baseline_type="paper"
      - baseline_type="random_matchedL"
    """
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

    out_proto = run_positionaware_paper_protocol(cfg=cfg, all_gates_builder=all_gates_builder, task_states=task_states)

    lam, q, Gq_diag, Mq, eps_q = make_q_geometry(cfg)
    all_gates = all_gates_builder(cfg)
    baseline_circuit = build_random_baseline_circuit(all_gates, seed=cfg.seed + 999)

    prefix_states = compute_prefix_states(
        realizable_circuit=baseline_circuit,
        task_states=task_states,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
    )
    prune_res = q_iPrune_positionaware_candidate_pools(
        all_gates,
        n_qubits=cfg.n_qubits,
        device_name=cfg.device,
        prefix_states_list=prefix_states,
        Gq_diag=Gq_diag,
        epsilon_q=eps_q,
        baseline_circuit=baseline_circuit,
    )
    pruned_circuit_paper = prune_res.selected_circuit

    out_base = simulate_outputs(baseline_circuit, task_states, cfg.n_qubits, cfg.device)
    out_paper_states = simulate_outputs(pruned_circuit_paper, task_states, cfg.n_qubits, cfg.device)

    e_paper = evaluate_vqe_energy_deviation(
        n_qubits=cfg.n_qubits,
        states_base=out_base,
        states_prun=out_paper_states,
        J=J,
        h=h,
    )

    out_paper = dict(out_proto)
    out_paper.update(e_paper)
    out_paper.update(_extract_dq_stats(prune_res))
    out_paper["task"] = "vqe"
    out_paper["dataset"] = "tfim"
    out_paper["baseline_type"] = "paper"
    _attach_generation_params(out_paper, cfg)

    # RANDOM matched-L
    L_target = int(getattr(prune_res, "replaced_count", out_paper.get("L", 0)))
    rb = build_random_pruned_circuit_matched_L(
        all_gates=all_gates,
        selected_circuit_paper=pruned_circuit_paper,
        L_target=L_target,
        seed=cfg.seed + 2025,
    )
    pruned_circuit_rand = rb.selected_circuit

    out_rand_states = simulate_outputs(pruned_circuit_rand, task_states, cfg.n_qubits, cfg.device)
    e_rand = evaluate_vqe_energy_deviation(
        n_qubits=cfg.n_qubits,
        states_base=out_base,
        states_prun=out_rand_states,
        J=J,
        h=h,
    )

    # drift on the SAME task ensemble D
    td_mean, td_max = trace_distance_stats(out_base, out_rand_states)

    out_rand = dict(out_proto)
    out_rand.update(e_rand)
    out_rand["task"] = "vqe"
    out_rand["dataset"] = "tfim"
    out_rand["baseline_type"] = "random_matchedL"
    _attach_generation_params(out_rand, cfg)

    out_rand["trace_dist_mean"] = float(td_mean)
    out_rand["trace_dist_max"] = float(td_max)
    out_rand["L"] = int(rb.L)
    out_rand["replace_ratio_over_candidates"] = float(rb.replace_ratio_over_candidates)
    out_rand["dq_max_over_replaced"] = None
    out_rand["dq_mean_over_replaced"] = None
    out_rand["dq_violations_over_replaced"] = None

    return [out_paper, out_rand]


def average_results_over_seeds(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Average results over seeds for each unique (task, dataset, delta, sigma, n_qubits, baseline_type) combination.
    """
    from collections import defaultdict

    # Group by (task, dataset, delta, sigma, n_qubits, baseline_type)
    groups = defaultdict(list)
    for r in results:
        key = (
            r.get("task"),
            r.get("dataset"),
            r.get("delta"),
            r.get("rot_pool_sigma"),
            r.get("n_qubits"),
            r.get("baseline_type"),
        )
        groups[key].append(r)

    averaged = []

    # Metrics to average (numeric fields)
    numeric_keys = [
        "L", "replace_ratio_over_candidates",
        "trace_dist_mean", "trace_dist_max",
        "rhs_raw", "rhs_clipped",
        "dq_mean_avg", "dq_max_avg", "dq_max_global",
        "dq_max_over_replaced", "dq_mean_over_replaced", "dq_violations_over_replaced",
        "acc_base", "acc_prun", "acc_drop",  # classification
        "energy_base_mean", "energy_prun_mean", "dE_mean", "dE_max",  # vqe
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
            else:
                avg_result[field] = None

        # For integer fields, round to int
        if avg_result.get("L") is not None:
            avg_result["L"] = int(round(avg_result["L"]))
        if avg_result.get("dq_violations_over_replaced") is not None:
            avg_result["dq_violations_over_replaced"] = int(round(avg_result["dq_violations_over_replaced"]))

        # Add metadata about averaging
        avg_result["num_seeds_averaged"] = len(group)
        avg_result["seeds_used"] = [r.get("seed") for r in group]

        averaged.append(avg_result)

    return averaged


def main():
    # ---- sweep grid ----
    deltas = [0.01, 0.02]
    sigmas = [0.001, 0.006, 0.01]
    seeds = [42, 43, 44]  # 2 seeds for averaging

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
    data_dir = "./data"
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
    printed_sanity = False

    for seed in seeds:
        for delta in deltas:
            for sigma in sigmas:
                # --- mnist49 and fashion_sb use n_qubits=8 ---
                for ds in ["mnist49", "fashion_sb"]:
                    cfg = PaperProtocolConfig(
                        device=device,
                        n_qubits=8,  # 8 qubits for these datasets
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

                    if (not printed_sanity) and seed == 42 and abs(delta - 0.01) < 1e-12 and ds == "mnist49":
                        all_g = all_gates_builder(cfg)
                        debug_print_one_rot_pool(all_g, pos_id=0, wires=(0,))
                        printed_sanity = True

                    outs = run_one_classification(
                        dataset=ds,
                        cfg=cfg,
                        data_dir=data_dir,
                        download=download,
                        n_train_pool=n_train_pool,
                        n_val_pool=n_val_pool,
                        bas_h=bas_h,
                        bas_w=bas_w,
                    )
                    results.extend(outs)

                    for out in outs:
                        if out["baseline_type"] == "paper":
                            print(f"[OK] cls {ds} PAPER seed={seed} delta={delta} sigma={sigma} n_qubits=8 "
                                  f"L={out['L']} drift_max={out['trace_dist_max']:.6f} drop={out['acc_drop']:.4f} "
                                  f"viol={out.get('dq_violations_over_replaced', 'NA')}")
                        else:
                            print(f"[OK] cls {ds} RAND  seed={seed} delta={delta} sigma={sigma} n_qubits=8 "
                                  f"L={out['L']} drift_max={out['trace_dist_max']:.6f} drop={out['acc_drop']:.4f}")

                # --- bas and vqe use n_qubits=4 ---
                cfg = PaperProtocolConfig(
                    device=device,
                    n_qubits=4,  # 4 qubits for bas and vqe
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

                # bas classification
                outs = run_one_classification(
                    dataset="bas",
                    cfg=cfg,
                    data_dir=data_dir,
                    download=download,
                    n_train_pool=n_train_pool,
                    n_val_pool=n_val_pool,
                    bas_h=bas_h,
                    bas_w=bas_w,
                )
                results.extend(outs)

                for out in outs:
                    if out["baseline_type"] == "paper":
                        print(f"[OK] cls bas PAPER seed={seed} delta={delta} sigma={sigma} n_qubits=4 "
                              f"L={out['L']} drift_max={out['trace_dist_max']:.6f} drop={out['acc_drop']:.4f} "
                              f"viol={out.get('dq_violations_over_replaced', 'NA')}")
                    else:
                        print(f"[OK] cls bas RAND  seed={seed} delta={delta} sigma={sigma} n_qubits=4 "
                              f"L={out['L']} drift_max={out['trace_dist_max']:.6f} drop={out['acc_drop']:.4f}")

                # VQE
                outs = run_one_vqe(
                    cfg=cfg,
                    vqe_depth=vqe_depth,
                    vqe_steps=vqe_steps,
                    vqe_lr=vqe_lr,
                    J=J,
                    h=h,
                )
                results.extend(outs)

                for out in outs:
                    if out["baseline_type"] == "paper":
                        print(f"[OK] vqe tfim PAPER seed={seed} delta={delta} sigma={sigma} n_qubits=4 "
                              f"L={out['L']} drift_max={out['trace_dist_max']:.6f} dE_max={out['dE_max']:.6f} "
                              f"viol={out.get('dq_violations_over_replaced', 'NA')}")
                    else:
                        print(f"[OK] vqe tfim RAND  seed={seed} delta={delta} sigma={sigma} n_qubits=4 "
                              f"L={out['L']} drift_max={out['trace_dist_max']:.6f} dE_max={out['dE_max']:.6f}")

    # Average over seeds
    print("\n" + "=" * 70)
    print("Averaging results over seeds...")
    print("=" * 70)
    averaged_results = average_results_over_seeds(results)

    # Print averaged results summary
    print(f"\nOriginal results: {len(results)} (from {len(seeds)} seeds)")
    print(f"Averaged results: {len(averaged_results)} (unique configurations)")

    payload = {"meta": meta, "results": averaged_results, "elapsed_sec": time.time() - t0}

    os.makedirs("./results", exist_ok=True)
    out_path = "./results/qiprune_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    print(f"Total configurations: {len(averaged_results)}")
    print(f"Each averaged over {len(seeds)} seed(s)")
    print(f"Time elapsed: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    main()
