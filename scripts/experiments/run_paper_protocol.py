from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from iprune.gates import build_hea_gates
from iprune.experiments.protocol import PaperProtocolConfig, run_positionaware_paper_protocol, simulate_outputs
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


PROJECT_DIR = Path(__file__).resolve().parents[2]


def all_gates_builder(cfg: PaperProtocolConfig):
    return build_hea_gates(
        cfg.n_qubits, cfg.depth, seed=cfg.seed,
        rot_pool_k=cfg.rot_pool_k,
        rot_pool_sigma=cfg.rot_pool_sigma,
        include_cnot=cfg.include_cnot,
    )


def main():
    ap = argparse.ArgumentParser("Paper-aligned protocol runner with TASK metrics (qiprune_pl)")
    ap.add_argument("--task", type=str, choices=["classification", "vqe"], default="classification")
    ap.add_argument("--dataset", type=str, default="mnist49",
                    choices=["mnist49", "fashion_sb", "bas"],
                    help="classification dataset when task=classification")

    ap.add_argument("--device", type=str, default="default.qubit")
    ap.add_argument("--n_qubits", type=int, default=4)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    # paper params
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--qbeta", type=float, default=0.0,
                    help="q-deformation strength; default 0 selects the physical q=1 rule")
    ap.add_argument("--delta", type=float, choices=[0.01, 0.02], default=0.02)

    # candidate pool params
    ap.add_argument("--rot_pool_k", type=int, default=4)
    ap.add_argument("--rot_pool_sigma", type=float, default=0.003)
    ap.add_argument("--include_cnot", action="store_true")

    # classification data options
    ap.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_DIR / "data"),
        help="dataset directory (defaults to the data folder beside this script)",
    )
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--n_train_pool", type=int, default=4000)
    ap.add_argument("--n_val_pool", type=int, default=2000)

    # BAS
    ap.add_argument("--bas_h", type=int, default=4)
    ap.add_argument("--bas_w", type=int, default=4)

    # VQE options
    ap.add_argument("--vqe_depth", type=int, default=6)
    ap.add_argument("--vqe_steps", type=int, default=40)
    ap.add_argument("--vqe_lr", type=float, default=0.1)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="write this completed row to a new JSON file (parent directories are created)",
    )

    args = ap.parse_args()

    cfg = PaperProtocolConfig(
        device=args.device,
        n_qubits=args.n_qubits,
        depth=args.depth,
        M=args.M,
        seed=args.seed,
        gamma=args.gamma,
        alpha=args.alpha,
        qbeta=args.qbeta,
        delta=args.delta,
        rot_pool_k=args.rot_pool_k,
        rot_pool_sigma=args.rot_pool_sigma,
        include_cnot=args.include_cnot,
    )

    # -------- Build D per paper + compute protocol outputs --------
    task_states = None
    protocol_out = None

    if args.task == "classification":
        tcfg = ToyVisionConfig(
            n_qubits=args.n_qubits,
            data_dir=args.data_dir,
            seed=args.seed,
            download=(not args.no_download),
            n_train_pool=args.n_train_pool,
            n_val_pool=args.n_val_pool,
        )

        if args.dataset == "mnist49":
            tr_states, tr_y, va_states, va_y = build_mnist49_splits(tcfg)
        elif args.dataset == "fashion_sb":
            tr_states, tr_y, va_states, va_y = build_fashion_sandal_boot_splits(tcfg)
        else:
            tr_states, tr_y, va_states, va_y = build_bas_splits(tcfg, H=args.bas_h, W=args.bas_w)

        # Paper: D from validation, size M
        task_states, task_labels = sample_task_ensemble_from_validation(va_states, va_y, M=args.M, seed=args.seed)

        protocol_out, baseline_circuit, res = run_positionaware_paper_protocol(
            cfg=cfg,
            all_gates_builder=all_gates_builder,
            task_states=task_states,
            return_artifacts=True,
        )

        # ----- TASK metric: accuracy drop -----
        # We evaluate on *full* validation pool (not just M), which is standard.
        pruned_circuit = res.selected_circuit

        # --- NEW: paper-critical dq stats over actually replaced gates ---
        protocol_out["dq_max_over_replaced"] = float(res.dq_max_over_replaced)
        protocol_out["dq_mean_over_replaced"] = float(res.dq_mean_over_replaced)
        protocol_out["dq_violations_over_replaced"] = int(res.dq_violations_over_replaced)

        # optional: also store global stats computed inside pruning (if you added them)
        if hasattr(res, "dq_max_global"):
            protocol_out["dq_max_global_pruner"] = float(res.dq_max_global)
        if hasattr(res, "dq_mean_global"):
            protocol_out["dq_mean_global_pruner"] = float(res.dq_mean_global)

        # Get final states for train pool (baseline circuit only) and val pool (baseline + pruned)
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
        protocol_out.update(acc_out)

    else:
        vcfg = VQETwitterConfig(
            n_qubits=args.n_qubits,
            depth=args.vqe_depth,
            steps=args.vqe_steps,
            lr=args.vqe_lr,
            seed=args.seed,
            J=args.J,
            h=args.h,
        )
        task_states = sample_intermediate_states_from_vqe(vcfg, M=args.M, device=args.device)

        protocol_out, baseline_circuit, res = run_positionaware_paper_protocol(
            cfg=cfg,
            all_gates_builder=all_gates_builder,
            task_states=task_states,
            return_artifacts=True,
        )

        # ----- TASK metric: energy deviation ΔE on the same task ensemble -----
        pruned_circuit = res.selected_circuit

        protocol_out["dq_max_over_replaced"] = float(res.dq_max_over_replaced)
        protocol_out["dq_mean_over_replaced"] = float(res.dq_mean_over_replaced)
        protocol_out["dq_violations_over_replaced"] = int(res.dq_violations_over_replaced)

        if hasattr(res, "dq_max_global"):
            protocol_out["dq_max_global_pruner"] = float(res.dq_max_global)
        if hasattr(res, "dq_mean_global"):
            protocol_out["dq_mean_global_pruner"] = float(res.dq_mean_global)

        out_base = simulate_outputs(baseline_circuit, task_states, cfg.n_qubits, cfg.device)
        out_prun = simulate_outputs(pruned_circuit, task_states, cfg.n_qubits, cfg.device)

        eout = evaluate_vqe_energy_deviation(
            n_qubits=cfg.n_qubits,
            states_base=out_base,
            states_prun=out_prun,
            J=args.J,
            h=args.h,
        )
        protocol_out.update(eout)

    protocol_out["task"] = args.task
    protocol_out["dataset"] = args.dataset if args.task == "classification" else "tfim"
    protocol_out["baseline_type"] = "paper"
    protocol_out["rot_pool_k"] = int(args.rot_pool_k)
    protocol_out["rot_pool_sigma"] = float(args.rot_pool_sigma)
    protocol_out["include_cnot"] = bool(args.include_cnot)

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_path}"
            )
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(protocol_out, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(output_path)

    # -------- Print paper-aligned report --------
    print("\n=== Paper-aligned Experimental Setup & Evaluation Protocol (with TASK metrics) ===")
    print(f"task={args.task}" + (f", dataset={args.dataset}" if args.task == "classification" else ""))
    print(f"device={protocol_out['device']}")
    print(f"n_qubits={protocol_out['n_qubits']}, depth={protocol_out['depth']}, M={protocol_out['M']}, seed={protocol_out['seed']}")
    print(f"gamma={protocol_out['gamma']:.4f}, alpha={protocol_out['alpha']:.3f} -> lambda={protocol_out['lambda']:.4f}")
    print(f"qbeta={protocol_out['qbeta']:.3f} -> q={protocol_out['q']:.6f}")
    print(f"delta={protocol_out['delta']:.3f} -> epsilon_q={protocol_out['epsilon_q']:.6f} (paper rule)")
    print(
        f"all_candidates={protocol_out['all_candidates']} "
        f"prunable_candidates={protocol_out['prunable_candidates']} "
        f"positions={protocol_out['positions']}"
    )
    print(
        f"replaced_candidates(L)={protocol_out['L']} "
        f"replace_ratio_over_candidates={protocol_out['replace_ratio_over_candidates']:.2%}"
    )
    print(
        f"selected_replacement_count={protocol_out['selected_replacement_count']} "
        f"redundant_ordinary={protocol_out['redundant_ordinary_candidates']} "
        f"redundant_identity={protocol_out['redundant_identity_candidates']}"
    )
    print(f"Empirical drift (trace distance): mean={protocol_out['trace_dist_mean']:.6f}, max={protocol_out['trace_dist_max']:.6f}")
    print(f"Theory RHS raw={protocol_out['rhs_raw']:.6f}, clipped={protocol_out['rhs_clipped']:.6f}")
    print(f"dq_max_global={protocol_out['dq_max_global']:.6f}, dq_mean_avg={protocol_out['dq_mean_avg']:.6f}")
    # --- NEW: must-hold-by-construction paper check ---
    if "dq_max_over_replaced" in protocol_out:
        print(f"dq_max_over_replaced={protocol_out['dq_max_over_replaced']:.6f}, "
              f"dq_mean_over_replaced={protocol_out['dq_mean_over_replaced']:.6f}, "
              f"violations={protocol_out['dq_violations_over_replaced']} (expect 0)")

    if args.task == "classification":
        print(f"[TASK] acc_base={protocol_out['acc_base']:.4f}, acc_prun={protocol_out['acc_prun']:.4f}, acc_drop={protocol_out['acc_drop']:.4f}")
    else:
        print(f"[TASK] dE_mean={protocol_out['dE_mean']:.6f}, dE_max={protocol_out['dE_max']:.6f} "
              f"(E_base_mean={protocol_out['energy_base_mean']:.6f}, E_prun_mean={protocol_out['energy_prun_mean']:.6f})")
    if args.output_json:
        print(f"Saved row: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
