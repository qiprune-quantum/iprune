from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs, trace_distance_stats
from iprune.experiments.task_metrics import evaluate_vqe_energy_deviation
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters
from iprune.experiments.vqe_finetune import fine_tune_vqe_circuit
from iprune.gates import build_hea_gates_from_parameters
from iprune.pruning_true import q_iPrune_true_gate_pruning_by_identity
from scripts.legacy.sweep_all_v2 import _circuit_signature, _logical_resource_metrics


def _atomic_new_json(path: Path, payload: Dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-seed TFIM VQE true-pruning sweep with matched extra optimization control."
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 0.9, 1.0, 1.2, 1.5, 2.0])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqe_depth", type=int, default=6)
    parser.add_argument("--vqe_steps", type=int, default=40)
    parser.add_argument("--vqe_lr", type=float, default=0.1)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--finetune_lr", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--qbeta", type=float, default=0.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    args = parser.parse_args()

    vcfg = VQEConfig(
        n_qubits=args.n_qubits,
        depth=args.vqe_depth,
        steps=args.vqe_steps,
        lr=args.vqe_lr,
        seed=args.seed,
        J=args.J,
        h=args.h,
    )
    optimized_params, baseline_history = optimize_vqe_parameters(vcfg, device=args.device)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        optimized_params, include_cnot=True, include_identity_rot=True
    )
    initial_state = np.zeros((1, 2**args.n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    baseline_state = simulate_outputs(
        baseline_circuit, initial_state, args.n_qubits, args.device
    )

    # Fair control: give the unpruned circuit the same number of extra steps.
    control = fine_tune_vqe_circuit(
        circuit=baseline_circuit,
        n_qubits=args.n_qubits,
        J=args.J,
        h=args.h,
        steps=args.finetune_steps,
        lr=args.finetune_lr,
        device=args.device,
    )
    control_state = control.final_state.reshape(1, -1)
    control_metrics = evaluate_vqe_energy_deviation(
        n_qubits=args.n_qubits,
        states_base=baseline_state,
        states_prun=control_state,
        J=args.J,
        h=args.h,
    )
    energy_original = float(baseline_history[-1])
    energy_control = float(control.energy_history[-1])
    resources_before = _logical_resource_metrics(baseline_circuit, args.n_qubits)

    rows: List[Dict[str, Any]] = []
    for delta in args.deltas:
        cfg = PaperProtocolConfig(
            device=args.device,
            n_qubits=args.n_qubits,
            depth=args.vqe_depth,
            M=1,
            seed=args.seed,
            gamma=args.gamma,
            alpha=args.alpha,
            qbeta=args.qbeta,
            delta=delta,
            rot_pool_k=1,
            rot_pool_sigma=0.0,
            include_cnot=True,
        )
        lam, q, Gq_diag, Mq, epsilon_q = make_q_geometry(cfg)
        pruned = q_iPrune_true_gate_pruning_by_identity(
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            n_qubits=args.n_qubits,
            device_name=args.device,
            task_states=initial_state,
            Gq_diag=Gq_diag,
            epsilon_q=epsilon_q,
        )
        one_shot_state = simulate_outputs(
            pruned.compact_circuit, initial_state, args.n_qubits, args.device
        )
        one_shot_metrics = evaluate_vqe_energy_deviation(
            n_qubits=args.n_qubits,
            states_base=baseline_state,
            states_prun=one_shot_state,
            J=args.J,
            h=args.h,
        )
        tuned = fine_tune_vqe_circuit(
            circuit=pruned.compact_circuit,
            n_qubits=args.n_qubits,
            J=args.J,
            h=args.h,
            steps=args.finetune_steps,
            lr=args.finetune_lr,
            device=args.device,
        )
        tuned_state = tuned.final_state.reshape(1, -1)
        tuned_vs_original = evaluate_vqe_energy_deviation(
            n_qubits=args.n_qubits,
            states_base=baseline_state,
            states_prun=tuned_state,
            J=args.J,
            h=args.h,
        )
        trace_one_mean, trace_one_max = trace_distance_stats(baseline_state, one_shot_state)
        trace_tuned_mean, trace_tuned_max = trace_distance_stats(control_state, tuned_state)
        resources_after = _logical_resource_metrics(pruned.compact_circuit, args.n_qubits)
        energy_one = float(tuned.energy_history[0])
        energy_tuned = float(tuned.energy_history[-1])
        gap_one_control = energy_one - energy_control
        gap_tuned_control = energy_tuned - energy_control
        recovery = None
        if gap_one_control > 1e-12:
            recovery = float((gap_one_control - gap_tuned_control) / gap_one_control)

        row = {
            "experiment_family": "supplementary_upgrade",
            "experiment": "true_pruning_vqe_post_finetune",
            "task": "vqe",
            "dataset": "tfim",
            "seed": args.seed,
            "n_qubits": args.n_qubits,
            "vqe_depth": args.vqe_depth,
            "vqe_steps": args.vqe_steps,
            "vqe_lr": args.vqe_lr,
            "finetune_steps": args.finetune_steps,
            "finetune_lr": args.finetune_lr,
            "delta": delta,
            "epsilon_q": float(epsilon_q),
            "lambda": float(lam),
            "q": float(q),
            "Mq": float(Mq),
            "pruned_positions": int(pruned.pruned_positions),
            "kept_positions": int(pruned.kept_positions),
            "actual_gate_prune_ratio": float(pruned.pruned_positions / max(1, pruned.gate_count_before)),
            "gate_count_before": int(pruned.gate_count_before),
            "gate_count_after": int(pruned.gate_count_after),
            "one_qubit_gate_count_before": resources_before["one_qubit"],
            "one_qubit_gate_count_after": resources_after["one_qubit"],
            "two_qubit_gate_count_before": resources_before["two_qubit"],
            "two_qubit_gate_count_after": resources_after["two_qubit"],
            "logical_depth_before": resources_before["logical_depth"],
            "logical_depth_after": resources_after["logical_depth"],
            "trainable_rotations_after": tuned.trainable_rotations,
            "energy_initial": float(baseline_history[0]),
            "energy_original_optimized": energy_original,
            "energy_full_control_after_extra_steps": energy_control,
            "energy_pruned_one_shot": energy_one,
            "energy_pruned_finetuned": energy_tuned,
            "abs_dE_one_shot_vs_original": float(one_shot_metrics["dE_max"]),
            "abs_dE_finetuned_vs_original": float(tuned_vs_original["dE_max"]),
            "energy_gap_one_shot_vs_control": float(gap_one_control),
            "energy_gap_finetuned_vs_control": float(gap_tuned_control),
            "recovery_fraction_vs_control": recovery,
            "trace_dist_one_shot_vs_original_mean": float(trace_one_mean),
            "trace_dist_one_shot_vs_original_max": float(trace_one_max),
            "trace_dist_finetuned_vs_control_mean": float(trace_tuned_mean),
            "trace_dist_finetuned_vs_control_max": float(trace_tuned_max),
            "control_abs_dE_vs_original": float(control_metrics["dE_max"]),
            "dq_mean_over_replaced": float(pruned.dq_mean_over_replaced),
            "dq_max_over_replaced": float(pruned.dq_max_over_replaced),
            "dq_violations_over_replaced": int(pruned.dq_violations_over_replaced),
            "physical_angle_max_over_replaced": float(pruned.physical_angle_max_over_replaced),
            "physical_violations_over_replaced": int(pruned.physical_violations_over_replaced),
            "baseline_circuit_signature": _circuit_signature(baseline_circuit),
            "pruned_circuit_signature": _circuit_signature(pruned.compact_circuit),
            "finetuned_circuit_signature": _circuit_signature(tuned.circuit),
        }
        rows.append(row)
        print(
            f"[OK] seed={args.seed} delta={delta:g} gates="
            f"{row['gate_count_before']}->{row['gate_count_after']} "
            f"E_one={energy_one:.6f} E_tuned={energy_tuned:.6f} "
            f"gap_control={gap_tuned_control:.6f}",
            flush=True,
        )

    payload = {
        "meta": {
            "experiment_family": "supplementary_upgrade",
            "protocol": "optimized_tfim_true_identity_deletion_then_structure_fixed_finetune",
            "fair_control": "unpruned circuit receives the same extra optimization steps",
            "seed": args.seed,
            "deltas": args.deltas,
            "expected_rows": len(args.deltas),
        },
        "results": rows,
    }
    _atomic_new_json(args.output_json, payload)
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
