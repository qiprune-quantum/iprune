"""Matched-K ablation: dynamic hybrid-prefix deletion versus frozen-prefix ranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters_with_diagnostics
from iprune.experiments.vqe_finetune import fine_tune_vqe_circuit
from iprune.gates import build_hea_gates_from_parameters
from iprune.pruning_true import (
    audit_fixed_identity_deletion_path,
    q_iPrune_fixed_k_identity,
    q_iPrune_iterative_identity_audited,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="TFIM dynamic-versus-static pruning ablation.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--finetune_steps", type=int, default=40)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)

    n_qubits, J, h = 4, 1.0, 1.0
    optimized = optimize_vqe_parameters_with_diagnostics(VQEConfig(
        n_qubits=n_qubits, depth=args.depth, steps=args.steps, lr=0.1,
        seed=args.seed, J=J, h=h, min_steps=200, convergence_window=20,
        convergence_tol=5e-5,
    ))
    if not optimized.converged:
        raise RuntimeError("TFIM baseline did not meet the frozen convergence rule.")
    all_gates, baseline = build_hea_gates_from_parameters(
        optimized.params, include_cnot=True, include_identity_rot=True
    )
    initial = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial[0, 0] = 1.0
    control = fine_tune_vqe_circuit(
        circuit=baseline, n_qubits=n_qubits, J=J, h=h, steps=args.finetune_steps,
        lr=0.05, device="default.qubit"
    )
    control_energy = float(control.energy_history[-1])
    rows: list[dict] = []
    for delta in args.deltas:
        cfg = PaperProtocolConfig(device="default.qubit", n_qubits=n_qubits,
            depth=args.depth, M=1, seed=args.seed, gamma=0.0, alpha=0.6,
            qbeta=0.0, delta=float(delta), rot_pool_k=1, rot_pool_sigma=0.0,
            include_cnot=True)
        _, q, weights, _, epsilon = make_q_geometry(cfg)
        dynamic = q_iPrune_iterative_identity_audited(
            all_gates=all_gates, baseline_circuit=baseline, n_qubits=n_qubits,
            device_name="default.qubit", task_states=initial, Gq_diag=weights,
            q_epsilon=epsilon, physical_epsilon=epsilon)
        static = q_iPrune_fixed_k_identity(
            all_gates=all_gates, baseline_circuit=baseline, n_qubits=n_qubits,
            device_name="default.qubit", task_states=initial, Gq_diag=weights,
            physical_epsilon=epsilon, deletion_budget=len(dynamic.accepted_positions),
            require_full_budget=True)
        static_audit = audit_fixed_identity_deletion_path(
            all_gates=all_gates, baseline_circuit=baseline,
            selected_positions=static.selected_positions, n_qubits=n_qubits,
            device_name="default.qubit", task_states=initial, Gq_diag=weights,
            physical_epsilon=epsilon)
        for method, circuit, positions, path_sum, actual, violations in [
            ("dynamic_hybrid_prefix", dynamic.compact_circuit, dynamic.accepted_positions,
             dynamic.path_trace_sum_max, dynamic.actual_trace_distance_max,
             dynamic.physical_threshold_violations),
            ("static_frozen_prefix_matched_k", static.compact_circuit, static.selected_positions,
             static_audit.path_trace_sum_max, static_audit.actual_trace_distance_max,
             static_audit.physical_threshold_violations),
        ]:
            tuned = fine_tune_vqe_circuit(circuit=circuit, n_qubits=n_qubits, J=J, h=h,
                steps=args.finetune_steps, lr=0.05, device="default.qubit")
            rows.append({"seed": args.seed, "delta": float(delta), "q": float(q),
                "method": method, "matched_delete_count": len(positions),
                "deleted_pos_ids": sorted(int(p) for p in positions),
                "energy_finetuned": float(tuned.energy_history[-1]),
                "energy_gap_finetuned_vs_control": float(tuned.energy_history[-1] - control_energy),
                "path_trace_sum_max": float(path_sum), "actual_trace_distance_max": float(actual),
                "physical_threshold_violations": int(violations),
                "static_path_bound_violations": int(static_audit.path_bound_violations) if method.startswith("static") else 0,
            })
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"meta": {"protocol": "same_trained_TFIM__dynamic_hybrid_prefix_vs_frozen_prefix__matched_realized_K", "seed": args.seed, "vqe_converged": optimized.converged, "vqe_steps_run": optimized.steps_run, "deltas": args.deltas}, "results": rows}, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
