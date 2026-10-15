from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters_with_diagnostics
from iprune.experiments.vqe_finetune import fine_tune_vqe_circuit
from iprune.gates import build_hea_gates_from_parameters
from iprune.pruning_true import q_iPrune_iterative_identity_audited


def main() -> None:
    parser = argparse.ArgumentParser(description="Cumulative path-budget extension for TFIM.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)

    n_qubits = args.n_qubits
    J, h = 1.0, 1.0
    optimized = optimize_vqe_parameters_with_diagnostics(
        VQEConfig(
            n_qubits=n_qubits,
            depth=args.depth,
            steps=args.steps,
            lr=0.1,
            seed=args.seed,
            J=J,
            h=h,
            min_steps=200,
            convergence_window=20,
            convergence_tol=5e-5,
        )
    )
    if not optimized.converged:
        raise RuntimeError("TFIM baseline did not meet the frozen convergence rule.")
    all_gates, baseline = build_hea_gates_from_parameters(
        optimized.params, include_cnot=True, include_identity_rot=True
    )
    initial = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial[0, 0] = 1.0
    control = fine_tune_vqe_circuit(
        circuit=baseline,
        n_qubits=n_qubits,
        J=J,
        h=h,
        steps=args.finetune_steps,
        lr=0.05,
        device="default.qubit",
    )
    control_energy = float(control.energy_history[-1])

    rows: list[dict[str, Any]] = []
    for delta in args.deltas:
        cfg = PaperProtocolConfig(
            device="default.qubit",
            n_qubits=n_qubits,
            depth=args.depth,
            M=1,
            seed=args.seed,
            gamma=0.0,
            alpha=0.6,
            qbeta=0.0,
            delta=float(delta),
            rot_pool_k=1,
            rot_pool_sigma=0.0,
            include_cnot=True,
        )
        _lam, q, weights, _Mq, epsilon = make_q_geometry(cfg)
        budgets = [None, *args.budgets]
        for budget in budgets:
            result = q_iPrune_iterative_identity_audited(
                all_gates=all_gates,
                baseline_circuit=baseline,
                n_qubits=n_qubits,
                device_name="default.qubit",
                task_states=initial,
                Gq_diag=weights,
                q_epsilon=epsilon,
                physical_epsilon=epsilon,
                path_budget=budget,
            )
            compact = result.compact_circuit
            tuned = fine_tune_vqe_circuit(
                circuit=compact,
                n_qubits=n_qubits,
                J=J,
                h=h,
                steps=args.finetune_steps,
                lr=0.05,
                device="default.qubit",
            )
            rows.append(
                {
                    "seed": args.seed,
                    "delta": float(delta),
                    "q": float(q),
                    "epsilon": float(epsilon),
                    "path_budget": None if budget is None else float(budget),
                    "deleted_count": len(result.accepted_positions),
                    "deleted_pos_ids": sorted(int(p) for p in result.accepted_positions),
                    "energy_gap_finetuned_vs_control": float(
                        tuned.energy_history[-1] - control_energy
                    ),
                    "actual_trace_distance_max": float(result.actual_trace_distance_max),
                    "path_trace_sum_max": float(result.path_trace_sum_max),
                    "physical_threshold_violations": int(
                        result.physical_threshold_violations
                    ),
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "protocol": "tfim_cumulative_path_budget_extension",
            "seed": args.seed,
            "n_qubits": n_qubits,
            "vqe_converged": bool(optimized.converged),
            "vqe_steps_run": int(optimized.steps_run),
            "deltas": args.deltas,
            "budgets": budgets,
        },
        "results": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
