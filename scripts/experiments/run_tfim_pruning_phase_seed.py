from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pennylane as qml

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters_with_diagnostics, tfim_hamiltonian
from iprune.experiments.vqe_finetune import fine_tune_vqe_circuit
from iprune.gates import build_hea_gates_from_parameters
from iprune.pruning_true import q_iPrune_iterative_identity_audited
from scripts.experiments.run_vqe_qiprune_vs_gradient_seed import _atomic_new_json, _evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense TFIM pruning-transition scan for one seed.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqe_depth", type=int, default=4)
    parser.add_argument("--vqe_steps", type=int, default=1200)
    parser.add_argument("--vqe_min_steps", type=int, default=200)
    parser.add_argument("--vqe_convergence_window", type=int, default=20)
    parser.add_argument("--vqe_convergence_tol", type=float, default=5e-5)
    parser.add_argument("--vqe_lr", type=float, default=0.1)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--finetune_lr", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument(
        "--qbeta", type=float, default=0.0,
        help="Optional q-weighted diagnostic. The main physical-only protocol uses 0 (q=1).",
    )
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    args = parser.parse_args()

    optimization = optimize_vqe_parameters_with_diagnostics(
        VQEConfig(
            n_qubits=args.n_qubits,
            depth=args.vqe_depth,
            steps=args.vqe_steps,
            lr=args.vqe_lr,
            seed=args.seed,
            J=args.J,
            h=args.h,
            min_steps=args.vqe_min_steps,
            convergence_window=args.vqe_convergence_window,
            convergence_tol=args.vqe_convergence_tol,
        ),
        device=args.device,
    )
    if not optimization.converged:
        raise RuntimeError(
            f"VQE seed {args.seed} did not converge: steps={optimization.steps_run}, "
            f"window_range={optimization.final_window_range:.6g}"
        )

    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        optimization.params, include_cnot=True, include_identity_rot=True
    )
    initial_state = np.zeros((1, 2**args.n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    baseline_state = simulate_outputs(baseline_circuit, initial_state, args.n_qubits, args.device)
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
    energy_original = float(optimization.energy_history[-1])
    energy_control = float(control.energy_history[-1])
    hamiltonian = np.asarray(
        qml.matrix(tfim_hamiltonian(args.n_qubits, args.J, args.h), wire_order=range(args.n_qubits)),
        dtype=np.complex128,
    )
    operator_norm = float(np.max(np.abs(np.linalg.eigvalsh(hamiltonian))))
    rotation_count = sum(g.gate_type == "Rot" for g in baseline_circuit)

    rows = []
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
        started = perf_counter()
        result = q_iPrune_iterative_identity_audited(
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            n_qubits=args.n_qubits,
            device_name=args.device,
            task_states=initial_state,
            Gq_diag=Gq_diag,
            q_epsilon=epsilon_q,
            physical_epsilon=epsilon_q,
        )
        selection_time_ms = (perf_counter() - started) * 1000.0
        row = _evaluate(
            method="qiprune_iterative_hybrid_path",
            compact_circuit=result.compact_circuit,
            deleted_pos_ids=result.accepted_positions,
            baseline_circuit=baseline_circuit,
            baseline_state=baseline_state,
            control_state=control_state,
            energy_original=energy_original,
            energy_control=energy_control,
            energy_initial=float(optimization.energy_history[0]),
            n_qubits=args.n_qubits,
            J=args.J,
            h=args.h,
            finetune_steps=args.finetune_steps,
            finetune_lr=args.finetune_lr,
            device=args.device,
            include_finetune_history=True,
        )
        observed = float(row["abs_dE_one_shot_vs_original"])
        path_bound = float(2.0 * operator_norm * min(1.0, result.path_trace_sum_max))
        calibration_bound = float(
            2.0 * operator_norm * min(1.0, len(result.accepted_positions) * np.sin(epsilon_q))
        )
        if observed > path_bound + 1e-9 or observed > calibration_bound + 1e-9:
            raise AssertionError("Observed energy drift exceeded an audited bound.")
        cumulative_path_cost = 0.0
        accepted_path = []
        for step, position in enumerate(result.accepted_positions, start=1):
            physical_angle = float(result.physical_angle_max_by_position[position])
            q_angle = float(result.q_angle_max_by_position[position])
            local_trace_cost = float(np.sin(physical_angle))
            cumulative_path_cost += local_trace_cost
            accepted_path.append(
                {
                    "step": step,
                    "position": int(position),
                    "q_angle": q_angle,
                    "physical_angle": physical_angle,
                    "local_trace_cost": local_trace_cost,
                    "cumulative_trace_cost": cumulative_path_cost,
                    "trace_distance_upper_bound": min(1.0, cumulative_path_cost),
                }
            )
        if not np.isclose(cumulative_path_cost, result.path_trace_sum_max, atol=1e-10):
            raise AssertionError(
                "Accepted-path reconstruction does not match the audited path sum."
            )
        row.update(
            {
                "experiment": "tfim_pruning_transition_dense_scan",
                "task": "vqe",
                "dataset": "tfim",
                "seed": args.seed,
                "n_qubits": args.n_qubits,
                "vqe_depth": args.vqe_depth,
                "vqe_steps_budget": args.vqe_steps,
                "vqe_steps_run": int(optimization.steps_run),
                "vqe_converged": bool(optimization.converged),
                "vqe_final_window_range": float(optimization.final_window_range),
                "finetune_steps": args.finetune_steps,
                "delta": float(delta),
                "epsilon_q": float(epsilon_q),
                "lambda": float(lam),
                "q": float(q),
                "Mq": float(Mq),
                "rotation_count_before": int(rotation_count),
                "rotation_prune_ratio": len(result.accepted_positions) / max(1, rotation_count),
                "selection_time_ms": float(selection_time_ms),
                "realized_path_trace_sum_max": float(result.path_trace_sum_max),
                "actual_trace_distance_max": float(result.actual_trace_distance_max),
                "accepted_path": accepted_path,
                "q_angles_by_accepted_position": {
                    str(position): float(result.q_angle_max_by_position[position])
                    for position in result.accepted_positions
                },
                "physical_angles_by_accepted_position": {
                    str(position): float(result.physical_angle_max_by_position[position])
                    for position in result.accepted_positions
                },
                "hamiltonian_operator_norm": operator_norm,
                "realized_path_observable_bound": path_bound,
                "calibration_L_epsilon_observable_bound": calibration_bound,
                "physical_threshold_violations": int(result.physical_threshold_violations),
            }
        )
        rows.append(row)
        print(
            f"[PHASE] seed={args.seed} delta={delta:g} K={row['pruned_positions']} "
            f"rot_ratio={row['rotation_prune_ratio']:.3f} "
            f"one={row['energy_gap_one_shot_vs_control']:.6f} "
            f"tuned={row['energy_gap_finetuned_vs_control']:.6f}",
            flush=True,
        )

    _atomic_new_json(
        args.output_json,
        {
            "meta": {
                "status": "formal",
                "protocol": "converged_tfim_dynamic_prefix_real_deletion_dense_threshold_scan",
                "selection_geometry": "physical_fubini_study" if args.qbeta == 0.0 else "q_weighted_and_physical_dual_threshold",
                "qbeta": float(args.qbeta),
                "seed": args.seed,
                "deltas": args.deltas,
                "expected_rows": len(args.deltas),
                "vqe_steps_run": int(optimization.steps_run),
                "vqe_converged": bool(optimization.converged),
                "vqe_final_window_range": float(optimization.final_window_range),
            },
            "results": rows,
        },
    )
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
