from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

import numpy as np
import pennylane as qml

from iprune.experiments.gradient_gate_pruning import vqe_integrated_gate_gradient_scores
from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs, trace_distance_stats
from iprune.experiments.qfi_gate_pruning import diagonal_qfi_gate_saliency_scores
from iprune.experiments.task_metrics import evaluate_vqe_energy_deviation
from iprune.experiments.tasks_vqe import (
    VQEConfig,
    optimize_vqe_parameters_with_diagnostics,
    tfim_hamiltonian,
)
from iprune.experiments.vqe_finetune import fine_tune_vqe_circuit
from iprune.gates import GateOp, build_hea_gates_from_parameters
from iprune.pruning_true import q_iPrune_iterative_identity_audited
from scripts.legacy.sweep_all_v2 import _circuit_signature, _logical_resource_metrics


def _atomic_new_json(path: Path, payload: Dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _deleted_ids(before: Sequence[GateOp], after: Sequence[GateOp]) -> List[int]:
    kept = {gate.pos_id for gate in after}
    return sorted(gate.pos_id for gate in before if gate.pos_id not in kept)


def _rotation_positions(circuit: Sequence[GateOp]) -> List[int]:
    return [int(gate.pos_id) for gate in circuit if gate.gate_type == "Rot"]


def _without_positions(
    circuit: Sequence[GateOp], positions: Sequence[int]
) -> List[GateOp]:
    deleted = {int(position) for position in positions}
    return [gate for gate in circuit if int(gate.pos_id) not in deleted]


def _circular_parameter_magnitude(gate: GateOp) -> float:
    if gate.params is None:
        raise ValueError("Magnitude baseline requires parameterized rotations.")
    wrapped = np.angle(np.exp(1j * np.asarray(gate.params, dtype=np.float64)))
    return float(np.linalg.norm(wrapped))


def _single_gate_energy_probe_ranking(
    *,
    baseline_circuit: Sequence[GateOp],
    baseline_state: np.ndarray,
    n_qubits: int,
    J: float,
    h: float,
    device: str,
) -> tuple[List[int], Dict[int, float], float]:
    initial_state = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    started = perf_counter()
    scores: Dict[int, float] = {}
    for position in _rotation_positions(baseline_circuit):
        state = simulate_outputs(
            _without_positions(baseline_circuit, [position]),
            initial_state,
            n_qubits,
            device,
        )
        scores[position] = float(
            evaluate_vqe_energy_deviation(
                n_qubits=n_qubits,
                states_base=baseline_state,
                states_prun=state,
                J=J,
                h=h,
            )["dE_max"]
        )
    return (
        sorted(scores, key=lambda position: (scores[position], position)),
        scores,
        (perf_counter() - started) * 1000.0,
    )


def _evaluate(
    *,
    method: str,
    compact_circuit: Sequence[GateOp],
    deleted_pos_ids: List[int],
    baseline_circuit: Sequence[GateOp],
    baseline_state: np.ndarray,
    control_state: np.ndarray,
    energy_original: float,
    energy_control: float,
    energy_initial: float,
    n_qubits: int,
    J: float,
    h: float,
    finetune_steps: int,
    finetune_lr: float,
    device: str,
    include_finetune_history: bool = False,
) -> Dict[str, Any]:
    initial_state = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    compact = list(compact_circuit)
    one_shot_state = simulate_outputs(compact, initial_state, n_qubits, device)
    one_shot_metrics = evaluate_vqe_energy_deviation(
        n_qubits=n_qubits,
        states_base=baseline_state,
        states_prun=one_shot_state,
        J=J,
        h=h,
    )
    tuned = fine_tune_vqe_circuit(
        circuit=compact,
        n_qubits=n_qubits,
        J=J,
        h=h,
        steps=finetune_steps,
        lr=finetune_lr,
        device=device,
    )
    tuned_state = tuned.final_state.reshape(1, -1)
    tuned_metrics = evaluate_vqe_energy_deviation(
        n_qubits=n_qubits,
        states_base=baseline_state,
        states_prun=tuned_state,
        J=J,
        h=h,
    )
    trace_one_mean, trace_one_max = trace_distance_stats(baseline_state, one_shot_state)
    trace_tuned_mean, trace_tuned_max = trace_distance_stats(control_state, tuned_state)
    resources_before = _logical_resource_metrics(list(baseline_circuit), n_qubits)
    resources_after = _logical_resource_metrics(compact, n_qubits)
    energy_one = float(tuned.energy_history[0])
    energy_tuned = float(tuned.energy_history[-1])
    gap_one_control = energy_one - energy_control
    gap_tuned_control = energy_tuned - energy_control
    recovery = None
    if gap_one_control > 1e-12:
        recovery = float((gap_one_control - gap_tuned_control) / gap_one_control)

    output = {
        "method": method,
        "deleted_pos_ids": deleted_pos_ids,
        "pruned_positions": len(deleted_pos_ids),
        "kept_positions": len(baseline_circuit) - len(deleted_pos_ids),
        "actual_gate_prune_ratio": len(deleted_pos_ids) / max(1, len(baseline_circuit)),
        "gate_count_before": len(baseline_circuit),
        "gate_count_after": len(compact),
        "one_qubit_gate_count_before": resources_before["one_qubit"],
        "one_qubit_gate_count_after": resources_after["one_qubit"],
        "two_qubit_gate_count_before": resources_before["two_qubit"],
        "two_qubit_gate_count_after": resources_after["two_qubit"],
        "logical_depth_before": resources_before["logical_depth"],
        "logical_depth_after": resources_after["logical_depth"],
        "trainable_rotations_after": tuned.trainable_rotations,
        "energy_initial": energy_initial,
        "energy_original_optimized": energy_original,
        "energy_full_control_after_extra_steps": energy_control,
        "energy_pruned_one_shot": energy_one,
        "energy_pruned_finetuned": energy_tuned,
        "abs_dE_one_shot_vs_original": float(one_shot_metrics["dE_max"]),
        "abs_dE_finetuned_vs_original": float(tuned_metrics["dE_max"]),
        "energy_gap_one_shot_vs_control": float(gap_one_control),
        "energy_gap_finetuned_vs_control": float(gap_tuned_control),
        "recovery_fraction_vs_control": recovery,
        "trace_dist_one_shot_vs_original_mean": float(trace_one_mean),
        "trace_dist_one_shot_vs_original_max": float(trace_one_max),
        "trace_dist_finetuned_vs_control_mean": float(trace_tuned_mean),
        "trace_dist_finetuned_vs_control_max": float(trace_tuned_max),
        "baseline_circuit_signature": _circuit_signature(list(baseline_circuit)),
        "pruned_circuit_signature": _circuit_signature(compact),
        "finetuned_circuit_signature": _circuit_signature(tuned.circuit),
    }
    if include_finetune_history:
        output["finetune_energy_history"] = [
            float(value) for value in tuned.energy_history
        ]
        output["finetune_gap_vs_control_history"] = [
            float(value - energy_control) for value in tuned.energy_history
        ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converged, matched-realized-L iterative q-iPrune six-baseline comparison."
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument(
        "--vqe_diagnostic_json",
        type=Path,
        help="Write the baseline VQE trajectory before enforcing convergence.",
    )
    parser.add_argument(
        "--vqe_diagnostic_only",
        action="store_true",
        help="Stop after writing --vqe_diagnostic_json; no pruning is run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 0.9, 1.0, 1.2, 1.5, 2.0])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqe_depth", type=int, default=6)
    parser.add_argument("--vqe_steps", type=int, default=1200)
    parser.add_argument("--vqe_min_steps", type=int, default=200)
    parser.add_argument("--vqe_convergence_window", type=int, default=20)
    parser.add_argument("--vqe_convergence_tol", type=float, default=5e-5)
    parser.add_argument("--allow_unconverged", action="store_true")
    parser.add_argument("--vqe_lr", type=float, default=0.1)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--finetune_lr", type=float, default=0.05)
    parser.add_argument("--gradient_points", type=int, default=5)
    parser.add_argument("--random_seed_offset", type=int, default=20260911)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--qbeta", type=float, default=0.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    args = parser.parse_args()
    if args.vqe_diagnostic_only and args.vqe_diagnostic_json is None:
        parser.error("--vqe_diagnostic_only requires --vqe_diagnostic_json")

    vcfg = VQEConfig(
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
    )
    optimization = optimize_vqe_parameters_with_diagnostics(vcfg, device=args.device)
    if args.vqe_diagnostic_json is not None:
        diagnostic_payload = {
            "protocol": "baseline_vqe_convergence_diagnostic",
            "seed": args.seed,
            "n_qubits": args.n_qubits,
            "vqe_depth": args.vqe_depth,
            "vqe_steps_budget": args.vqe_steps,
            "vqe_lr": args.vqe_lr,
            "vqe_min_steps": args.vqe_min_steps,
            "vqe_convergence_window": args.vqe_convergence_window,
            "vqe_convergence_tol": args.vqe_convergence_tol,
            "steps_run": int(optimization.steps_run),
            "converged": bool(optimization.converged),
            "final_window_range": float(optimization.final_window_range),
            "stop_reason": optimization.stop_reason,
            "energy_history": [float(value) for value in optimization.energy_history],
        }
        _atomic_new_json(args.vqe_diagnostic_json, diagnostic_payload)
        print(f"Saved VQE diagnostic: {args.vqe_diagnostic_json.resolve()}", flush=True)
    if args.vqe_diagnostic_only:
        return
    if not optimization.converged and not args.allow_unconverged:
        raise RuntimeError(
            "Baseline VQE did not satisfy the convergence rule: "
            f"{optimization.steps_run}/{args.vqe_steps} steps, final-window "
            f"range={optimization.final_window_range:.6g}."
        )
    optimized_params = optimization.params
    baseline_history = optimization.energy_history
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        optimized_params, include_cnot=True, include_identity_rot=True
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
    energy_original = float(baseline_history[-1])
    energy_control = float(control.energy_history[-1])
    hamiltonian_matrix = np.asarray(
        qml.matrix(
            tfim_hamiltonian(args.n_qubits, args.J, args.h),
            wire_order=range(args.n_qubits),
        ),
        dtype=np.complex128,
    )
    hamiltonian_op_norm = float(
        np.max(np.abs(np.linalg.eigvalsh(hamiltonian_matrix)))
    )

    rot_pos_ids, gradient_scores, endpoint_scores, gradient_time_ms = (
        vqe_integrated_gate_gradient_scores(
            circuit=baseline_circuit,
            n_qubits=args.n_qubits,
            J=args.J,
            h=args.h,
            quadrature_points=args.gradient_points,
            device=args.device,
        )
    )
    gradient_ranking = [
        rot_pos_ids[index]
        for index in sorted(
            range(len(rot_pos_ids)),
            key=lambda index: (gradient_scores[index], rot_pos_ids[index]),
        )
    ]
    rot_by_position = {
        int(gate.pos_id): gate
        for gate in baseline_circuit
        if gate.gate_type == "Rot"
    }
    magnitude_started = perf_counter()
    magnitude_scores = {
        position: _circular_parameter_magnitude(gate)
        for position, gate in rot_by_position.items()
    }
    magnitude_ranking = sorted(
        magnitude_scores, key=lambda position: (magnitude_scores[position], position)
    )
    magnitude_time_ms = (perf_counter() - magnitude_started) * 1000.0
    random_started = perf_counter()
    random_ranking = [
        int(position)
        for position in np.random.default_rng(args.seed + args.random_seed_offset).permutation(
            _rotation_positions(baseline_circuit)
        )
    ]
    random_time_ms = (perf_counter() - random_started) * 1000.0
    probe_ranking, probe_scores, probe_time_ms = _single_gate_energy_probe_ranking(
        baseline_circuit=baseline_circuit,
        baseline_state=baseline_state,
        n_qubits=args.n_qubits,
        J=args.J,
        h=args.h,
        device=args.device,
    )
    qfi_ranking, qfi_scores, qfi_diagonals, qfi_time_ms = (
        diagonal_qfi_gate_saliency_scores(
            circuit=baseline_circuit,
            task_states=initial_state,
            n_qubits=args.n_qubits,
            device=args.device,
        )
    )
    print(
        f"[GRAD] seed={args.seed} rot={len(rot_pos_ids)} "
        f"score=[{gradient_scores.min():.6g},{gradient_scores.max():.6g}] "
        f"endpoint_max={endpoint_scores.max():.6g} time_ms={gradient_time_ms:.1f}",
        flush=True,
    )

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
        started = perf_counter()
        qiprune = q_iPrune_iterative_identity_audited(
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            n_qubits=args.n_qubits,
            device_name=args.device,
            task_states=initial_state,
            Gq_diag=Gq_diag,
            q_epsilon=epsilon_q,
            physical_epsilon=epsilon_q,
        )
        qiprune_time_ms = (perf_counter() - started) * 1000.0
        qiprune_deleted = qiprune.accepted_positions
        delete_count = len(qiprune_deleted)
        gradient_deleted = sorted(gradient_ranking[:delete_count])
        gradient_deleted_set = set(gradient_deleted)
        gradient_circuit = [
            gate for gate in baseline_circuit if gate.pos_id not in gradient_deleted_set
        ]

        common = {
            "experiment_family": "supplementary_upgrade",
            "experiment": "vqe_iterative_qiprune_six_baselines_matched_L",
            "task": "vqe",
            "dataset": "tfim",
            "seed": args.seed,
            "n_qubits": args.n_qubits,
            "vqe_depth": args.vqe_depth,
            "vqe_steps": args.vqe_steps,
            "vqe_lr": args.vqe_lr,
            "vqe_steps_run": int(optimization.steps_run),
            "vqe_converged": bool(optimization.converged),
            "vqe_final_window_range": float(optimization.final_window_range),
            "finetune_steps": args.finetune_steps,
            "finetune_lr": args.finetune_lr,
            "delta": delta,
            "epsilon_q": float(epsilon_q),
            "lambda": float(lam),
            "q": float(q),
            "Mq": float(Mq),
            "matched_delete_count": delete_count,
        }
        q_row = _evaluate(
            method="qiprune_iterative_hybrid_path",
            compact_circuit=qiprune.compact_circuit,
            deleted_pos_ids=qiprune_deleted,
            baseline_circuit=baseline_circuit,
            baseline_state=baseline_state,
            control_state=control_state,
            energy_original=energy_original,
            energy_control=energy_control,
            energy_initial=float(baseline_history[0]),
            n_qubits=args.n_qubits,
            J=args.J,
            h=args.h,
            finetune_steps=args.finetune_steps,
            finetune_lr=args.finetune_lr,
            device=args.device,
        )
        q_row.update(common)
        path_observable_bound = float(
            2.0
            * hamiltonian_op_norm
            * min(1.0, qiprune.path_trace_sum_max)
        )
        calibration_observable_bound = float(
            2.0
            * hamiltonian_op_norm
            * min(1.0, delete_count * np.sin(epsilon_q))
        )
        observed_one_shot_drift = float(q_row["abs_dE_one_shot_vs_original"])
        if observed_one_shot_drift > path_observable_bound + 1e-9:
            raise AssertionError(
                "One-shot TFIM energy drift exceeds the realized-path observable bound."
            )
        if observed_one_shot_drift > calibration_observable_bound + 1e-9:
            raise AssertionError(
                "One-shot TFIM energy drift exceeds the calibration L*sin(epsilon) bound."
            )
        q_row.update(
            {
                "selection_time_ms": qiprune_time_ms,
                "dq_max_over_replaced": float(max(qiprune.q_angle_max_by_position.values(), default=0.0)),
                "physical_angle_max_over_replaced": float(max(qiprune.physical_angle_max_by_position.values(), default=0.0)),
                "realized_path_trace_sum_mean": float(qiprune.path_trace_sum_mean),
                "realized_path_trace_sum_max": float(qiprune.path_trace_sum_max),
                "actual_trace_distance_mean": float(qiprune.actual_trace_distance_mean),
                "actual_trace_distance_max": float(qiprune.actual_trace_distance_max),
                "hamiltonian_operator_norm": hamiltonian_op_norm,
                "one_shot_observable_drift": observed_one_shot_drift,
                "realized_path_observable_bound": path_observable_bound,
                "calibration_L_epsilon_observable_bound": calibration_observable_bound,
                "observable_bound_violations": 0,
                "physical_threshold_violations": int(qiprune.physical_threshold_violations),
                "rejected_by_path_budget": len(qiprune.rejected_by_path_budget),
            }
        )
        g_row = _evaluate(
            method="gradient_integrated_energy_matched_k",
            compact_circuit=gradient_circuit,
            deleted_pos_ids=gradient_deleted,
            baseline_circuit=baseline_circuit,
            baseline_state=baseline_state,
            control_state=control_state,
            energy_original=energy_original,
            energy_control=energy_control,
            energy_initial=float(baseline_history[0]),
            n_qubits=args.n_qubits,
            J=args.J,
            h=args.h,
            finetune_steps=args.finetune_steps,
            finetune_lr=args.finetune_lr,
            device=args.device,
        )
        g_row.update(common)
        g_row.update(
            {
                "selection_time_ms": gradient_time_ms,
                "gradient_quadrature_points": args.gradient_points,
                "gradient_score_min": float(gradient_scores.min()),
                "gradient_score_max": float(gradient_scores.max()),
                "gradient_endpoint_max": float(endpoint_scores.max()),
            }
        )
        rows.extend([q_row, g_row])
        additional_baselines = {
            "parameter_magnitude_matched_l": (
                magnitude_ranking,
                magnitude_time_ms,
            ),
            "random_matched_l": (random_ranking, random_time_ms),
            "single_gate_energy_probe_matched_l": (probe_ranking, probe_time_ms),
            "diagonal_qfi_quadratic_saliency_matched_l": (qfi_ranking, qfi_time_ms),
        }
        for method, (ranking, selection_time_ms) in additional_baselines.items():
            deleted = sorted(ranking[:delete_count])
            row = _evaluate(
                method=method,
                compact_circuit=_without_positions(baseline_circuit, deleted),
                deleted_pos_ids=deleted,
                baseline_circuit=baseline_circuit,
                baseline_state=baseline_state,
                control_state=control_state,
                energy_original=energy_original,
                energy_control=energy_control,
                energy_initial=float(baseline_history[0]),
                n_qubits=args.n_qubits,
                J=args.J,
                h=args.h,
                finetune_steps=args.finetune_steps,
                finetune_lr=args.finetune_lr,
                device=args.device,
            )
            row.update(common)
            row["selection_time_ms"] = float(selection_time_ms)
            rows.append(row)
        print(
            f"[OK] seed={args.seed} delta={delta:g} K={delete_count} "
            f"q_gap={q_row['energy_gap_finetuned_vs_control']:.6f} "
            f"grad_gap={g_row['energy_gap_finetuned_vs_control']:.6f}",
            flush=True,
        )

    payload = {
        "meta": {
            "experiment_family": "supplementary_upgrade",
            "protocol": "converged_tfim_iterative_hybrid_prefix_identity_deletion_matched_L_same_finetune_budget",
            "legacy_notebook_note": "Original gradient baseline used np.random.rand as simulated gradients; this run uses real VQE energy gate-mask gradients.",
            "gradient_score": "integral_0^1 abs(dE/dm_i) dm_i with other gates fixed",
            "seed": args.seed,
            "deltas": args.deltas,
            "methods": [
                "qiprune_iterative_hybrid_path",
                "gradient_integrated_energy_matched_k",
                "parameter_magnitude_matched_l",
                "random_matched_l",
                "single_gate_energy_probe_matched_l",
                "diagonal_qfi_quadratic_saliency_matched_l",
            ],
            "expected_rows": len(args.deltas) * 6,
            "vqe_steps_run": int(optimization.steps_run),
            "vqe_converged": bool(optimization.converged),
            "vqe_final_window_range": float(optimization.final_window_range),
            "baseline_energy_history": [float(value) for value in baseline_history],
        },
        "gradient_audit": {
            "rot_pos_ids": rot_pos_ids,
            "integrated_gradient_scores": gradient_scores.tolist(),
            "endpoint_gradient_scores": endpoint_scores.tolist(),
            "selection_time_ms": gradient_time_ms,
        },
        "baseline_selection_audit": {
            "magnitude_scores": {str(key): value for key, value in magnitude_scores.items()},
            "random_ranking": random_ranking,
            "probe_scores": {str(key): value for key, value in probe_scores.items()},
            "qfi_quadratic_scores": {str(key): value for key, value in qfi_scores.items()},
            "qfi_diagonal_by_position": {str(key): value for key, value in qfi_diagonals.items()},
        },
        "results": rows,
    }
    _atomic_new_json(args.output_json, payload)
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
