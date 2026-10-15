"""Calibration-ensemble size sweep under the frozen dynamic-deletion contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry
from iprune.experiments.qnn_training import (
    evaluate_qnn,
    stratified_calibration_test_split,
    stratified_subsample,
    train_qnn_circuit,
    train_qnn_circuit_with_diagnostics,
)
from iprune.experiments.tasks_classification import ToyVisionConfig, build_mnist49_splits
from iprune.gates import build_hea_gates_from_parameters
from iprune.pruning_true import audit_fixed_identity_deletion_path, q_iPrune_iterative_identity_audited
from scripts.experiments.run_mnist_iterative_baselines_seed import _atomic_new_json, _evaluate


def _jaccard(left: Set[int], right: Set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MNIST49 calibration-size sweep for dynamic hybrid-prefix identity deletion."
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--calibration_sizes", nargs="+", type=int, default=[10, 30, 50, 100])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--qnn_depth", type=int, default=3)
    parser.add_argument("--n_train", type=int, default=256)
    parser.add_argument("--n_monitor", type=int, default=64)
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--n_test", type=int, default=256)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--min_train_steps", type=int, default=300)
    parser.add_argument("--monitor_interval", type=int, default=10)
    parser.add_argument("--monitor_window", type=int, default=5)
    parser.add_argument("--convergence_tol", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--finetune_steps", type=int, default=20)
    args = parser.parse_args()
    sizes = sorted(set(int(value) for value in args.calibration_sizes))
    if not sizes or sizes[0] < 1 or sizes[-1] > args.n_calibration:
        raise ValueError("calibration_sizes must be in [1, n_calibration].")

    task_config = ToyVisionConfig(
        n_qubits=args.n_qubits,
        data_dir=str(args.data_dir),
        seed=args.seed,
        download=args.allow_download,
        n_train_pool=max(args.n_train + args.n_monitor, 512),
        n_val_pool=args.n_calibration + args.n_test,
    )
    train_pool, train_pool_labels, validation_pool, validation_labels = build_mnist49_splits(task_config)
    train_states, train_labels, monitor_states, monitor_labels = stratified_calibration_test_split(
        train_pool, train_pool_labels, n_calibration=args.n_train, n_test=args.n_monitor, seed=args.seed + 10
    )
    calibration_states, calibration_labels, test_states, test_labels = stratified_calibration_test_split(
        validation_pool, validation_labels, n_calibration=args.n_calibration, n_test=args.n_test, seed=args.seed + 20
    )
    rng = np.random.default_rng(args.seed)
    initial_params = rng.normal(scale=0.15, size=(args.qnn_depth, args.n_qubits, 3))
    _, initial_circuit = build_hea_gates_from_parameters(
        initial_params, include_cnot=True, include_identity_rot=False
    )
    trained = train_qnn_circuit_with_diagnostics(
        circuit=initial_circuit,
        train_states=train_states,
        train_labels=train_labels,
        monitor_states=monitor_states,
        monitor_labels=monitor_labels,
        n_qubits=args.n_qubits,
        max_steps=args.max_train_steps,
        min_steps=args.min_train_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed + 100,
        monitor_interval=args.monitor_interval,
        monitor_window=args.monitor_window,
        convergence_tol=args.convergence_tol,
        device=args.device,
    )
    if not trained.converged:
        raise RuntimeError(
            "QNN did not satisfy the fixed-monitor convergence rule: "
            f"{trained.steps_run}/{args.max_train_steps} steps, range="
            f"{trained.final_monitor_window_range:.6g}."
        )
    trained_params = np.asarray(
        [gate.params for gate in trained.circuit if gate.gate_type == "Rot"], dtype=np.float64
    ).reshape(args.qnn_depth, args.n_qubits, 3)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        trained_params, include_cnot=True, include_identity_rot=True
    )
    baseline_accuracy = evaluate_qnn(
        circuit=baseline_circuit,
        states=test_states,
        labels=test_labels,
        readout_w=trained.readout_w,
        readout_b=trained.readout_b,
        n_qubits=args.n_qubits,
        device=args.device,
    )
    control = train_qnn_circuit(
        circuit=baseline_circuit,
        train_states=train_states,
        train_labels=train_labels,
        n_qubits=args.n_qubits,
        steps=args.finetune_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed + 200,
        device=args.device,
        initial_readout=(trained.readout_w, trained.readout_b),
    )
    control_accuracy = evaluate_qnn(
        circuit=control.circuit,
        states=test_states,
        labels=test_labels,
        readout_w=control.readout_w,
        readout_b=control.readout_b,
        n_qubits=args.n_qubits,
        device=args.device,
    )

    rows: List[Dict[str, Any]] = []
    deleted_sets: Dict[Tuple[int, float], Set[int]] = {}
    for M in sizes:
        pruning_states, _ = stratified_subsample(
            calibration_states, calibration_labels, M, args.seed + 30
        )
        for delta in args.deltas:
            cfg = PaperProtocolConfig(
                device=args.device,
                n_qubits=args.n_qubits,
                depth=args.qnn_depth,
                M=M,
                seed=args.seed,
                gamma=0.0,
                alpha=0.6,
                qbeta=0.0,
                delta=delta,
                rot_pool_k=1,
                rot_pool_sigma=0.0,
                include_cnot=True,
            )
            _, q, Gq_diag, _, epsilon_q = make_q_geometry(cfg)
            started = perf_counter()
            qiprune = q_iPrune_iterative_identity_audited(
                all_gates=all_gates,
                baseline_circuit=baseline_circuit,
                n_qubits=args.n_qubits,
                device_name=args.device,
                task_states=pruning_states,
                Gq_diag=Gq_diag,
                q_epsilon=epsilon_q,
                physical_epsilon=epsilon_q,
            )
            heldout = audit_fixed_identity_deletion_path(
                all_gates=all_gates,
                baseline_circuit=baseline_circuit,
                selected_positions=qiprune.accepted_positions,
                n_qubits=args.n_qubits,
                device_name=args.device,
                task_states=test_states,
                Gq_diag=Gq_diag,
                physical_epsilon=epsilon_q,
            )
            row = _evaluate(
                method="qiprune_iterative_hybrid_path",
                circuit=qiprune.compact_circuit,
                deleted_pos_ids=qiprune.accepted_positions,
                baseline_circuit=baseline_circuit,
                train_states=train_states,
                train_labels=train_labels,
                test_states=test_states,
                test_labels=test_labels,
                readout_w=trained.readout_w,
                readout_b=trained.readout_b,
                control_accuracy=control_accuracy,
                args=args,
            )
            row.update(
                {
                    "experiment": "mnist49_calibration_size_dynamic_identity_deletion",
                    "seed": args.seed,
                    "calibration_ensemble_size": M,
                    "delta": float(delta),
                    "q": float(q),
                    "epsilon_q": float(epsilon_q),
                    "baseline_accuracy": float(baseline_accuracy),
                    "control_accuracy": float(control_accuracy),
                    "qnn_steps_run": int(trained.steps_run),
                    "qnn_converged": bool(trained.converged),
                    "qnn_final_monitor_window_range": float(trained.final_monitor_window_range),
                    "selection_time_ms": (perf_counter() - started) * 1000.0,
                    "calibration_path_trace_sum_max": float(qiprune.path_trace_sum_max),
                    "calibration_actual_trace_distance_max": float(qiprune.actual_trace_distance_max),
                    "calibration_physical_threshold_violations": int(qiprune.physical_threshold_violations),
                    "heldout_path_trace_sum_max": float(heldout.path_trace_sum_max),
                    "heldout_actual_trace_distance_max": float(heldout.actual_trace_distance_max),
                    "heldout_physical_gate_violations": int(heldout.physical_threshold_violations),
                    "heldout_physical_state_position_violations": int(heldout.physical_state_position_violations),
                    "heldout_path_bound_violations": int(heldout.path_bound_violations),
                }
            )
            rows.append(row)
            deleted_sets[(M, float(delta))] = set(map(int, qiprune.accepted_positions))
            print(f"[OK] seed={args.seed} M={M} delta={delta:g} L={len(qiprune.accepted_positions)}", flush=True)

    reference_size = 50 if 50 in sizes else sizes[len(sizes) // 2]
    for row in rows:
        M, delta = int(row["calibration_ensemble_size"]), float(row["delta"])
        row["jaccard_to_reference_size"] = _jaccard(
            deleted_sets[(M, delta)], deleted_sets[(reference_size, delta)]
        )
        row["reference_calibration_ensemble_size"] = reference_size
    _atomic_new_json(
        args.output_json,
        {
            "meta": {
                "protocol": "converged_mnist49_disjoint_sets_dynamic_identity_deletion_calibration_size_sweep",
                "seed": args.seed,
                "calibration_sizes": sizes,
                "deltas": args.deltas,
                "reference_calibration_ensemble_size": reference_size,
                "expected_rows": len(sizes) * len(args.deltas),
            },
            "results": rows,
        },
    )
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
