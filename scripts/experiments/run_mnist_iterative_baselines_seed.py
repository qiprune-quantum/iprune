from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

import numpy as np
from pennylane import numpy as pnp

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs
from iprune.experiments.qfi_gate_pruning import diagonal_qfi_gate_saliency_scores
from iprune.experiments.qnn_training import (
    _features,
    _make_feature_qnode,
    _rot_params,
    evaluate_qnn,
    stratified_calibration_test_split,
    stratified_subsample,
    train_qnn_circuit,
    train_qnn_circuit_with_diagnostics,
)
from iprune.experiments.tasks_classification import ToyVisionConfig, build_mnist49_splits
from iprune.gates import GateOp, build_hea_gates_from_parameters
from iprune.pruning_true import audit_fixed_identity_deletion_path, q_iPrune_iterative_identity_audited
from scripts.legacy.sweep_all_v2 import _circuit_signature, _logical_resource_metrics


def _atomic_new_json(path: Path, payload: Dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _rotation_positions(circuit: Sequence[GateOp]) -> List[int]:
    return [int(gate.pos_id) for gate in circuit if gate.gate_type == "Rot"]


def _without_positions(circuit: Sequence[GateOp], positions: Sequence[int]) -> List[GateOp]:
    deleted = {int(position) for position in positions}
    return [gate for gate in circuit if int(gate.pos_id) not in deleted]


def _circular_parameter_magnitude(gate: GateOp) -> float:
    if gate.params is None:
        raise ValueError("Magnitude baseline requires parameterized rotations.")
    return float(np.linalg.norm(np.angle(np.exp(1j * np.asarray(gate.params)))))


def _classification_loss(
    *,
    circuit: Sequence[GateOp],
    states: np.ndarray,
    labels: np.ndarray,
    readout_w: np.ndarray,
    readout_b: float,
    n_qubits: int,
    device: str,
) -> float:
    qnode = _make_feature_qnode(circuit, n_qubits, device)
    features = np.asarray(
        _features(qnode, np.asarray(states, dtype=np.complex128), pnp.array(_rot_params(circuit))),
        dtype=np.float64,
    )
    logits = features @ np.asarray(readout_w, dtype=np.float64) + float(readout_b)
    labels = np.asarray(labels, dtype=np.float64)
    return float(
        -np.mean(labels * np.logaddexp(0.0, -logits) + (1.0 - labels) * np.logaddexp(0.0, logits))
    )


def _single_gate_loss_probe_ranking(
    *,
    baseline_circuit: Sequence[GateOp],
    calibration_states: np.ndarray,
    calibration_labels: np.ndarray,
    readout_w: np.ndarray,
    readout_b: float,
    n_qubits: int,
    device: str,
) -> tuple[List[int], Dict[int, float], float]:
    reference = _classification_loss(
        circuit=baseline_circuit,
        states=calibration_states,
        labels=calibration_labels,
        readout_w=readout_w,
        readout_b=readout_b,
        n_qubits=n_qubits,
        device=device,
    )
    started = perf_counter()
    scores: Dict[int, float] = {}
    for position in _rotation_positions(baseline_circuit):
        scores[position] = abs(
            _classification_loss(
                circuit=_without_positions(baseline_circuit, [position]),
                states=calibration_states,
                labels=calibration_labels,
                readout_w=readout_w,
                readout_b=readout_b,
                n_qubits=n_qubits,
                device=device,
            )
            - reference
        )
    return sorted(scores, key=lambda position: (scores[position], position)), scores, (perf_counter() - started) * 1000.0


def _evaluate(
    *,
    method: str,
    circuit: Sequence[GateOp],
    deleted_pos_ids: Sequence[int],
    baseline_circuit: Sequence[GateOp],
    train_states: np.ndarray,
    train_labels: np.ndarray,
    test_states: np.ndarray,
    test_labels: np.ndarray,
    readout_w: np.ndarray,
    readout_b: float,
    control_accuracy: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    compact = list(circuit)
    one_shot_accuracy = evaluate_qnn(
        circuit=compact,
        states=test_states,
        labels=test_labels,
        readout_w=readout_w,
        readout_b=readout_b,
        n_qubits=args.n_qubits,
        device=args.device,
    )
    tuned = train_qnn_circuit(
        circuit=compact,
        train_states=train_states,
        train_labels=train_labels,
        n_qubits=args.n_qubits,
        steps=args.finetune_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed + 200,
        device=args.device,
        initial_readout=(readout_w, readout_b),
    )
    tuned_accuracy = evaluate_qnn(
        circuit=tuned.circuit,
        states=test_states,
        labels=test_labels,
        readout_w=tuned.readout_w,
        readout_b=tuned.readout_b,
        n_qubits=args.n_qubits,
        device=args.device,
    )
    before = _logical_resource_metrics(list(baseline_circuit), args.n_qubits)
    after = _logical_resource_metrics(compact, args.n_qubits)
    return {
        "method": method,
        "deleted_pos_ids": sorted(int(position) for position in deleted_pos_ids),
        "pruned_positions": len(deleted_pos_ids),
        "gate_count_before": len(baseline_circuit),
        "gate_count_after": len(compact),
        "actual_gate_prune_ratio": len(deleted_pos_ids) / max(1, len(baseline_circuit)),
        "logical_depth_before": before["logical_depth"],
        "logical_depth_after": after["logical_depth"],
        "accuracy_one_shot": float(one_shot_accuracy),
        "accuracy_finetuned": float(tuned_accuracy),
        "accuracy_drop_one_shot_vs_control": float(control_accuracy - one_shot_accuracy),
        "accuracy_drop_finetuned_vs_control": float(control_accuracy - tuned_accuracy),
        "pruned_circuit_signature": _circuit_signature(compact),
        "finetuned_circuit_signature": _circuit_signature(tuned.circuit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converged MNIST49 dynamic identity deletion with matched-realized-L baselines."
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.3, 0.5])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--qnn_depth", type=int, default=3)
    parser.add_argument("--n_train", type=int, default=512)
    parser.add_argument("--n_monitor", type=int, default=128)
    parser.add_argument("--n_calibration", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=400)
    parser.add_argument("--min_train_steps", type=int, default=100)
    parser.add_argument("--monitor_interval", type=int, default=10)
    parser.add_argument("--monitor_window", type=int, default=5)
    parser.add_argument("--convergence_tol", type=float, default=2e-3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--qbeta", type=float, default=0.0)
    args = parser.parse_args()

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
        train_pool,
        train_pool_labels,
        n_calibration=args.n_train,
        n_test=args.n_monitor,
        seed=args.seed + 10,
    )
    calibration_states, calibration_labels, test_states, test_labels = stratified_calibration_test_split(
        validation_pool,
        validation_labels,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        seed=args.seed + 20,
    )
    pruning_states, pruning_labels = stratified_subsample(
        calibration_states, calibration_labels, args.M, args.seed + 30
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
    trained_params = np.asarray([gate.params for gate in trained.circuit if gate.gate_type == "Rot"], dtype=np.float64).reshape(args.qnn_depth, args.n_qubits, 3)
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
    rot_by_position = {int(gate.pos_id): gate for gate in baseline_circuit if gate.gate_type == "Rot"}
    magnitude_started = perf_counter()
    magnitude_ranking = sorted(rot_by_position, key=lambda position: (_circular_parameter_magnitude(rot_by_position[position]), position))
    magnitude_time_ms = (perf_counter() - magnitude_started) * 1000.0
    random_started = perf_counter()
    random_ranking = [int(position) for position in np.random.default_rng(args.seed + 20260911).permutation(_rotation_positions(baseline_circuit))]
    random_time_ms = (perf_counter() - random_started) * 1000.0
    probe_ranking, probe_scores, probe_time_ms = _single_gate_loss_probe_ranking(
        baseline_circuit=baseline_circuit,
        calibration_states=pruning_states,
        calibration_labels=pruning_labels,
        readout_w=trained.readout_w,
        readout_b=trained.readout_b,
        n_qubits=args.n_qubits,
        device=args.device,
    )
    qfi_ranking, qfi_scores, qfi_diagonals, qfi_time_ms = (
        diagonal_qfi_gate_saliency_scores(
            circuit=baseline_circuit,
            task_states=pruning_states,
            n_qubits=args.n_qubits,
            device=args.device,
        )
    )

    rows: List[Dict[str, Any]] = []
    for delta in args.deltas:
        cfg = PaperProtocolConfig(
            device=args.device,
            n_qubits=args.n_qubits,
            depth=args.qnn_depth,
            M=args.M,
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
        deleted = qiprune.accepted_positions
        heldout = audit_fixed_identity_deletion_path(
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            selected_positions=deleted,
            n_qubits=args.n_qubits,
            device_name=args.device,
            task_states=test_states,
            Gq_diag=Gq_diag,
            physical_epsilon=epsilon_q,
        )
        common = {
            "experiment": "mnist49_iterative_hybrid_path_matched_L",
            "seed": args.seed,
            "delta": float(delta),
            "q": float(q),
            "lambda": float(lam),
            "epsilon_q": float(epsilon_q),
            "matched_delete_count": len(deleted),
            "baseline_accuracy": float(baseline_accuracy),
            "control_accuracy": float(control_accuracy),
            "qnn_steps_run": int(trained.steps_run),
            "qnn_converged": bool(trained.converged),
            "qnn_final_monitor_window_range": float(trained.final_monitor_window_range),
        }
        q_row = _evaluate(
            method="qiprune_iterative_hybrid_path",
            circuit=qiprune.compact_circuit,
            deleted_pos_ids=deleted,
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
        q_row.update(common)
        q_row.update({
            "calibration_path_trace_sum_max": float(qiprune.path_trace_sum_max),
            "calibration_actual_trace_distance_max": float(qiprune.actual_trace_distance_max),
            "calibration_physical_threshold_violations": int(qiprune.physical_threshold_violations),
            "heldout_path_trace_sum_max": float(heldout.path_trace_sum_max),
            "heldout_actual_trace_distance_max": float(heldout.actual_trace_distance_max),
            "heldout_physical_gate_violations": int(heldout.physical_threshold_violations),
            "heldout_physical_state_position_violations": int(heldout.physical_state_position_violations),
            "heldout_path_bound_violations": int(heldout.path_bound_violations),
        })
        rows.append(q_row)
        for method, ranking, selection_time_ms in [
            ("parameter_magnitude_matched_l", magnitude_ranking, magnitude_time_ms),
            ("random_matched_l", random_ranking, random_time_ms),
            ("single_gate_loss_probe_matched_l", probe_ranking, probe_time_ms),
            ("diagonal_qfi_quadratic_saliency_matched_l", qfi_ranking, qfi_time_ms),
        ]:
            baseline_deleted = sorted(ranking[:len(deleted)])
            row = _evaluate(
                method=method,
                circuit=_without_positions(baseline_circuit, baseline_deleted),
                deleted_pos_ids=baseline_deleted,
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
            row.update(common)
            row["selection_time_ms"] = float(selection_time_ms)
            rows.append(row)
        print(f"[OK] seed={args.seed} delta={delta:g} L={len(deleted)} q_acc={q_row['accuracy_finetuned']:.4f}", flush=True)

    _atomic_new_json(
        args.output_json,
        {
            "meta": {
                "protocol": "converged_training_disjoint_monitor_calibration_test_dynamic_identity_deletion_matched_realized_L",
                "seed": args.seed,
                "deltas": args.deltas,
                "expected_rows": len(args.deltas) * 5,
                "probe_score_count": len(probe_scores),
                "qfi_score_count": len(qfi_scores),
                "qfi_diagonal_by_position": qfi_diagonals,
            },
            "results": rows,
        },
    )
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
