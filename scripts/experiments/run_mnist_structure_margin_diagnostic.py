"""Diagnose why the q=1 geometry behaves differently on MNIST49 than on TFIM.

This is an auxiliary, pre-specified diagnostic.  It does not change the main
pruning protocol: q-iPrune still uses the maximum physical Fubini--Study angle
on its calibration ensemble.  For each selected deletion path, the script
checks whether classification loss is concentrated near the learned decision
boundary and verifies the state-distance-to-readout bound

    |z(x)-z'(x)| <= 2 ||sum_i w_i Z_i|| D_tr(rho_x, rho'_x).

The learned observable has norm sum_i |w_i| because the Z_i commute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from pennylane import numpy as pnp

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs
from iprune.experiments.qnn_training import (
    _features,
    _make_feature_qnode,
    _rot_params,
    evaluate_qnn,
    stratified_calibration_test_split,
    stratified_subsample,
    train_qnn_circuit_with_diagnostics,
)
from iprune.experiments.tasks_classification import ToyVisionConfig, build_mnist49_splits
from iprune.gates import GateOp, build_hea_gates_from_parameters
from iprune.pruning_true import audit_fixed_identity_deletion_path, q_iPrune_iterative_identity_audited


def _atomic_new_json(path: Path, payload: Dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _logits(
    circuit: Sequence[GateOp], states: np.ndarray, readout_w: np.ndarray, readout_b: float, n_qubits: int, device: str
) -> np.ndarray:
    qnode = _make_feature_qnode(circuit, n_qubits, device)
    features = np.asarray(
        _features(qnode, np.asarray(states, dtype=np.complex128), pnp.array(_rot_params(circuit))),
        dtype=np.float64,
    )
    return features @ np.asarray(readout_w, dtype=np.float64) + float(readout_b)


def _per_sample_trace_distance(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    overlaps = np.abs(np.sum(np.conj(reference) * candidate, axis=1))
    return np.sqrt(np.maximum(0.0, 1.0 - np.clip(overlaps, 0.0, 1.0) ** 2))


def _mean_or_nan(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _margin_diagnostic(
    *,
    baseline_circuit: Sequence[GateOp],
    pruned_circuit: Sequence[GateOp],
    states: np.ndarray,
    labels: np.ndarray,
    readout_w: np.ndarray,
    readout_b: float,
    n_qubits: int,
    device: str,
) -> Dict[str, Any]:
    baseline_logits = _logits(baseline_circuit, states, readout_w, readout_b, n_qubits, device)
    pruned_logits = _logits(pruned_circuit, states, readout_w, readout_b, n_qubits, device)
    baseline_states = simulate_outputs(baseline_circuit, states, n_qubits, device)
    pruned_states = simulate_outputs(pruned_circuit, states, n_qubits, device)
    trace_distance = _per_sample_trace_distance(baseline_states, pruned_states)
    logit_drift = np.abs(pruned_logits - baseline_logits)
    margin = np.abs(baseline_logits)
    baseline_prediction = (baseline_logits >= 0.0).astype(int)
    pruned_prediction = (pruned_logits >= 0.0).astype(int)
    labels = np.asarray(labels, dtype=int)
    flip = baseline_prediction != pruned_prediction

    # The readout is O_w=sum_i w_i Z_i; its exact operator norm is sum_i |w_i|.
    observable_norm = float(np.sum(np.abs(readout_w)))
    readout_bound = 2.0 * observable_norm * trace_distance
    nonzero_bound = readout_bound > 1e-12
    ratios = logit_drift[nonzero_bound] / readout_bound[nonzero_bound]
    if np.any(~nonzero_bound & (logit_drift > 1e-10)):
        raise AssertionError("Nonzero logit drift with a zero trace-distance readout bound.")
    if np.any(logit_drift > readout_bound + 1e-8):
        raise AssertionError("Observed logit drift exceeded the trace-distance readout bound.")

    ordered = np.argsort(margin, kind="stable")
    margin_quartiles = []
    for quartile, indices in enumerate(np.array_split(ordered, 4), start=1):
        margin_quartiles.append(
            {
                "quartile": quartile,
                "n": int(len(indices)),
                "margin_min": float(np.min(margin[indices])),
                "margin_max": float(np.max(margin[indices])),
                "mean_margin": float(np.mean(margin[indices])),
                "mean_trace_distance": float(np.mean(trace_distance[indices])),
                "mean_abs_logit_drift": float(np.mean(logit_drift[indices])),
                "prediction_flip_rate": float(np.mean(flip[indices])),
                "accuracy_baseline": float(np.mean(baseline_prediction[indices] == labels[indices])),
                "accuracy_pruned": float(np.mean(pruned_prediction[indices] == labels[indices])),
            }
        )

    corr = None
    if np.std(trace_distance) > 1e-12 and np.std(logit_drift) > 1e-12:
        corr = float(np.corrcoef(trace_distance, logit_drift)[0, 1])
    return {
        "n_test": int(len(labels)),
        "baseline_accuracy": float(np.mean(baseline_prediction == labels)),
        "pruned_accuracy": float(np.mean(pruned_prediction == labels)),
        "accuracy_drop": float(np.mean(baseline_prediction == labels) - np.mean(pruned_prediction == labels)),
        "prediction_flip_count": int(np.sum(flip)),
        "prediction_flip_rate": float(np.mean(flip)),
        "mean_baseline_margin": float(np.mean(margin)),
        "mean_baseline_margin_flipped": _mean_or_nan(margin[flip]),
        "mean_baseline_margin_not_flipped": _mean_or_nan(margin[~flip]),
        "mean_trace_distance": float(np.mean(trace_distance)),
        "max_trace_distance": float(np.max(trace_distance)),
        "mean_abs_logit_drift": float(np.mean(logit_drift)),
        "max_abs_logit_drift": float(np.max(logit_drift)),
        "descriptive_pearson_trace_vs_logit_drift": corr,
        "readout_observable_norm_sum_abs_w": observable_norm,
        "readout_bound_violation_count": 0,
        "max_logit_drift_over_readout_bound": float(np.max(ratios)) if len(ratios) else 0.0,
        "margin_quartiles": margin_quartiles,
    }


def _reference_positions(reference: Path, delta: float) -> list[int]:
    payload = json.loads(reference.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["results"]
        if row["method"] == "qiprune_iterative_hybrid_path" and abs(float(row["delta"]) - delta) < 1e-12
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one q-iPrune row for delta={delta} in {reference}")
    return [int(position) for position in rows[0]["deleted_pos_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST49 geometry/readout-margin diagnostic for dynamic q=1 deletion.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--reference_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--qnn_depth", type=int, default=3)
    # These defaults intentionally match the already frozen three-seed pilot.
    parser.add_argument("--n_train", type=int, default=256)
    parser.add_argument("--n_monitor", type=int, default=64)
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--n_test", type=int, default=256)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--min_train_steps", type=int, default=300)
    parser.add_argument("--monitor_interval", type=int, default=10)
    parser.add_argument("--monitor_window", type=int, default=5)
    parser.add_argument("--convergence_tol", type=float, default=2e-2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    task_config = ToyVisionConfig(
        n_qubits=args.n_qubits, data_dir=str(args.data_dir), seed=args.seed, download=False,
        n_train_pool=max(args.n_train + args.n_monitor, 512), n_val_pool=args.n_calibration + args.n_test,
    )
    train_pool, train_pool_labels, validation_pool, validation_labels = build_mnist49_splits(task_config)
    train_states, train_labels, monitor_states, monitor_labels = stratified_calibration_test_split(
        train_pool, train_pool_labels, n_calibration=args.n_train, n_test=args.n_monitor, seed=args.seed + 10
    )
    calibration_states, calibration_labels, test_states, test_labels = stratified_calibration_test_split(
        validation_pool, validation_labels, n_calibration=args.n_calibration, n_test=args.n_test, seed=args.seed + 20
    )
    pruning_states, pruning_labels = stratified_subsample(calibration_states, calibration_labels, args.M, args.seed + 30)
    rng = np.random.default_rng(args.seed)
    initial_params = rng.normal(scale=0.15, size=(args.qnn_depth, args.n_qubits, 3))
    _, initial_circuit = build_hea_gates_from_parameters(initial_params, include_cnot=True, include_identity_rot=False)
    trained = train_qnn_circuit_with_diagnostics(
        circuit=initial_circuit, train_states=train_states, train_labels=train_labels,
        monitor_states=monitor_states, monitor_labels=monitor_labels, n_qubits=args.n_qubits,
        max_steps=args.max_train_steps, min_steps=args.min_train_steps, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed + 100, monitor_interval=args.monitor_interval, monitor_window=args.monitor_window,
        convergence_tol=args.convergence_tol, device=args.device,
    )
    if not trained.converged:
        raise RuntimeError("QNN did not satisfy the fixed-monitor convergence rule.")
    trained_params = np.asarray([gate.params for gate in trained.circuit if gate.gate_type == "Rot"], dtype=np.float64).reshape(args.qnn_depth, args.n_qubits, 3)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(trained_params, include_cnot=True, include_identity_rot=True)

    rows = []
    for delta in args.deltas:
        cfg = PaperProtocolConfig(device=args.device, n_qubits=args.n_qubits, depth=args.qnn_depth, M=args.M, seed=args.seed,
                                  gamma=0.0, alpha=0.6, qbeta=0.0, delta=delta, rot_pool_k=1, rot_pool_sigma=0.0, include_cnot=True)
        lam, q, Gq_diag, _, epsilon_q = make_q_geometry(cfg)
        result = q_iPrune_iterative_identity_audited(
            all_gates=all_gates, baseline_circuit=baseline_circuit, n_qubits=args.n_qubits, device_name=args.device,
            task_states=pruning_states, Gq_diag=Gq_diag, q_epsilon=epsilon_q, physical_epsilon=epsilon_q,
        )
        expected_positions = _reference_positions(args.reference_json, delta)
        if result.accepted_positions != expected_positions:
            raise AssertionError(
                f"Reproduction mismatch at delta={delta}: got {result.accepted_positions}, expected {expected_positions}."
            )
        heldout = audit_fixed_identity_deletion_path(
            all_gates=all_gates, baseline_circuit=baseline_circuit, selected_positions=result.accepted_positions,
            n_qubits=args.n_qubits, device_name=args.device, task_states=test_states, Gq_diag=Gq_diag, physical_epsilon=epsilon_q,
        )
        diagnostic = _margin_diagnostic(
            baseline_circuit=baseline_circuit, pruned_circuit=result.compact_circuit, states=test_states, labels=test_labels,
            readout_w=trained.readout_w, readout_b=trained.readout_b, n_qubits=args.n_qubits, device=args.device,
        )
        rows.append({
            "seed": args.seed, "delta": float(delta), "q": float(q), "lambda": float(lam), "epsilon_q": float(epsilon_q),
            "accepted_positions": result.accepted_positions, "L": len(result.accepted_positions),
            "qnn_steps_run": int(trained.steps_run), "qnn_final_monitor_window_range": float(trained.final_monitor_window_range),
            "heldout_path_trace_sum_max": float(heldout.path_trace_sum_max),
            "heldout_actual_trace_distance_max": float(heldout.actual_trace_distance_max),
            "heldout_path_bound_violations": int(heldout.path_bound_violations),
            "diagnostic": diagnostic,
        })
        print(f"[OK] seed={args.seed} delta={delta:g} L={len(result.accepted_positions)} flip={diagnostic['prediction_flip_rate']:.4f}", flush=True)
    _atomic_new_json(args.output_json, {"meta": {"protocol": "auxiliary_margin_diagnostic_reproduces_frozen_mnist_q1_pilot", "reference_json": str(args.reference_json.resolve())}, "results": rows})
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
