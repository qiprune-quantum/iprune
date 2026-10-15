from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry
from iprune.experiments.qnn_training import (
    evaluate_qnn,
    stratified_calibration_test_split,
    stratified_subsample,
    train_qnn_circuit,
)
from iprune.experiments.tasks_classification import ToyVisionConfig, build_mnist49_splits
from iprune.gates import GateOp, build_hea_gates_from_parameters
from iprune.qgeometry import epsilon_from_delta
from scripts.experiments.run_q_matched_k_tfim_seed import (
    _compact_without,
    _fast_native_deleted_positions,
    _score_positions,
)
from scripts.experiments.run_q_sensitivity_selection_diagnostic import (
    _jaccard,
    _rank_correlations,
    _rank_positions,
)
from scripts.legacy.sweep_all_v2 import _circuit_signature, _logical_resource_metrics


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CHECKPOINT = (
    PROJECT_DIR
    / "results"
    / "supplementary_upgrade_20260807"
    / "trained_qnn_true_pruning_mnist49_seed42_46"
    / "trained_qnn_pruning_checkpoint.json"
)


def _atomic_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _load_source_rows(
    checkpoint: Path, seed: int, deltas: Sequence[float]
) -> Dict[float, Mapping[str, Any]]:
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    rows = {
        float(row["delta"]): row
        for row in payload.get("results", [])
        if int(row["seed"]) == seed
    }
    missing = [delta for delta in deltas if float(delta) not in rows]
    if missing:
        raise ValueError(f"Missing MNIST49 source rows for seed={seed}, deltas={missing}")
    return rows


def _rot_parameters(circuit: Sequence[GateOp], depth: int, n_qubits: int) -> np.ndarray:
    return np.asarray(
        [gate.params for gate in circuit if gate.gate_type == "Rot"],
        dtype=np.float64,
    ).reshape(depth, n_qubits, 3)


def _load_or_train_snapshot(
    *,
    args: argparse.Namespace,
    train_states: np.ndarray,
    train_labels: np.ndarray,
) -> Tuple[List[GateOp], np.ndarray, float, np.ndarray, str]:
    if args.snapshot_json:
        payload = json.loads(args.snapshot_json.resolve().read_text(encoding="utf-8"))
        snapshot = payload["trained_snapshot"]
        if int(snapshot["seed"]) != args.seed:
            raise ValueError("Snapshot seed does not match requested seed")
        params = np.asarray(snapshot["trained_parameters"], dtype=np.float64)
        _, trained_circuit = build_hea_gates_from_parameters(
            params, include_cnot=True, include_identity_rot=False
        )
        signature = _circuit_signature(trained_circuit)
        if signature != snapshot["baseline_circuit_signature"]:
            raise RuntimeError("Loaded snapshot circuit signature mismatch")
        return (
            trained_circuit,
            np.asarray(snapshot["readout_w"], dtype=np.float64),
            float(snapshot["readout_b"]),
            np.asarray(snapshot["loss_history"], dtype=np.float64),
            "loaded",
        )

    rng = np.random.default_rng(args.seed)
    initial_params = rng.normal(
        scale=0.15, size=(args.qnn_depth, args.n_qubits, 3)
    )
    _, initial_circuit = build_hea_gates_from_parameters(
        initial_params, include_cnot=True, include_identity_rot=False
    )
    trained = train_qnn_circuit(
        circuit=initial_circuit,
        train_states=train_states,
        train_labels=train_labels,
        n_qubits=args.n_qubits,
        steps=args.train_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed + 100,
        device=args.device,
    )
    return (
        trained.circuit,
        trained.readout_w,
        trained.readout_b,
        trained.loss_history,
        "deterministically_reconstructed",
    )


def _evaluate_selection(
    *,
    deleted: Sequence[int],
    baseline_circuit: Sequence[GateOp],
    trained_readout_w: np.ndarray,
    trained_readout_b: float,
    train_states: np.ndarray,
    train_labels: np.ndarray,
    test_states: np.ndarray,
    test_labels: np.ndarray,
    acc_control: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    compact = _compact_without(baseline_circuit, deleted)
    acc_one = evaluate_qnn(
        circuit=compact,
        states=test_states,
        labels=test_labels,
        readout_w=trained_readout_w,
        readout_b=trained_readout_b,
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
        initial_readout=(trained_readout_w, trained_readout_b),
    )
    acc_tuned = evaluate_qnn(
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
        "acc_pruned_one_shot": float(acc_one),
        "acc_pruned_finetuned": float(acc_tuned),
        "acc_drop_one_shot_vs_control": float(acc_control - acc_one),
        "acc_drop_finetuned_vs_control": float(acc_control - acc_tuned),
        "gate_count_before": len(baseline_circuit),
        "gate_count_after": len(compact),
        "actual_gate_prune_ratio": len(deleted) / len(baseline_circuit),
        "logical_depth_before": before["logical_depth"],
        "logical_depth_after": after["logical_depth"],
        "pruned_circuit_signature": _circuit_signature(compact),
        "finetuned_circuit_signature": _circuit_signature(tuned.circuit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-seed MNIST49 matched-K q-geometry contribution pilot."
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--source_checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--snapshot_json", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.3, 0.5])
    parser.add_argument("--qbetas", nargs="+", type=float, default=[0, 1, 3, 6, 10, 20])
    parser.add_argument("--selection_only", action="store_true")
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--qnn_depth", type=int, default=3)
    parser.add_argument("--train_steps", type=int, default=40)
    parser.add_argument("--finetune_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_train", type=int, default=512)
    parser.add_argument("--n_calibration", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    args = parser.parse_args()

    qbetas = sorted(set(float(value) for value in args.qbetas))
    if 0.0 not in qbetas or 1.0 not in qbetas:
        raise ValueError("qbetas must include 0 and 1")
    source_checkpoint = args.source_checkpoint.resolve()
    source_rows = _load_source_rows(source_checkpoint, args.seed, args.deltas)

    tcfg = ToyVisionConfig(
        n_qubits=args.n_qubits,
        data_dir=str(args.data_dir),
        seed=args.seed,
        download=args.allow_download,
        n_train_pool=max(args.n_train, 512),
        n_val_pool=args.n_calibration + args.n_test,
    )
    training_pool, training_labels, validation_pool, validation_labels = (
        build_mnist49_splits(tcfg)
    )
    train_states, train_labels = stratified_subsample(
        training_pool, training_labels, args.n_train, args.seed + 10
    )
    calibration_states, calibration_labels, test_states, test_labels = (
        stratified_calibration_test_split(
            validation_pool,
            validation_labels,
            n_calibration=args.n_calibration,
            n_test=args.n_test,
            seed=args.seed + 20,
        )
    )
    pruning_states, _ = stratified_subsample(
        calibration_states, calibration_labels, args.M, args.seed + 30
    )

    trained_circuit, readout_w, readout_b, loss_history, snapshot_origin = (
        _load_or_train_snapshot(
            args=args, train_states=train_states, train_labels=train_labels
        )
    )
    baseline_signature = _circuit_signature(trained_circuit)
    source_signatures = {
        str(source_rows[float(delta)]["baseline_circuit_signature"])
        for delta in args.deltas
    }
    if source_signatures != {baseline_signature}:
        raise RuntimeError(
            f"Reconstructed MNIST49 signature mismatch: {baseline_signature} vs {source_signatures}"
        )

    acc_trained = evaluate_qnn(
        circuit=trained_circuit,
        states=test_states,
        labels=test_labels,
        readout_w=readout_w,
        readout_b=readout_b,
        n_qubits=args.n_qubits,
        device=args.device,
    )
    reference_source = source_rows[float(args.deltas[0])]
    if not math.isclose(
        acc_trained,
        float(reference_source["acc_trained_before_extra_steps"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Reconstructed trained accuracy differs from source")

    control = None
    acc_control = None
    if not args.selection_only:
        control = train_qnn_circuit(
            circuit=trained_circuit,
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
        acc_control = evaluate_qnn(
            circuit=control.circuit,
            states=test_states,
            labels=test_labels,
            readout_w=control.readout_w,
            readout_b=control.readout_b,
            n_qubits=args.n_qubits,
            device=args.device,
        )
        if not math.isclose(
            acc_control,
            float(reference_source["acc_full_control_after_extra_steps"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Reconstructed control accuracy differs from source")

    trained_params = _rot_parameters(trained_circuit, args.qnn_depth, args.n_qubits)
    all_gates, pruning_baseline = build_hea_gates_from_parameters(
        trained_params, include_cnot=True, include_identity_rot=True
    )
    if _circuit_signature(pruning_baseline) != baseline_signature:
        raise RuntimeError("Pruning baseline differs from trained snapshot")

    q_scores, physical_scores, score_rows = _score_positions(
        all_gates=all_gates,
        baseline_circuit=pruning_baseline,
        prefix_circuit=pruning_baseline,
        initial_state=pruning_states,
        n_qubits=args.n_qubits,
        qbetas=qbetas,
        gamma=args.gamma,
        alpha=args.alpha,
    )
    q1_order = _rank_positions(q_scores[0.0])
    result_rows: List[Dict[str, Any]] = []
    evaluation_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    for delta in args.deltas:
        source = source_rows[float(delta)]
        source_target_k = int(source["pruned_positions"])
        epsilon_q = epsilon_from_delta(float(delta), 1.0)
        physical_eligible = {
            position
            for position, score in physical_scores.items()
            if score <= epsilon_q + 1e-12
        }
        matched_k = min(source_target_k, len(physical_eligible))
        q1_selected = [
            position for position in q1_order if position in physical_eligible
        ][:matched_k]

        for qbeta in qbetas:
            cfg = PaperProtocolConfig(
                device=args.device,
                n_qubits=args.n_qubits,
                depth=args.qnn_depth,
                M=args.M,
                seed=args.seed,
                gamma=args.gamma,
                alpha=args.alpha,
                qbeta=qbeta,
                delta=float(delta),
                rot_pool_k=1,
                rot_pool_sigma=0.0,
                include_cnot=True,
            )
            lam, q, Gq_diag, Mq, _ = make_q_geometry(cfg)
            feasible_order = [
                position
                for position in _rank_positions(q_scores[qbeta])
                if position in physical_eligible
            ]
            deleted = sorted(feasible_order[:matched_k])
            native_deleted = _fast_native_deleted_positions(
                all_gates=all_gates,
                baseline_circuit=pruning_baseline,
                task_states=pruning_states,
                Gq_diag=Gq_diag,
                epsilon_q=epsilon_q,
                n_qubits=args.n_qubits,
            )
            spearman, kendall = _rank_correlations(
                q1_order, _rank_positions(q_scores[qbeta])
            )
            metrics: Dict[str, Any] = {}
            reused = False
            if not args.selection_only:
                key = tuple(deleted)
                if key not in evaluation_cache:
                    evaluation_cache[key] = _evaluate_selection(
                        deleted=deleted,
                        baseline_circuit=pruning_baseline,
                        trained_readout_w=readout_w,
                        trained_readout_b=readout_b,
                        train_states=train_states,
                        train_labels=train_labels,
                        test_states=test_states,
                        test_labels=test_labels,
                        acc_control=acc_control,
                        args=args,
                    )
                else:
                    reused = True
                metrics = dict(evaluation_cache[key])

            row = {
                **metrics,
                "experiment_family": "q_sensitivity_upgrade_20260810",
                "experiment": "mnist49_q_geometry_matched_k_pilot",
                "selection_only": bool(args.selection_only),
                "task": "classification",
                "dataset": "mnist49",
                "seed": args.seed,
                "delta": float(delta),
                "qbeta": float(qbeta),
                "q": float(q),
                "lambda": float(lam),
                "Mq": float(Mq),
                "epsilon_q": float(epsilon_q),
                "source_target_delete_count": source_target_k,
                "matched_delete_count": matched_k,
                "physical_certificate_k_shortfall": source_target_k - matched_k,
                "deleted_pos_ids": deleted,
                "native_threshold_delete_count": len(native_deleted),
                "native_threshold_deleted_pos_ids": native_deleted,
                "fixed_physical_eligible_positions": len(physical_eligible),
                "fixed_physical_certificate_pass": all(
                    position in physical_eligible for position in deleted
                ),
                "matched_jaccard_vs_q1": _jaccard(deleted, q1_selected),
                "matched_different_positions_vs_q1": len(
                    set(deleted) ^ set(q1_selected)
                ),
                "spearman_vs_q1": spearman,
                "kendall_vs_q1": kendall,
                "evaluation_reused_for_identical_selection": reused,
                "acc_trained_before_extra_steps": acc_trained,
                "acc_full_control_after_extra_steps": acc_control,
                "baseline_circuit_signature": baseline_signature,
            }
            result_rows.append(row)
            print(
                f"[OK] seed={args.seed} delta={delta:g} qbeta={qbeta:g} "
                f"K={matched_k} diff={row['matched_different_positions_vs_q1']} "
                f"nativeK={len(native_deleted)}"
                + (
                    ""
                    if args.selection_only
                    else f" acc={row['acc_pruned_finetuned']:.4f}"
                ),
                flush=True,
            )

    _atomic_new_json(
        args.output_json,
        {
            "meta": {
                "experiment_family": "q_sensitivity_upgrade_20260810",
                "protocol": "same_trained_mnist49_checkpoint_same_K_same_fixed_physical_eligible_set_same_finetune_budget",
                "source_checkpoint": str(source_checkpoint),
                "source_snapshot": str(args.snapshot_json.resolve()) if args.snapshot_json else None,
                "snapshot_origin": snapshot_origin,
                "seed": args.seed,
                "deltas": args.deltas,
                "qbetas": qbetas,
                "selection_only": bool(args.selection_only),
                "expected_rows": len(args.deltas) * len(qbetas),
            },
            "trained_snapshot": {
                "seed": args.seed,
                "trained_parameters": trained_params.tolist(),
                "readout_w": readout_w.tolist(),
                "readout_b": float(readout_b),
                "loss_history": loss_history.tolist(),
                "baseline_circuit_signature": baseline_signature,
                "acc_trained": acc_trained,
            },
            "score_rows": score_rows,
            "results": result_rows,
        },
    )
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
