from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry
from iprune.experiments.qnn_training import (
    evaluate_qnn,
    stratified_calibration_test_split,
    stratified_subsample,
    train_qnn_circuit,
)
from iprune.experiments.tasks_classification import (
    ToyVisionConfig,
    build_bas_splits,
    build_fashion_sandal_boot_splits,
    build_mnist49_splits,
)
from iprune.gates import build_hea_gates_from_parameters
from iprune.pruning_true import q_iPrune_true_gate_pruning_by_identity
from scripts.legacy.sweep_all_v2 import _circuit_signature, _logical_resource_metrics


def _atomic_new_json(path: Path, payload: Dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a QNN, prune using calibration only, then fine-tune and test."
    )
    parser.add_argument("--dataset", choices=["mnist49", "fashion_sb", "bas"], required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.5, 0.8, 1.0, 1.2])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--n_qubits", type=int, default=None)
    parser.add_argument("--qnn_depth", type=int, default=3)
    parser.add_argument("--train_steps", type=int, default=40)
    parser.add_argument("--finetune_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_train", type=int, default=512)
    parser.add_argument("--n_calibration", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--qbeta", type=float, default=0.0)
    args = parser.parse_args()

    n_qubits = args.n_qubits or (4 if args.dataset == "bas" else 8)
    tcfg = ToyVisionConfig(
        n_qubits=n_qubits,
        data_dir=str(args.data_dir),
        seed=args.seed,
        download=not args.no_download,
        n_train_pool=max(args.n_train, 512),
        n_val_pool=args.n_calibration + args.n_test,
    )
    if args.dataset == "mnist49":
        tr_states, tr_y, va_states, va_y = build_mnist49_splits(tcfg)
    elif args.dataset == "fashion_sb":
        tr_states, tr_y, va_states, va_y = build_fashion_sandal_boot_splits(tcfg)
    else:
        tr_states, tr_y, va_states, va_y = build_bas_splits(tcfg, H=4, W=4)

    train_states, train_y = stratified_subsample(tr_states, tr_y, args.n_train, args.seed + 10)
    cal_states, cal_y, test_states, test_y = stratified_calibration_test_split(
        va_states,
        va_y,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        seed=args.seed + 20,
    )
    calibration_states, _ = stratified_subsample(cal_states, cal_y, args.M, args.seed + 30)

    rng = np.random.default_rng(args.seed)
    initial_params = rng.normal(scale=0.15, size=(args.qnn_depth, n_qubits, 3))
    _, initial_circuit = build_hea_gates_from_parameters(
        initial_params, include_cnot=True, include_identity_rot=False
    )
    trained = train_qnn_circuit(
        circuit=initial_circuit,
        train_states=train_states,
        train_labels=train_y,
        n_qubits=n_qubits,
        steps=args.train_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed + 100,
        device=args.device,
    )
    acc_trained = evaluate_qnn(
        circuit=trained.circuit,
        states=test_states,
        labels=test_y,
        readout_w=trained.readout_w,
        readout_b=trained.readout_b,
        n_qubits=n_qubits,
        device=args.device,
    )

    control = train_qnn_circuit(
        circuit=trained.circuit,
        train_states=train_states,
        train_labels=train_y,
        n_qubits=n_qubits,
        steps=args.finetune_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed + 200,
        device=args.device,
        initial_readout=(trained.readout_w, trained.readout_b),
    )
    acc_control = evaluate_qnn(
        circuit=control.circuit,
        states=test_states,
        labels=test_y,
        readout_w=control.readout_w,
        readout_b=control.readout_b,
        n_qubits=n_qubits,
        device=args.device,
    )
    all_gates, trained_circuit = build_hea_gates_from_parameters(
        np.asarray([gate.params for gate in trained.circuit if gate.gate_type == "Rot"], dtype=np.float64).reshape(args.qnn_depth, n_qubits, 3),
        include_cnot=True,
        include_identity_rot=True,
    )
    resources_before = _logical_resource_metrics(trained_circuit, n_qubits)

    rows: List[Dict[str, Any]] = []
    for delta in args.deltas:
        cfg = PaperProtocolConfig(
            device=args.device,
            n_qubits=n_qubits,
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
        pruned = q_iPrune_true_gate_pruning_by_identity(
            all_gates=all_gates,
            baseline_circuit=trained_circuit,
            n_qubits=n_qubits,
            device_name=args.device,
            task_states=calibration_states,
            Gq_diag=Gq_diag,
            epsilon_q=epsilon_q,
        )
        acc_one = evaluate_qnn(
            circuit=pruned.compact_circuit,
            states=test_states,
            labels=test_y,
            readout_w=trained.readout_w,
            readout_b=trained.readout_b,
            n_qubits=n_qubits,
            device=args.device,
        )
        tuned = train_qnn_circuit(
            circuit=pruned.compact_circuit,
            train_states=train_states,
            train_labels=train_y,
            n_qubits=n_qubits,
            steps=args.finetune_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed + 200,
            device=args.device,
            initial_readout=(trained.readout_w, trained.readout_b),
        )
        acc_tuned = evaluate_qnn(
            circuit=tuned.circuit,
            states=test_states,
            labels=test_y,
            readout_w=tuned.readout_w,
            readout_b=tuned.readout_b,
            n_qubits=n_qubits,
            device=args.device,
        )
        resources_after = _logical_resource_metrics(pruned.compact_circuit, n_qubits)
        one_drop = acc_control - acc_one
        tuned_drop = acc_control - acc_tuned
        recovery = None if one_drop <= 1e-12 else (one_drop - tuned_drop) / one_drop
        row = {
            "experiment_family": "supplementary_upgrade",
            "experiment": "trained_qnn_true_pruning_post_finetune",
            "task": "classification",
            "dataset": args.dataset,
            "seed": args.seed,
            "n_qubits": n_qubits,
            "qnn_depth": args.qnn_depth,
            "train_steps": args.train_steps,
            "finetune_steps": args.finetune_steps,
            "train_size": len(train_y),
            "calibration_size": len(cal_y),
            "test_size": len(test_y),
            "pruning_ensemble_M": len(calibration_states),
            "delta": delta,
            "epsilon_q": float(epsilon_q),
            "lambda": float(lam),
            "q": float(q),
            "Mq": float(Mq),
            "pruned_positions": int(pruned.pruned_positions),
            "gate_count_before": int(pruned.gate_count_before),
            "gate_count_after": int(pruned.gate_count_after),
            "actual_gate_prune_ratio": float(pruned.pruned_positions / max(1, pruned.gate_count_before)),
            "logical_depth_before": resources_before["logical_depth"],
            "logical_depth_after": resources_after["logical_depth"],
            "acc_trained_before_extra_steps": acc_trained,
            "acc_full_control_after_extra_steps": acc_control,
            "acc_pruned_one_shot": acc_one,
            "acc_pruned_finetuned": acc_tuned,
            "acc_drop_one_shot_vs_control": float(one_drop),
            "acc_drop_finetuned_vs_control": float(tuned_drop),
            "recovery_fraction_vs_control": None if recovery is None else float(recovery),
            "dq_max_over_replaced": float(pruned.dq_max_over_replaced),
            "physical_angle_max_over_replaced": float(pruned.physical_angle_max_over_replaced),
            "baseline_circuit_signature": _circuit_signature(trained_circuit),
            "pruned_circuit_signature": _circuit_signature(pruned.compact_circuit),
            "finetuned_circuit_signature": _circuit_signature(tuned.circuit),
        }
        rows.append(row)
        print(
            f"[OK] dataset={args.dataset} seed={args.seed} delta={delta:g} "
            f"gates={row['gate_count_before']}->{row['gate_count_after']} "
            f"acc_control={acc_control:.4f} acc_one={acc_one:.4f} acc_tuned={acc_tuned:.4f}",
            flush=True,
        )

    _atomic_new_json(
        args.output_json,
        {
            "meta": {
                "protocol": "trained_qnn_train_calibration_test_true_pruning_post_finetune",
                "dataset": args.dataset,
                "seed": args.seed,
                "deltas": args.deltas,
                "expected_rows": len(args.deltas),
            },
            "results": rows,
        },
    )
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
