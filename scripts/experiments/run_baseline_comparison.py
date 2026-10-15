from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from iprune.experiments.matched_position_baselines import (
    MatchedPositionBaseline,
    build_matched_position_baselines,
)
from iprune.experiments.protocol import (
    PaperProtocolConfig,
    make_q_geometry,
    run_positionaware_paper_protocol,
    simulate_outputs,
    trace_distance_stats,
)
from iprune.experiments.task_metrics import (
    evaluate_classification_accuracy_drop,
    evaluate_vqe_energy_deviation,
)
from iprune.experiments.tasks_classification import (
    ToyVisionConfig,
    build_bas_splits,
    build_fashion_sandal_boot_splits,
    build_mnist49_splits,
    sample_task_ensemble_from_validation,
)
from iprune.experiments.tasks_vqe import VQEConfig, sample_intermediate_states_from_vqe
from iprune.gates import GateOp, build_hea_gates


PROJECT_DIR = Path(__file__).resolve().parents[2]


def all_gates_builder(cfg: PaperProtocolConfig) -> List[GateOp]:
    return build_hea_gates(
        cfg.n_qubits,
        cfg.depth,
        seed=cfg.seed,
        rot_pool_k=cfg.rot_pool_k,
        rot_pool_sigma=cfg.rot_pool_sigma,
        include_cnot=cfg.include_cnot,
    )


def _signature(circuit: List[GateOp]) -> str:
    payload = "\n".join(
        f"{gate.pos_id}|{gate.name}|{gate.params}" for gate in circuit
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _atomic_new_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _paper_method(
    prune_result: Any,
    baseline_circuit: List[GateOp],
    eligible_positions: int,
) -> MatchedPositionBaseline:
    replaced_positions = [
        position
        for position, (baseline_gate, selected_gate) in enumerate(
            zip(baseline_circuit, prune_result.selected_circuit)
        )
        if selected_gate is not baseline_gate
    ]
    return MatchedPositionBaseline(
        method="qiprune_positionaware",
        selected_circuit=list(prune_result.selected_circuit),
        replacement_count=int(prune_result.selected_replacement_count),
        eligible_positions=int(eligible_positions),
        replaced_positions=replaced_positions,
        selected_scores=[],
    )


def _base_record(
    *,
    method: MatchedPositionBaseline,
    cfg: PaperProtocolConfig,
    dataset: str,
    candidate_L: int,
    baseline_signature: str,
    trace_mean: float,
    trace_max: float,
) -> Dict[str, Any]:
    return {
        "task": "classification" if dataset != "tfim" else "vqe",
        "dataset": dataset,
        "baseline_type": method.method,
        "comparison_budget": "matched_selected_forward_replacements",
        "candidate_L_qiprune": int(candidate_L),
        "selected_replacement_count": int(method.replacement_count),
        "eligible_reference_replacement_positions": int(method.eligible_positions),
        "replaced_positions": method.replaced_positions,
        "selected_scores": method.selected_scores,
        "baseline_circuit_signature": baseline_signature,
        "selected_circuit_signature": _signature(method.selected_circuit),
        "n_qubits": cfg.n_qubits,
        "depth": cfg.depth,
        "M": cfg.M,
        "seed": cfg.seed,
        "gamma": cfg.gamma,
        "alpha": cfg.alpha,
        "qbeta": cfg.qbeta,
        "delta": cfg.delta,
        "rot_pool_k": cfg.rot_pool_k,
        "rot_pool_sigma": cfg.rot_pool_sigma,
        "include_cnot": cfg.include_cnot,
        "trace_dist_mean": float(trace_mean),
        "trace_dist_max": float(trace_max),
    }


def _classification_rows(args: argparse.Namespace, cfg: PaperProtocolConfig) -> List[Dict[str, Any]]:
    tcfg = ToyVisionConfig(
        n_qubits=cfg.n_qubits,
        data_dir=str(args.data_dir),
        seed=cfg.seed,
        download=not args.no_download,
        n_train_pool=args.n_train_pool,
        n_val_pool=args.n_val_pool,
    )
    if args.dataset == "mnist49":
        tr_states, tr_y, va_states, va_y = build_mnist49_splits(tcfg)
    elif args.dataset == "fashion_sb":
        tr_states, tr_y, va_states, va_y = build_fashion_sandal_boot_splits(tcfg)
    else:
        tr_states, tr_y, va_states, va_y = build_bas_splits(
            tcfg, H=args.bas_h, W=args.bas_w
        )
    task_states, _ = sample_task_ensemble_from_validation(
        va_states, va_y, M=cfg.M, seed=cfg.seed
    )
    report, baseline, prune_result = run_positionaware_paper_protocol(
        cfg=cfg,
        all_gates_builder=all_gates_builder,
        task_states=task_states,
        return_artifacts=True,
    )
    all_gates = prune_result.all_gates
    prefixes = prune_result.prefix_states_list
    _, _, Gq_diag, _, _ = make_q_geometry(cfg)
    comparisons = build_matched_position_baselines(
            all_gates=all_gates,
            baseline_circuit=baseline,
            prefix_states_list=prefixes,
            task_states=task_states,
            Gq_diag=Gq_diag,
            replacement_count=prune_result.selected_replacement_count,
            n_qubits=cfg.n_qubits,
            seed=cfg.seed + 2025,
    )
    methods = [
        _paper_method(prune_result, baseline, comparisons[0].eligible_positions)
    ]
    methods.extend(comparisons)
    task_base = simulate_outputs(baseline, task_states, cfg.n_qubits, cfg.device)
    train_base = simulate_outputs(baseline, tr_states, cfg.n_qubits, cfg.device)
    val_base = simulate_outputs(baseline, va_states, cfg.n_qubits, cfg.device)
    baseline_signature = _signature(baseline)
    rows = []
    for method in methods:
        task_selected = simulate_outputs(
            method.selected_circuit, task_states, cfg.n_qubits, cfg.device
        )
        trace_mean, trace_max = trace_distance_stats(task_base, task_selected)
        val_selected = simulate_outputs(
            method.selected_circuit, va_states, cfg.n_qubits, cfg.device
        )
        metrics = evaluate_classification_accuracy_drop(
            n_qubits=cfg.n_qubits,
            train_out_states_base=train_base,
            train_labels=tr_y,
            val_out_states_base=val_base,
            val_out_states_prun=val_selected,
            val_labels=va_y,
            seed=cfg.seed,
        )
        row = _base_record(
            method=method,
            cfg=cfg,
            dataset=args.dataset,
            candidate_L=report["L"],
            baseline_signature=baseline_signature,
            trace_mean=trace_mean,
            trace_max=trace_max,
        )
        row.update(metrics)
        rows.append(row)
        print(
            f"[OK] {method.method} K={method.replacement_count} "
            f"drift_max={trace_max:.6f} acc_drop={metrics['acc_drop']:.6f}",
            flush=True,
        )
    return rows


def _vqe_rows(args: argparse.Namespace, cfg: PaperProtocolConfig) -> List[Dict[str, Any]]:
    vcfg = VQEConfig(
        n_qubits=cfg.n_qubits,
        depth=args.vqe_depth,
        steps=args.vqe_steps,
        lr=args.vqe_lr,
        seed=cfg.seed,
        J=args.J,
        h=args.h,
    )
    task_states = sample_intermediate_states_from_vqe(
        vcfg, M=cfg.M, device=cfg.device
    )
    report, baseline, prune_result = run_positionaware_paper_protocol(
        cfg=cfg,
        all_gates_builder=all_gates_builder,
        task_states=task_states,
        return_artifacts=True,
    )
    all_gates = prune_result.all_gates
    prefixes = prune_result.prefix_states_list
    _, _, Gq_diag, _, _ = make_q_geometry(cfg)
    comparisons = build_matched_position_baselines(
            all_gates=all_gates,
            baseline_circuit=baseline,
            prefix_states_list=prefixes,
            task_states=task_states,
            Gq_diag=Gq_diag,
            replacement_count=prune_result.selected_replacement_count,
            n_qubits=cfg.n_qubits,
            seed=cfg.seed + 2025,
    )
    methods = [
        _paper_method(prune_result, baseline, comparisons[0].eligible_positions)
    ]
    methods.extend(comparisons)
    task_base = simulate_outputs(baseline, task_states, cfg.n_qubits, cfg.device)
    baseline_signature = _signature(baseline)
    rows = []
    for method in methods:
        task_selected = simulate_outputs(
            method.selected_circuit, task_states, cfg.n_qubits, cfg.device
        )
        trace_mean, trace_max = trace_distance_stats(task_base, task_selected)
        metrics = evaluate_vqe_energy_deviation(
            n_qubits=cfg.n_qubits,
            states_base=task_base,
            states_prun=task_selected,
            J=args.J,
            h=args.h,
        )
        row = _base_record(
            method=method,
            cfg=cfg,
            dataset="tfim",
            candidate_L=report["L"],
            baseline_signature=baseline_signature,
            trace_mean=trace_mean,
            trace_max=trace_max,
        )
        row.update(metrics)
        rows.append(row)
        print(
            f"[OK] {method.method} K={method.replacement_count} "
            f"drift_max={trace_max:.6f} dE_max={metrics['dE_max']:.6f}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched-forward-budget comparison baselines for the candidate protocol."
    )
    parser.add_argument("--task", choices=["classification", "vqe"], default="classification")
    parser.add_argument("--dataset", choices=["mnist49", "fashion_sb", "bas"], default="bas")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--qbeta", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--rot_pool_k", type=int, default=4)
    parser.add_argument("--rot_pool_sigma", type=float, default=0.001)
    parser.add_argument("--include_cnot", action="store_true")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--n_train_pool", type=int, default=4000)
    parser.add_argument("--n_val_pool", type=int, default=2000)
    parser.add_argument("--bas_h", type=int, default=4)
    parser.add_argument("--bas_w", type=int, default=4)
    parser.add_argument("--vqe_depth", type=int, default=6)
    parser.add_argument("--vqe_steps", type=int, default=40)
    parser.add_argument("--vqe_lr", type=float, default=0.1)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    args = parser.parse_args()
    cfg = PaperProtocolConfig(
        device=args.device,
        n_qubits=args.n_qubits,
        depth=args.depth,
        M=args.M,
        seed=args.seed,
        gamma=args.gamma,
        alpha=args.alpha,
        qbeta=args.qbeta,
        delta=args.delta,
        rot_pool_k=args.rot_pool_k,
        rot_pool_sigma=args.rot_pool_sigma,
        include_cnot=args.include_cnot,
    )
    rows = (
        _classification_rows(args, cfg)
        if args.task == "classification"
        else _vqe_rows(args, cfg)
    )
    payload = {
        "meta": {
            "comparison_budget": "matched_selected_forward_replacements",
            "methods": [row["baseline_type"] for row in rows],
            "warning": "candidate_L is reported separately and is not a gate count",
        },
        "results": rows,
    }
    _atomic_new_json(args.output_json, payload)
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
