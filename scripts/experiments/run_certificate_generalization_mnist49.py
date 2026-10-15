from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry
from iprune.experiments.qnn_training import (
    stratified_calibration_test_split,
    stratified_subsample,
    train_qnn_circuit,
)
from iprune.experiments.readout import z_expectations_from_statevectors
from iprune.experiments.tasks_classification import ToyVisionConfig, build_mnist49_splits
from iprune.gates import GateOp, build_hea_gates_from_parameters, is_identity_rot
from iprune.metrics import bound_rhs
from iprune.pruning_true import q_iPrune_true_gate_pruning_by_identity
from scripts.experiments.run_q_sensitivity_selection_diagnostic import (
    _apply_cnot,
    _apply_one_qubit,
    _rot_matrix,
    _weighted_angles,
)
from scripts.legacy.sweep_all_v2 import _circuit_signature, _logical_resource_metrics


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "certificate_generalization_20260812" / "mnist49"
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _apply_gate(states: np.ndarray, gate: GateOp, n_qubits: int) -> np.ndarray:
    if gate.gate_type == "Rot":
        return _apply_one_qubit(states, _rot_matrix(gate), int(gate.wires[0]), n_qubits)
    if gate.gate_type == "CNOT":
        return _apply_cnot(states, int(gate.wires[0]), int(gate.wires[1]), n_qubits)
    raise ValueError(f"Unsupported gate type: {gate.gate_type}")


def _propagate(
    circuit: Sequence[GateOp], states: np.ndarray, n_qubits: int
) -> np.ndarray:
    current = np.asarray(states, dtype=np.complex128).copy()
    for index, gate in enumerate(circuit):
        if int(gate.pos_id) != index:
            raise ValueError("Circuit positions must be consecutive")
        current = _apply_gate(current, gate, n_qubits)
    return current


def _trace_distances(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    overlaps = np.abs(np.sum(np.conjugate(states_a) * states_b, axis=1))
    norm_a = np.linalg.norm(states_a, axis=1)
    norm_b = np.linalg.norm(states_b, axis=1)
    normalized = np.clip(overlaps / np.maximum(norm_a * norm_b, 1e-15), 0.0, 1.0)
    residual = np.maximum(0.0, 1.0 - normalized**2)
    residual[residual <= 32.0 * np.finfo(np.float64).eps] = 0.0
    return np.sqrt(residual)


def _accuracy_from_states(
    states: np.ndarray,
    labels: np.ndarray,
    readout_w: np.ndarray,
    readout_b: float,
    n_qubits: int,
) -> float:
    features = z_expectations_from_statevectors(states, n_qubits)
    logits = features @ np.asarray(readout_w, dtype=np.float64) + float(readout_b)
    return float(np.mean((logits >= 0.0).astype(int) == np.asarray(labels, dtype=int)))


def _audit_selected_path(
    *,
    baseline: Sequence[GateOp],
    selected: Sequence[GateOp],
    states: np.ndarray,
    q_weights: np.ndarray,
    epsilon_q: float,
    n_qubits: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
    """Audit deleted gates on a specified ensemble along the selected prefix.

    The pruning theorem is calibrated on the calibration ensemble.  Applying the
    same checks to a disjoint test ensemble is therefore a generalization audit,
    not an extra assumption silently folded into selection.
    """
    current = np.asarray(states, dtype=np.complex128).copy()
    physical_weights = np.ones(2**n_qubits, dtype=np.float64)
    local_rows: List[Dict[str, Any]] = []

    for position, (baseline_gate, selected_gate) in enumerate(zip(baseline, selected)):
        deleted = (
            baseline_gate.gate_type == "Rot"
            and not is_identity_rot(baseline_gate)
            and is_identity_rot(selected_gate)
        )
        if deleted:
            q_angles = _weighted_angles(
                baseline_gate, selected_gate, current, q_weights, n_qubits
            )
            physical_angles = _weighted_angles(
                baseline_gate, selected_gate, current, physical_weights, n_qubits
            )
            physical_trace = np.sin(physical_angles)
            local_rows.append(
                {
                    "position": position,
                    "q_angle_mean": float(np.mean(q_angles)),
                    "q_angle_max": float(np.max(q_angles)),
                    "physical_angle_mean": float(np.mean(physical_angles)),
                    "physical_angle_max": float(np.max(physical_angles)),
                    "physical_trace_mean": float(np.mean(physical_trace)),
                    "physical_trace_max": float(np.max(physical_trace)),
                    "q_state_violations": int(np.sum(q_angles > epsilon_q + 1e-12)),
                    "physical_state_violations": int(
                        np.sum(physical_angles > epsilon_q + 1e-12)
                    ),
                    "states": int(len(states)),
                }
            )
        current = _apply_gate(current, selected_gate, n_qubits)

    deleted_count = len(local_rows)
    state_checks = deleted_count * len(states)
    physical_gate_violations = sum(
        row["physical_angle_max"] > epsilon_q + 1e-12 for row in local_rows
    )
    q_gate_violations = sum(
        row["q_angle_max"] > epsilon_q + 1e-12 for row in local_rows
    )
    summary = {
        "deleted_positions": deleted_count,
        "state_position_checks": state_checks,
        "q_gate_violations": int(q_gate_violations),
        "physical_gate_violations": int(physical_gate_violations),
        "q_state_violations": int(sum(row["q_state_violations"] for row in local_rows)),
        "physical_state_violations": int(
            sum(row["physical_state_violations"] for row in local_rows)
        ),
        "q_angle_max": float(
            max((row["q_angle_max"] for row in local_rows), default=0.0)
        ),
        "physical_angle_max": float(
            max((row["physical_angle_max"] for row in local_rows), default=0.0)
        ),
        "path_sum_physical_trace_max": float(
            sum(row["physical_trace_max"] for row in local_rows)
        ),
    }
    return summary, local_rows, current


def _bootstrap(values: Sequence[float], seed: int, draws: int = 20000) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }


def _train_and_audit_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    n_qubits = 8
    task_config = ToyVisionConfig(
        n_qubits=n_qubits,
        data_dir=str(args.data_dir),
        seed=seed,
        download=not args.no_download,
        n_train_pool=max(args.n_train, 512),
        n_val_pool=args.n_calibration + args.n_test,
    )
    train_pool_states, train_pool_labels, validation_states, validation_labels = (
        build_mnist49_splits(task_config)
    )
    train_states, train_labels = stratified_subsample(
        train_pool_states, train_pool_labels, args.n_train, seed + 10
    )
    cal_states, cal_labels, test_states, test_labels = stratified_calibration_test_split(
        validation_states,
        validation_labels,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        seed=seed + 20,
    )
    pruning_states, _ = stratified_subsample(
        cal_states, cal_labels, args.M, seed + 30
    )

    rng = np.random.default_rng(seed)
    initial_params = rng.normal(scale=0.15, size=(args.qnn_depth, n_qubits, 3))
    _, initial_circuit = build_hea_gates_from_parameters(
        initial_params, include_cnot=True, include_identity_rot=False
    )
    trained = train_qnn_circuit(
        circuit=initial_circuit,
        train_states=train_states,
        train_labels=train_labels,
        n_qubits=n_qubits,
        steps=args.train_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=seed + 100,
        device=args.device,
    )
    trained_params = np.asarray(
        [gate.params for gate in trained.circuit if gate.gate_type == "Rot"],
        dtype=np.float64,
    ).reshape(args.qnn_depth, n_qubits, 3)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        trained_params, include_cnot=True, include_identity_rot=True
    )
    baseline_test_outputs = _propagate(baseline_circuit, test_states, n_qubits)
    baseline_accuracy = _accuracy_from_states(
        baseline_test_outputs,
        test_labels,
        trained.readout_w,
        trained.readout_b,
        n_qubits,
    )
    resources_before = _logical_resource_metrics(baseline_circuit, n_qubits)

    rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []
    for delta in args.deltas:
        config = PaperProtocolConfig(
            device=args.device,
            n_qubits=n_qubits,
            depth=args.qnn_depth,
            M=args.M,
            seed=seed,
            gamma=args.gamma,
            alpha=args.alpha,
            qbeta=args.qbeta,
            delta=delta,
            rot_pool_k=1,
            rot_pool_sigma=0.0,
            include_cnot=True,
        )
        lam, q, q_weights, mq, epsilon_q = make_q_geometry(config)
        pruned = q_iPrune_true_gate_pruning_by_identity(
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            n_qubits=n_qubits,
            device_name=args.device,
            task_states=pruning_states,
            Gq_diag=q_weights,
            epsilon_q=epsilon_q,
        )
        calibration_audit, calibration_local, _ = _audit_selected_path(
            baseline=baseline_circuit,
            selected=pruned.selected_circuit,
            states=pruning_states,
            q_weights=q_weights,
            epsilon_q=epsilon_q,
            n_qubits=n_qubits,
        )
        test_audit, test_local, pruned_test_outputs = _audit_selected_path(
            baseline=baseline_circuit,
            selected=pruned.selected_circuit,
            states=test_states,
            q_weights=q_weights,
            epsilon_q=epsilon_q,
            n_qubits=n_qubits,
        )
        trace_distances = _trace_distances(baseline_test_outputs, pruned_test_outputs)
        pruned_accuracy = _accuracy_from_states(
            pruned_test_outputs,
            test_labels,
            trained.readout_w,
            trained.readout_b,
            n_qubits,
        )
        resources_after = _logical_resource_metrics(pruned.compact_circuit, n_qubits)
        deleted = int(pruned.pruned_positions)
        physical_threshold_rhs = float(deleted * np.sin(epsilon_q))
        original_paper_rhs = float(bound_rhs(epsilon_q, mq, deleted))
        test_trace_max = float(np.max(trace_distances))
        row = {
            "seed": seed,
            "dataset": "mnist49",
            "delta": float(delta),
            "epsilon_q": float(epsilon_q),
            "lambda": float(lam),
            "q": float(q),
            "Mq": float(mq),
            "train_size": int(len(train_labels)),
            "calibration_pool_size": int(len(cal_labels)),
            "pruning_ensemble_M": int(len(pruning_states)),
            "test_size": int(len(test_labels)),
            "gate_count_before": int(pruned.gate_count_before),
            "gate_count_after": int(pruned.gate_count_after),
            "pruned_positions": deleted,
            "actual_gate_prune_ratio": float(
                deleted / max(1, pruned.gate_count_before)
            ),
            "logical_depth_before": int(resources_before["logical_depth"]),
            "logical_depth_after": int(resources_after["logical_depth"]),
            "baseline_accuracy": baseline_accuracy,
            "pruned_one_shot_accuracy": pruned_accuracy,
            "accuracy_change_pruned_minus_baseline": float(
                pruned_accuracy - baseline_accuracy
            ),
            "calibration_q_gate_violations": calibration_audit["q_gate_violations"],
            "calibration_physical_gate_violations": calibration_audit[
                "physical_gate_violations"
            ],
            "test_q_gate_violations": test_audit["q_gate_violations"],
            "test_physical_gate_violations": test_audit["physical_gate_violations"],
            "test_q_state_violations": test_audit["q_state_violations"],
            "test_physical_state_violations": test_audit[
                "physical_state_violations"
            ],
            "test_state_position_checks": test_audit["state_position_checks"],
            "test_physical_gate_violation_rate": float(
                test_audit["physical_gate_violations"] / max(1, deleted)
            ),
            "test_physical_state_violation_rate": float(
                test_audit["physical_state_violations"]
                / max(1, test_audit["state_position_checks"])
            ),
            "test_trace_distance_mean": float(np.mean(trace_distances)),
            "test_trace_distance_max": test_trace_max,
            "physical_threshold_rhs_raw": physical_threshold_rhs,
            "physical_threshold_rhs_clipped": min(1.0, physical_threshold_rhs),
            "original_paper_rhs_raw": original_paper_rhs,
            "original_paper_rhs_clipped": min(1.0, original_paper_rhs),
            "test_path_sum_physical_trace_max": test_audit[
                "path_sum_physical_trace_max"
            ],
            "test_trace_exceeds_physical_threshold_rhs": bool(
                test_trace_max > physical_threshold_rhs + 1e-10
            ),
            "test_trace_exceeds_test_path_sum": bool(
                test_trace_max
                > test_audit["path_sum_physical_trace_max"] + 1e-10
            ),
            "threshold_bound_tightness_ratio": float(
                test_trace_max / physical_threshold_rhs
                if physical_threshold_rhs > 0.0
                else 0.0
            ),
            "test_path_bound_tightness_ratio": float(
                test_trace_max / test_audit["path_sum_physical_trace_max"]
                if test_audit["path_sum_physical_trace_max"] > 0.0
                else 0.0
            ),
            "baseline_circuit_signature": _circuit_signature(baseline_circuit),
            "pruned_circuit_signature": _circuit_signature(pruned.compact_circuit),
        }
        rows.append(row)
        for ensemble, items in (("calibration", calibration_local), ("test", test_local)):
            for item in items:
                local_rows.append(
                    {
                        "seed": seed,
                        "delta": float(delta),
                        "epsilon_q": float(epsilon_q),
                        "ensemble": ensemble,
                        **item,
                    }
                )
        print(
            f"[OK] seed={seed} delta={delta:g} gates={pruned.gate_count_before}->"
            f"{pruned.gate_count_after} test_gate_viol="
            f"{test_audit['physical_gate_violations']}/{deleted} "
            f"Dmax={test_trace_max:.4f} rhs={physical_threshold_rhs:.4f}",
            flush=True,
        )

    return {
        "meta": {
            "experiment": "mnist49_certificate_calibration_to_heldout_generalization",
            "seed": seed,
            "selection_ensemble": "50 states sampled only from calibration partition",
            "heldout_ensemble": "512 disjoint test states",
            "certificate_semantics": (
                "selection requires both maximum q-angle and maximum physical "
                "Fubini-Study angle <= epsilon on calibration states"
            ),
            "heldout_semantics": (
                "test violations audit certificate generalization and are not "
                "used to revise the selected circuit"
            ),
            "no_finetuning": True,
        },
        "results": rows,
        "local_audits": local_rows,
    }


def _load_seed_payloads(rows_dir: Path, seeds: Iterable[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        path = rows_dir / f"mnist49_seed{seed}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["results"])
        local_rows.extend(payload["local_audits"])
    return rows, local_rows


def _summarize(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    metrics = (
        "actual_gate_prune_ratio",
        "accuracy_change_pruned_minus_baseline",
        "test_physical_gate_violation_rate",
        "test_physical_state_violation_rate",
        "test_trace_distance_mean",
        "test_trace_distance_max",
        "physical_threshold_rhs_raw",
        "test_path_sum_physical_trace_max",
        "threshold_bound_tightness_ratio",
        "test_path_bound_tightness_ratio",
    )
    for delta_index, delta in enumerate(sorted({float(row["delta"]) for row in rows})):
        group = [row for row in rows if float(row["delta"]) == delta]
        item: Dict[str, Any] = {
            "delta": delta,
            "seeds": len(group),
            "total_deleted_positions": int(sum(int(row["pruned_positions"]) for row in group)),
            "total_test_gate_violations": int(
                sum(int(row["test_physical_gate_violations"]) for row in group)
            ),
            "total_test_state_violations": int(
                sum(int(row["test_physical_state_violations"]) for row in group)
            ),
            "total_test_state_position_checks": int(
                sum(int(row["test_state_position_checks"]) for row in group)
            ),
            "whole_trace_threshold_bound_violations": int(
                sum(bool(row["test_trace_exceeds_physical_threshold_rhs"]) for row in group)
            ),
            "whole_trace_test_path_bound_violations": int(
                sum(bool(row["test_trace_exceeds_test_path_sum"]) for row in group)
            ),
        }
        for metric_index, metric in enumerate(metrics):
            estimate = _bootstrap(
                [float(row[metric]) for row in group],
                20260812 + 100 * delta_index + metric_index,
            )
            item[f"{metric}_mean"] = estimate["mean"]
            item[f"{metric}_ci_low"] = estimate["ci_low"]
            item[f"{metric}_ci_high"] = estimate["ci_high"]
        summary.append(item)
    return summary


def _plot(summary: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    delta = np.asarray([row["delta"] for row in summary], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))

    def errorbar(axis, metric: str, label: str, color: str) -> None:
        mean = np.asarray([row[f"{metric}_mean"] for row in summary], dtype=float)
        low = np.asarray([row[f"{metric}_ci_low"] for row in summary], dtype=float)
        high = np.asarray([row[f"{metric}_ci_high"] for row in summary], dtype=float)
        axis.errorbar(delta, mean, yerr=[mean - low, high - mean], marker="o", label=label, color=color)

    errorbar(axes[0], "actual_gate_prune_ratio", "Actual deletion", "#2878B5")
    axes[0].set_title("Certified compression")
    axes[0].set_ylabel("Total-gate pruning ratio")
    axes[0].set_xlabel(r"Pruning scale $\delta$")
    axes[0].grid(alpha=0.25)

    errorbar(axes[1], "test_physical_gate_violation_rate", "Gate max", "#D95319")
    errorbar(axes[1], "test_physical_state_violation_rate", "State-position", "#77AC30")
    axes[1].set_title("Held-out certificate violations")
    axes[1].set_ylabel("Violation rate")
    axes[1].set_xlabel(r"Pruning scale $\delta$")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    errorbar(axes[2], "test_trace_distance_max", "Observed max", "#7E2F8E")
    errorbar(axes[2], "physical_threshold_rhs_raw", r"$L\sin(\epsilon)$", "#4DBEEE")
    errorbar(axes[2], "test_path_sum_physical_trace_max", "Held-out path sum", "#A2142F")
    axes[2].set_title("Circuit drift and bounds")
    axes[2].set_ylabel("Trace distance / bound")
    axes[2].set_xlabel(r"Pruning scale $\delta$")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.suptitle("MNIST 4-vs-9: calibration certificate to disjoint held-out states")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_certificate_generalization.png", dpi=220)
    fig.savefig(output_dir / "fig_certificate_generalization.pdf")
    plt.close(fig)


def _write_summary(summary: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    seed_count = max((int(row["seeds"]) for row in summary), default=0)
    threshold_count = len(summary)
    total_deleted = sum(int(row["total_deleted_positions"]) for row in summary)
    total_gate_violations = sum(int(row["total_test_gate_violations"]) for row in summary)
    total_state_violations = sum(int(row["total_test_state_violations"]) for row in summary)
    total_state_checks = sum(int(row["total_test_state_position_checks"]) for row in summary)
    moderate = [row for row in summary if 0.0 < float(row["delta"]) <= 0.5]
    moderate_deleted = sum(int(row["total_deleted_positions"]) for row in moderate)
    moderate_gate_violations = sum(
        int(row["total_test_gate_violations"]) for row in moderate
    )
    moderate_state_violations = sum(
        int(row["total_test_state_violations"]) for row in moderate
    )
    moderate_state_checks = sum(
        int(row["total_test_state_position_checks"]) for row in moderate
    )
    whole_threshold_violations = sum(
        int(row["whole_trace_threshold_bound_violations"]) for row in summary
    )
    whole_path_violations = sum(
        int(row["whole_trace_test_path_bound_violations"]) for row in summary
    )
    lines = [
        "# MNIST49 物理证书从校准集到独立测试集的外推核验",
        "",
        "## 协议",
        "",
        "- 沿用原 MNIST 4-vs-9 训练与 q-iPrune 真实删门协议。",
        "- 每个种子仅用 50 个 calibration states 做删门决策。",
        "- 512 个互不重叠的 test states 只用于事后证书外推审计，不参与选门。",
        "- 每次删除在 calibration states 上必须同时通过 q 加权角度和普通 Fubini--Study 最大角度阈值。",
        "- 本实验不微调，隔离删门与证书本身。",
        "",
        "## 汇总",
        "",
        "| delta | 实际删门率 | test 门级违规 | test 状态-位置违规 | 整电路最大 trace | L sin(eps) | test 路径和界 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        gate_denominator = int(row["total_deleted_positions"])
        state_denominator = int(row["total_test_state_position_checks"])
        lines.append(
            f"| {float(row['delta']):.2f} | "
            f"{100*float(row['actual_gate_prune_ratio_mean']):.1f}% | "
            f"{int(row['total_test_gate_violations'])}/{gate_denominator} | "
            f"{int(row['total_test_state_violations'])}/{state_denominator} | "
            f"{float(row['test_trace_distance_max_mean']):.4f} | "
            f"{float(row['physical_threshold_rhs_raw_mean']):.4f} | "
            f"{float(row['test_path_sum_physical_trace_max_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 可支持的结论",
            "",
            f"- {seed_count} 个种子、{threshold_count} 个阈值共接受 {total_deleted} 次真实门删除。",
            f"- 独立测试集的门级物理证书违规为 {total_gate_violations}/{total_deleted}。",
            f"- 独立测试集的状态-位置级违规为 {total_state_violations}/{total_state_checks}。",
            f"- 在中等剪枝区间 0<delta<=0.5，共 {moderate_deleted} 次真实删门、{moderate_state_checks} 次独立测试检查；门级违规 {moderate_gate_violations}/{moderate_deleted}，状态-位置违规 {moderate_state_violations}/{moderate_state_checks}。",
            f"- 全部种子和阈值中，整电路最大 trace distance 超过 L sin(epsilon) 的次数为 {whole_threshold_violations}/{seed_count * threshold_count}，超过 held-out 逐门路径和界的次数为 {whole_path_violations}/{seed_count * threshold_count}。",
            "- calibration 上零违规是算法构造性质；test 上的结果衡量该证书在同任务未见样本上的经验外推，而不是全希尔伯特空间保证。",
            "- L sin(epsilon) 是预先由阈值和删门数给出的保守物理界；test 路径和界使用 held-out 局部角度，仅用于评估松紧度。",
            "- delta=0.5 和 0.8 时 L sin(epsilon) 通常超过 1，截断后只剩平凡上界；因此可认证压缩的主要信息区间位于更温和阈值，激进剪枝用于展示证书饱和与外推失败边界。",
            "",
            "## 表述边界",
            "",
            "该实验若显示 test 零或极低违规，可支持“证书在同分布未见任务样本上具有经验泛化”；仍不能宣称全局电路等价，也不能把经验外推写成新的理论定理。",
        ]
    )
    (output_dir / "RESULTS_SUMMARY_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MNIST49 calibration-certificate to held-out-state generalization audit."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--qnn_depth", type=int, default=3)
    parser.add_argument("--train_steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_train", type=int, default=512)
    parser.add_argument("--n_calibration", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--qbeta", type=float, default=0.0)
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    failures: List[Dict[str, Any]] = []
    for seed in args.seeds:
        output_path = rows_dir / f"mnist49_seed{seed}.json"
        if output_path.exists():
            print(f"[SKIP] seed={seed}: {output_path.name}", flush=True)
            continue
        try:
            payload = _train_and_audit_seed(args, seed)
            _atomic_json(output_path, payload)
        except Exception as error:  # pragma: no cover - resumable batch boundary
            failures.append(
                {"seed": seed, "error": repr(error), "traceback": traceback.format_exc()}
            )
            _atomic_json(output_dir / "failures.json", failures)
            if not args.continue_on_error:
                raise

    rows, local_rows = _load_seed_payloads(rows_dir, args.seeds)
    expected = len(args.seeds) * len(args.deltas)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} result rows, found {len(rows)}")
    summary = _summarize(rows)
    _atomic_csv(output_dir / "seed_level.csv", rows)
    _atomic_csv(output_dir / "position_level.csv", local_rows)
    _atomic_csv(output_dir / "summary_by_delta.csv", summary)
    _plot(summary, output_dir)
    _write_summary(summary, output_dir)
    _atomic_json(
        output_dir / "manifest.json",
        {
            "experiment": "mnist49_certificate_generalization_audit",
            "seeds": args.seeds,
            "deltas": args.deltas,
            "rows": len(rows),
            "local_rows": len(local_rows),
            "elapsed_seconds_this_invocation": time.time() - started,
            "failures": failures,
        },
    )
    print(f"Saved complete audit: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
