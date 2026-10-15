from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from iprune.experiments.qnn_training import stratified_calibration_test_split, stratified_subsample
from iprune.experiments.tasks_classification import (
    ToyVisionConfig,
    build_fashion_sandal_boot_splits,
    build_mnist49_splits,
)
from iprune.gates import GateOp, build_hea_gates_from_parameters
from scripts.experiments.run_q_matched_k_tfim_seed import _identity_candidate, _pools_by_position
from scripts.experiments.run_q_sensitivity_selection_diagnostic import _compute_prefix_states_fast, _weighted_angles


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_DIR = (
    PROJECT_DIR
    / "results"
    / "q_sensitivity_upgrade_20260810"
    / "mnist49_matched_k_selection"
    / "rows"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "task_conditioning_ablation_20260811" / "mnist49"
)
MODES = ("correct_mnist49_prefix", "mismatched_fashion_prefix", "haar_prefix", "no_prefix")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _correlation(x: Sequence[float], y: Sequence[float], kind: str) -> float:
    value = spearmanr(x, y).statistic if kind == "spearman" else kendalltau(x, y).statistic
    return float(value) if np.isfinite(value) else 0.0


def _haar_states(seed: int, count: int, dimension: int) -> np.ndarray:
    rng = np.random.default_rng(20260811 + seed)
    states = rng.normal(size=(count, dimension)) + 1j * rng.normal(size=(count, dimension))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return states.astype(np.complex128)


def _z_features(states: np.ndarray, n_qubits: int) -> np.ndarray:
    probabilities = np.abs(np.asarray(states, dtype=np.complex128)) ** 2
    basis = np.arange(2**n_qubits, dtype=np.int64)
    signs = np.stack(
        [1.0 - 2.0 * ((basis >> (n_qubits - 1 - wire)) & 1) for wire in range(n_qubits)],
        axis=1,
    )
    return probabilities @ signs


def _logits(
    circuit: Sequence[GateOp], states: np.ndarray, readout_w: np.ndarray, readout_b: float, n_qubits: int
) -> tuple[np.ndarray, np.ndarray]:
    outputs = _compute_prefix_states_fast(circuit, states, n_qubits)[-1]
    logits = _z_features(outputs, n_qubits) @ readout_w + readout_b
    return np.asarray(logits, dtype=np.float64), np.asarray(outputs, dtype=np.complex128)


def _mean_trace_distance(states_a: np.ndarray, states_b: np.ndarray) -> float:
    overlaps = np.abs(np.sum(np.conj(states_a) * states_b, axis=1))
    return float(np.mean(np.sqrt(np.maximum(0.0, 1.0 - np.clip(overlaps, 0.0, 1.0) ** 2))))


def _score_mode(
    *, mode: str, all_gates: Sequence[GateOp], baseline: Sequence[GateOp], correct: np.ndarray,
    fashion: np.ndarray, haar: np.ndarray, n_qubits: int
) -> Dict[int, float]:
    if mode == "correct_mnist49_prefix":
        prefixes = _compute_prefix_states_fast(baseline, correct, n_qubits)
    elif mode == "mismatched_fashion_prefix":
        prefixes = _compute_prefix_states_fast(baseline, fashion, n_qubits)
    elif mode == "haar_prefix":
        prefixes = _compute_prefix_states_fast(baseline, haar, n_qubits)
    elif mode == "no_prefix":
        prefixes = [correct.copy() for _ in range(len(baseline) + 1)]
    else:
        raise ValueError(mode)
    pools = _pools_by_position(all_gates)
    weights = np.ones(2**n_qubits, dtype=np.float64)
    scores: Dict[int, float] = {}
    for gate in baseline:
        if gate.gate_type != "Rot":
            continue
        position = int(gate.pos_id)
        angles = _weighted_angles(
            gate, _identity_candidate(pools[position]), prefixes[position], weights, n_qubits
        )
        scores[position] = float(np.max(angles))
    return scores


def _bootstrap(values: Sequence[float], seed: int, draws: int = 20000) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {"mean": float(array.mean()), "ci_low": float(np.quantile(sampled, 0.025)), "ci_high": float(np.quantile(sampled, 0.975))}


def _evaluate_seed(
    *, seed: int, snapshot_path: Path, data_dir: Path, n_qubits: int,
    n_calibration: int, n_test: int, M: int, top_ks: Sequence[int]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot = payload["trained_snapshot"]
    parameters = np.asarray(snapshot["trained_parameters"], dtype=np.float64)
    readout_w = np.asarray(snapshot["readout_w"], dtype=np.float64)
    readout_b = float(snapshot["readout_b"])
    all_gates, baseline = build_hea_gates_from_parameters(
        parameters, include_cnot=True, include_identity_rot=True
    )

    cfg = ToyVisionConfig(
        n_qubits=n_qubits, data_dir=str(data_dir), seed=seed, download=False,
        n_train_pool=512, n_val_pool=n_calibration + n_test,
    )
    _, _, mnist_val, mnist_labels = build_mnist49_splits(cfg)
    calibration, calibration_labels, test, test_labels = stratified_calibration_test_split(
        mnist_val, mnist_labels, n_calibration=n_calibration, n_test=n_test, seed=seed + 20
    )
    correct, _ = stratified_subsample(calibration, calibration_labels, M, seed + 30)
    _, _, fashion_val, fashion_labels = build_fashion_sandal_boot_splits(cfg)
    fashion, _ = stratified_subsample(fashion_val, fashion_labels, M, seed + 31)
    haar = _haar_states(seed, M, 2**n_qubits)

    baseline_logits, baseline_outputs = _logits(baseline, test, readout_w, readout_b, n_qubits)
    baseline_predictions = (baseline_logits >= 0.0).astype(int)
    baseline_accuracy = float(np.mean(baseline_predictions == test_labels))
    if abs(baseline_accuracy - float(snapshot["acc_trained"])) > 1e-12:
        raise RuntimeError(
            f"Fast baseline accuracy mismatch seed={seed}: {baseline_accuracy} vs {snapshot['acc_trained']}"
        )

    positions = sorted(int(gate.pos_id) for gate in baseline if gate.gate_type == "Rot")
    pools = _pools_by_position(all_gates)
    truth: Dict[int, Dict[str, float]] = {}
    for position in positions:
        selected = list(baseline)
        selected[position] = _identity_candidate(pools[position])
        logits, outputs = _logits(selected, test, readout_w, readout_b, n_qubits)
        predictions = (logits >= 0.0).astype(int)
        accuracy = float(np.mean(predictions == test_labels))
        truth[position] = {
            "mean_abs_logit_drift": float(np.mean(np.abs(logits - baseline_logits))),
            "prediction_flip_rate": float(np.mean(predictions != baseline_predictions)),
            "abs_accuracy_drift": abs(accuracy - baseline_accuracy),
            "mean_trace_distance": _mean_trace_distance(baseline_outputs, outputs),
        }

    mode_scores = {
        mode: _score_mode(
            mode=mode, all_gates=all_gates, baseline=baseline, correct=correct,
            fashion=fashion, haar=haar, n_qubits=n_qubits
        )
        for mode in MODES
    }
    target = [truth[position]["mean_abs_logit_drift"] for position in positions]
    position_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    for mode, scores in mode_scores.items():
        vector = [scores[position] for position in positions]
        order = sorted(positions, key=lambda position: (scores[position], position))
        row: Dict[str, Any] = {
            "seed": seed, "mode": mode, "positions": len(positions),
            "baseline_accuracy": baseline_accuracy,
            "spearman_logit": _correlation(vector, target, "spearman"),
            "kendall_logit": _correlation(vector, target, "kendall"),
            "spearman_trace": _correlation(
                vector, [truth[position]["mean_trace_distance"] for position in positions], "spearman"
            ),
        }
        for top_k in top_ks:
            selected = order[: min(top_k, len(order))]
            values = [truth[position]["mean_abs_logit_drift"] for position in selected]
            oracle = sorted(target)[: len(selected)]
            row[f"top{top_k}_positions"] = ",".join(str(value) for value in selected)
            row[f"top{top_k}_mean_logit_drift"] = float(np.mean(values))
            row[f"top{top_k}_regret"] = float(np.mean(values) - np.mean(oracle))
        seed_rows.append(row)
        for position in positions:
            position_rows.append(
                {"seed": seed, "mode": mode, "position": position, "score": scores[position], **truth[position]}
            )
    return position_rows, seed_rows, {
        "seed": seed, "baseline_accuracy": baseline_accuracy,
        "snapshot": str(snapshot_path.resolve()), "test_samples": len(test), "calibration_M": M,
    }


def _summarize(rows: Sequence[Mapping[str, Any]], top_ks: Sequence[int]) -> List[Dict[str, Any]]:
    correct = {int(row["seed"]): row for row in rows if row["mode"] == "correct_mnist49_prefix"}
    metrics = ["spearman_logit", "kendall_logit", "spearman_trace"]
    metrics.extend(f"top{top_k}_mean_logit_drift" for top_k in top_ks)
    metrics.extend(f"top{top_k}_regret" for top_k in top_ks)
    output: List[Dict[str, Any]] = []
    for mode_index, mode in enumerate(MODES):
        group = [row for row in rows if row["mode"] == mode]
        item: Dict[str, Any] = {"mode": mode, "seeds": len(group)}
        for metric_index, metric in enumerate(metrics):
            values = [float(row[metric]) for row in group]
            estimate = _bootstrap(values, 20260811 + 100 * mode_index + metric_index)
            differences = [float(row[metric]) - float(correct[int(row["seed"])][metric]) for row in group]
            paired = _bootstrap(differences, 20261811 + 100 * mode_index + metric_index)
            for key, value in estimate.items():
                item[f"{metric}_{key}"] = value
            item[f"{metric}_paired_vs_correct"] = paired["mean"]
            item[f"{metric}_paired_vs_correct_ci_low"] = paired["ci_low"]
            item[f"{metric}_paired_vs_correct_ci_high"] = paired["ci_high"]
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST49 task-conditioning ablation.")
    parser.add_argument("--snapshot_dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--n_calibration", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--top_ks", nargs="+", type=int, default=[2, 5])
    args = parser.parse_args()

    positions: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    seed_meta: List[Dict[str, Any]] = []
    for seed in args.seeds:
        p, s, m = _evaluate_seed(
            seed=seed, snapshot_path=args.snapshot_dir.resolve() / f"mnist49_seed{seed}.json",
            data_dir=args.data_dir.resolve(), n_qubits=args.n_qubits,
            n_calibration=args.n_calibration, n_test=args.n_test, M=args.M, top_ks=args.top_ks,
        )
        positions.extend(p)
        seed_rows.extend(s)
        seed_meta.append(m)
        print(f"[OK] seed={seed} baseline_acc={m['baseline_accuracy']:.6f}", flush=True)
    summary = _summarize(seed_rows, args.top_ks)
    output_dir = args.output_dir.resolve()
    _atomic_csv(output_dir / "position_level.csv", positions)
    _atomic_csv(output_dir / "seed_level.csv", seed_rows)
    _atomic_csv(output_dir / "summary_by_mode.csv", summary)
    _atomic_json(
        output_dir / "manifest.json",
        {
            "experiment": "mnist49_task_conditioning_ablation", "seeds": args.seeds,
            "modes": list(MODES), "q": 1.0, "primary_ground_truth": "mean absolute test-logit drift",
            "secondary_ground_truths": ["prediction flip rate", "absolute accuracy drift", "mean pure-state trace distance"],
            "correct_calibration_samples": args.M, "mismatched_fashion_samples": args.M,
            "haar_samples": args.M, "test_samples": args.n_test, "no_finetuning": True,
            "seed_meta": seed_meta,
        },
    )
    print(f"Saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
