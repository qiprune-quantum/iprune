from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from iprune.gates import GateOp, build_hea_gates_from_parameters
from scripts.experiments.run_q_matched_k_tfim_seed import _identity_candidate, _pools_by_position
from scripts.experiments.run_q_sensitivity_selection_diagnostic import (
    _compute_prefix_states_fast,
    _weighted_angles,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = (
    PROJECT_DIR
    / "results"
    / "q_sensitivity_upgrade_20260810"
    / "tfim_matched_k"
    / "rows"
)
DEFAULT_GROUND_TRUTH = (
    PROJECT_DIR
    / "results"
    / "q_similarity_validation_20260811"
    / "tfim"
    / "position_level.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "task_conditioning_ablation_20260811" / "tfim"
)
MODES = ("correct_prefix", "mismatched_0101_prefix", "haar_prefix", "no_prefix")


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


def _basis_state(index: int, n_qubits: int) -> np.ndarray:
    state = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    state[0, index] = 1.0
    return state


def _haar_state(seed: int, n_qubits: int) -> np.ndarray:
    rng = np.random.default_rng(20260811 + int(seed))
    vector = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    vector /= np.linalg.norm(vector)
    return vector.reshape(1, -1).astype(np.complex128)


def _load_ground_truth(path: Path) -> Dict[int, Dict[int, Dict[str, float]]]:
    rows: Dict[int, Dict[int, Dict[str, float]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if float(row["qbeta"]) != 0.0:
                continue
            seed = int(row["seed"])
            position = int(row["position"])
            rows.setdefault(seed, {})[position] = {
                "abs_energy_drift": float(row["abs_energy_drift"]),
                "trace_distance": float(row["trace_distance"]),
            }
    return rows


def _correlation(x: Sequence[float], y: Sequence[float], kind: str) -> float:
    value = spearmanr(x, y).statistic if kind == "spearman" else kendalltau(x, y).statistic
    return float(value) if np.isfinite(value) else 0.0


def _bootstrap(values: Sequence[float], seed: int, draws: int = 20000) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }


def _mode_prefixes(
    mode: str,
    baseline_circuit: Sequence[GateOp],
    correct: np.ndarray,
    mismatched: np.ndarray,
    haar: np.ndarray,
    n_qubits: int,
) -> List[np.ndarray]:
    if mode == "correct_prefix":
        return _compute_prefix_states_fast(baseline_circuit, correct, n_qubits)
    if mode == "mismatched_0101_prefix":
        return _compute_prefix_states_fast(baseline_circuit, mismatched, n_qubits)
    if mode == "haar_prefix":
        return _compute_prefix_states_fast(baseline_circuit, haar, n_qubits)
    if mode == "no_prefix":
        return [correct.copy() for _ in range(len(baseline_circuit) + 1)]
    raise ValueError(mode)


def _score_mode(
    *,
    mode: str,
    all_gates: Sequence[GateOp],
    baseline_circuit: Sequence[GateOp],
    correct: np.ndarray,
    mismatched: np.ndarray,
    haar: np.ndarray,
    n_qubits: int,
) -> Dict[int, float]:
    prefixes = _mode_prefixes(
        mode, baseline_circuit, correct, mismatched, haar, n_qubits
    )
    pools = _pools_by_position(all_gates)
    weights = np.ones(2**n_qubits, dtype=np.float64)
    scores: Dict[int, float] = {}
    for gate in baseline_circuit:
        if gate.gate_type != "Rot":
            continue
        position = int(gate.pos_id)
        identity = _identity_candidate(pools[position])
        angles = _weighted_angles(
            gate, identity, prefixes[position], weights, n_qubits
        )
        scores[position] = float(np.max(angles))
    return scores


def _evaluate_seed(
    *,
    seed: int,
    source_path: Path,
    truth: Mapping[int, Mapping[str, float]],
    n_qubits: int,
    top_ks: Sequence[int],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    parameters = np.asarray(source["optimized_parameters"], dtype=np.float64)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        parameters, include_cnot=True, include_identity_rot=True
    )
    correct = _basis_state(0, n_qubits)
    mismatched = _basis_state(int("0101", 2), n_qubits)
    haar = _haar_state(seed, n_qubits)
    mode_scores = {
        mode: _score_mode(
            mode=mode,
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            correct=correct,
            mismatched=mismatched,
            haar=haar,
            n_qubits=n_qubits,
        )
        for mode in MODES
    }
    positions = sorted(truth)
    if any(set(scores) != set(positions) for scores in mode_scores.values()):
        raise RuntimeError(f"Position mismatch for seed={seed}")
    energy = [float(truth[position]["abs_energy_drift"]) for position in positions]
    trace = [float(truth[position]["trace_distance"]) for position in positions]

    position_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    for mode, scores in mode_scores.items():
        score_vector = [scores[position] for position in positions]
        order = sorted(positions, key=lambda position: (scores[position], position))
        row: Dict[str, Any] = {
            "seed": seed,
            "mode": mode,
            "positions": len(positions),
            "spearman_energy": _correlation(score_vector, energy, "spearman"),
            "kendall_energy": _correlation(score_vector, energy, "kendall"),
            "spearman_trace": _correlation(score_vector, trace, "spearman"),
            "kendall_trace": _correlation(score_vector, trace, "kendall"),
        }
        for top_k in top_ks:
            selected = order[: min(int(top_k), len(order))]
            selected_energy = [float(truth[position]["abs_energy_drift"]) for position in selected]
            oracle = sorted(energy)[: len(selected)]
            row[f"top{top_k}_positions"] = ",".join(str(value) for value in selected)
            row[f"top{top_k}_mean_energy_drift"] = float(np.mean(selected_energy))
            row[f"top{top_k}_regret"] = float(np.mean(selected_energy) - np.mean(oracle))
        seed_rows.append(row)
        for position in positions:
            position_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "position": position,
                    "score": scores[position],
                    "abs_energy_drift": float(truth[position]["abs_energy_drift"]),
                    "trace_distance": float(truth[position]["trace_distance"]),
                }
            )
    return position_rows, seed_rows


def _summarize(seed_rows: Sequence[Mapping[str, Any]], top_ks: Sequence[int]) -> List[Dict[str, Any]]:
    correct = {
        int(row["seed"]): row for row in seed_rows if row["mode"] == "correct_prefix"
    }
    metrics = ["spearman_energy", "kendall_energy", "spearman_trace", "kendall_trace"]
    metrics.extend(f"top{top_k}_mean_energy_drift" for top_k in top_ks)
    metrics.extend(f"top{top_k}_regret" for top_k in top_ks)
    summary: List[Dict[str, Any]] = []
    for mode_index, mode in enumerate(MODES):
        rows = [row for row in seed_rows if row["mode"] == mode]
        item: Dict[str, Any] = {"mode": mode, "seeds": len(rows)}
        for metric_index, metric in enumerate(metrics):
            values = [float(row[metric]) for row in rows]
            estimate = _bootstrap(values, 20260811 + 100 * mode_index + metric_index)
            item[f"{metric}_mean"] = estimate["mean"]
            item[f"{metric}_ci_low"] = estimate["ci_low"]
            item[f"{metric}_ci_high"] = estimate["ci_high"]
            differences = [
                float(row[metric]) - float(correct[int(row["seed"])][metric])
                for row in rows
            ]
            paired = _bootstrap(differences, 20261811 + 100 * mode_index + metric_index)
            item[f"{metric}_paired_vs_correct"] = paired["mean"]
            item[f"{metric}_paired_vs_correct_ci_low"] = paired["ci_low"]
            item[f"{metric}_paired_vs_correct_ci_high"] = paired["ci_high"]
        summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="TFIM task-conditioning state ablation.")
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--ground_truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--top_ks", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--n_qubits", type=int, default=4)
    args = parser.parse_args()

    truth_all = _load_ground_truth(args.ground_truth.resolve())
    position_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        if seed not in truth_all:
            raise ValueError(f"Missing ground truth for seed={seed}")
        source_path = args.source_dir.resolve() / f"tfim_seed{seed}.json"
        seed_positions, seed_summary = _evaluate_seed(
            seed=seed,
            source_path=source_path,
            truth=truth_all[seed],
            n_qubits=args.n_qubits,
            top_ks=args.top_ks,
        )
        position_rows.extend(seed_positions)
        seed_rows.extend(seed_summary)
        print(f"[OK] seed={seed}", flush=True)

    summary_rows = _summarize(seed_rows, args.top_ks)
    output_dir = args.output_dir.resolve()
    _atomic_csv(output_dir / "position_level.csv", position_rows)
    _atomic_csv(output_dir / "seed_level.csv", seed_rows)
    _atomic_csv(output_dir / "summary_by_mode.csv", summary_rows)
    _atomic_json(
        output_dir / "manifest.json",
        {
            "experiment": "tfim_task_conditioning_state_ablation",
            "seeds": args.seeds,
            "modes": list(MODES),
            "q": 1.0,
            "primary_ground_truth": "absolute TFIM energy drift after actual single-gate deletion",
            "secondary_ground_truth": "full-circuit pure-state trace distance",
            "correct_state": "|0000>",
            "mismatched_state": "|0101>",
            "haar_policy": "one deterministic complex Haar state per seed, RNG seed 20260811+seed",
            "no_prefix_policy": "score every position on the unpropagated |0000> state",
            "ground_truth_source": str(args.ground_truth.resolve()),
            "source_dir": str(args.source_dir.resolve()),
            "no_finetuning": True,
            "position_rows": len(position_rows),
            "seed_rows": len(seed_rows),
        },
    )
    print(f"Saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
