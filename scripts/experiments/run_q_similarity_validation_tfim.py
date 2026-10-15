from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pennylane as qml
from scipy.stats import kendalltau, spearmanr

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs
from iprune.experiments.tasks_vqe import tfim_hamiltonian
from iprune.gates import GateOp, build_hea_gates_from_parameters
from scripts.experiments.run_q_matched_k_tfim_seed import _compact_without, _score_positions


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = (
    PROJECT_DIR
    / "results"
    / "q_sensitivity_upgrade_20260810"
    / "tfim_matched_k"
    / "rows"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "q_similarity_validation_20260811" / "tfim"
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
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


def _pure_trace_distance(psi: np.ndarray, phi: np.ndarray) -> float:
    overlap = abs(np.vdot(psi.reshape(-1), phi.reshape(-1)))
    overlap = float(np.clip(overlap, 0.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - overlap * overlap)))


def _state_energy(state: np.ndarray, hamiltonian_matrix: np.ndarray) -> float:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    return float(np.real(np.vdot(vector, hamiltonian_matrix @ vector)))


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, draws: int = 20000
) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }


def _finite_correlation(x: Sequence[float], y: Sequence[float], kind: str) -> float:
    if kind == "spearman":
        value = spearmanr(x, y).statistic
    elif kind == "kendall":
        value = kendalltau(x, y).statistic
    else:
        raise ValueError(kind)
    return float(value) if np.isfinite(value) else 0.0


def _seed_source(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "optimized_parameters" not in payload:
        raise ValueError(f"Missing optimized_parameters: {path}")
    return payload


def _q_value(qbeta: float, n_qubits: int, gamma: float, alpha: float) -> float:
    cfg = PaperProtocolConfig(
        n_qubits=n_qubits,
        gamma=gamma,
        alpha=alpha,
        qbeta=qbeta,
        delta=1.0,
    )
    _, q, _, _, _ = make_q_geometry(cfg)
    return float(q)


def _evaluate_seed(
    *,
    source_path: Path,
    qbetas: Sequence[float],
    n_qubits: int,
    gamma: float,
    alpha: float,
    J: float,
    h: float,
    device: str,
    top_ks: Sequence[int],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    payload = _seed_source(source_path)
    seed = int(payload["meta"]["seed"])
    parameters = np.asarray(payload["optimized_parameters"], dtype=np.float64)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        parameters, include_cnot=True, include_identity_rot=True
    )
    initial_state = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    baseline_output = simulate_outputs(
        baseline_circuit, initial_state, n_qubits, device
    )[0]
    hamiltonian_matrix = np.asarray(
        qml.matrix(tfim_hamiltonian(n_qubits, J=J, h=h), wire_order=range(n_qubits)),
        dtype=np.complex128,
    )
    baseline_energy = _state_energy(baseline_output, hamiltonian_matrix)

    q_scores, _, score_rows = _score_positions(
        all_gates=all_gates,
        baseline_circuit=baseline_circuit,
        prefix_circuit=baseline_circuit,
        initial_state=initial_state,
        n_qubits=n_qubits,
        qbetas=qbetas,
        gamma=gamma,
        alpha=alpha,
    )
    score_lookup = {
        (float(row["qbeta"]), int(row["position"])): row for row in score_rows
    }

    position_rows: List[Dict[str, Any]] = []
    true_impacts: Dict[int, Dict[str, float]] = {}
    rot_gates = [gate for gate in baseline_circuit if gate.gate_type == "Rot"]
    for gate in rot_gates:
        position = int(gate.pos_id)
        deleted_output = simulate_outputs(
            _compact_without(baseline_circuit, [position]),
            initial_state,
            n_qubits,
            device,
        )[0]
        deleted_energy = _state_energy(deleted_output, hamiltonian_matrix)
        trace_distance = _pure_trace_distance(baseline_output, deleted_output)
        true_impacts[position] = {
            "abs_energy_drift": abs(deleted_energy - baseline_energy),
            "trace_distance": trace_distance,
        }
        for qbeta in qbetas:
            score = score_lookup[(float(qbeta), position)]
            position_rows.append(
                {
                    "seed": seed,
                    "position": position,
                    "layer": int(gate.layer),
                    "wire": int(gate.wires[0]),
                    "qbeta": float(qbeta),
                    "q": float(score["q"]),
                    "q_score_max": float(score["q_score_max"]),
                    "q_score_mean": float(score["q_score_mean"]),
                    "physical_angle_max": float(score["physical_score_max"]),
                    "baseline_energy": baseline_energy,
                    "deleted_energy": deleted_energy,
                    "signed_energy_drift": deleted_energy - baseline_energy,
                    "abs_energy_drift": abs(deleted_energy - baseline_energy),
                    "trace_distance": trace_distance,
                    "q1_trace_identity_error": (
                        abs(math.sin(float(score["q_score_max"])) - trace_distance)
                        if float(qbeta) == 0.0
                        else None
                    ),
                }
            )

    seed_rows: List[Dict[str, Any]] = []
    positions = sorted(true_impacts)
    energy_target = [true_impacts[position]["abs_energy_drift"] for position in positions]
    trace_target = [true_impacts[position]["trace_distance"] for position in positions]
    for qbeta in qbetas:
        score = [q_scores[float(qbeta)][position] for position in positions]
        order = sorted(positions, key=lambda position: (q_scores[float(qbeta)][position], position))
        base = {
            "seed": seed,
            "qbeta": float(qbeta),
            "q": _q_value(float(qbeta), n_qubits, gamma, alpha),
            "positions": len(positions),
            "spearman_energy": _finite_correlation(score, energy_target, "spearman"),
            "kendall_energy": _finite_correlation(score, energy_target, "kendall"),
            "spearman_trace": _finite_correlation(score, trace_target, "spearman"),
            "kendall_trace": _finite_correlation(score, trace_target, "kendall"),
        }
        for top_k in top_ks:
            selected = order[: min(int(top_k), len(order))]
            selected_energy = [true_impacts[position]["abs_energy_drift"] for position in selected]
            oracle = sorted(energy_target)[: len(selected)]
            base[f"top{top_k}_mean_energy_drift"] = float(np.mean(selected_energy))
            base[f"top{top_k}_oracle_mean_energy_drift"] = float(np.mean(oracle))
            base[f"top{top_k}_regret"] = float(np.mean(selected_energy) - np.mean(oracle))
            base[f"top{top_k}_positions"] = ",".join(str(value) for value in selected)
        seed_rows.append(base)
    return position_rows, seed_rows


def _summarize(
    seed_rows: Sequence[Mapping[str, Any]], qbetas: Sequence[float], top_ks: Sequence[int]
) -> List[Dict[str, Any]]:
    q1 = {int(row["seed"]): row for row in seed_rows if float(row["qbeta"]) == 0.0}
    summary: List[Dict[str, Any]] = []
    metrics = ["spearman_energy", "kendall_energy", "spearman_trace", "kendall_trace"]
    metrics.extend(f"top{top_k}_mean_energy_drift" for top_k in top_ks)
    metrics.extend(f"top{top_k}_regret" for top_k in top_ks)
    for qbeta in qbetas:
        rows = [row for row in seed_rows if float(row["qbeta"]) == float(qbeta)]
        item: Dict[str, Any] = {
            "qbeta": float(qbeta),
            "q": float(rows[0]["q"]),
            "seeds": len(rows),
        }
        for metric_index, metric in enumerate(metrics):
            values = [float(row[metric]) for row in rows]
            estimate = _bootstrap_mean_ci(
                values, seed=20260811 + 100 * int(qbeta) + metric_index
            )
            item[f"{metric}_mean"] = estimate["mean"]
            item[f"{metric}_ci_low"] = estimate["ci_low"]
            item[f"{metric}_ci_high"] = estimate["ci_high"]
            differences = [
                float(row[metric]) - float(q1[int(row["seed"])][metric]) for row in rows
            ]
            paired = _bootstrap_mean_ci(
                differences, seed=20261811 + 100 * int(qbeta) + metric_index
            )
            item[f"{metric}_paired_vs_q1"] = paired["mean"]
            item[f"{metric}_paired_vs_q1_ci_low"] = paired["ci_low"]
            item[f"{metric}_paired_vs_q1_ci_high"] = paired["ci_high"]
        summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate whether q-distance predicts actual single-gate task impact."
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--qbetas", nargs="+", type=float, default=[0, 1, 3, 6, 10, 20])
    parser.add_argument("--top_ks", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--device", default="default.qubit")
    args = parser.parse_args()

    qbetas = sorted(set(float(value) for value in args.qbetas))
    if 0.0 not in qbetas:
        raise ValueError("qbetas must include 0 (q=1 control)")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    position_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        source = args.source_dir.resolve() / f"tfim_seed{seed}.json"
        if not source.exists():
            raise FileNotFoundError(source)
        seed_position_rows, seed_summary_rows = _evaluate_seed(
            source_path=source,
            qbetas=qbetas,
            n_qubits=args.n_qubits,
            gamma=args.gamma,
            alpha=args.alpha,
            J=args.J,
            h=args.h,
            device=args.device,
            top_ks=args.top_ks,
        )
        position_rows.extend(seed_position_rows)
        seed_rows.extend(seed_summary_rows)
        print(f"[OK] seed={seed} positions={len(seed_position_rows) // len(qbetas)}", flush=True)

    summary_rows = _summarize(seed_rows, qbetas, args.top_ks)
    max_identity_error = max(
        float(row["q1_trace_identity_error"])
        for row in position_rows
        if row["q1_trace_identity_error"] is not None
    )
    manifest = {
        "experiment": "tfim_q_similarity_predictive_validity",
        "scientific_question": "Does q-weighted distance predict held-out full-circuit task impact of a single gate replacement better than q=1?",
        "source_dir": str(args.source_dir.resolve()),
        "seeds": args.seeds,
        "qbetas": qbetas,
        "top_ks": args.top_ks,
        "ground_truth": {
            "state_similarity": "full-circuit pure-state trace distance after deleting exactly one Rot gate",
            "task_similarity": "absolute TFIM energy drift after deleting exactly one Rot gate",
        },
        "important_non_circularity": "The safe/low-impact target is measured by executing the modified full circuit; it is not defined by either q-distance or the physical certificate.",
        "q1_identity_check": "For a single replacement in a noiseless unitary suffix, trace_distance = sin(q=1 angle).",
        "q1_trace_identity_max_abs_error": max_identity_error,
        "position_rows": len(position_rows),
        "seed_summary_rows": len(seed_rows),
    }
    _atomic_csv(output_dir / "position_level.csv", position_rows)
    _atomic_csv(output_dir / "seed_level.csv", seed_rows)
    _atomic_csv(output_dir / "summary_by_q.csv", summary_rows)
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_json(
        output_dir / "results.json",
        {"manifest": manifest, "summary_by_q": summary_rows, "seed_level": seed_rows},
    )
    print(f"Saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
