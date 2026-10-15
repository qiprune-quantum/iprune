from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pennylane as qml
from scipy.stats import spearmanr

from iprune.experiments.protocol import simulate_outputs
from iprune.experiments.tasks_vqe import tfim_hamiltonian
from iprune.gates import GateOp, build_hea_gates_from_parameters, is_identity_rot
from iprune.qgeometry import build_Gq_diag_popcount, epsilon_from_delta
from scripts.experiments.run_q_matched_k_tfim_seed import _compact_without
from scripts.experiments.run_q_sensitivity_selection_diagnostic import (
    _apply_one_qubit,
    _compute_prefix_states_fast,
    _rot_matrix,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = (
    PROJECT_DIR
    / "results"
    / "q_sensitivity_upgrade_20260810"
    / "tfim_matched_k"
    / "rows"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "q_basis_alignment_validation_20260813" / "tfim"
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
    columns = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _source_parameters(source_dir: Path, seed: int) -> np.ndarray:
    path = source_dir / f"tfim_seed{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["optimized_parameters"], dtype=np.float64)


def _pools_by_position(all_gates: Sequence[GateOp]) -> Dict[int, List[GateOp]]:
    pools: Dict[int, List[GateOp]] = {}
    for gate in all_gates:
        pools.setdefault(int(gate.pos_id), []).append(gate)
    return pools


def _identity_candidate(pool: Sequence[GateOp]) -> GateOp:
    identities = [gate for gate in pool if is_identity_rot(gate)]
    if len(identities) != 1:
        raise ValueError("Each rotation position must have exactly one identity candidate")
    return identities[0]


def _hadamard_matrix(n_qubits: int) -> np.ndarray:
    single = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    result = np.asarray([[1.0]], dtype=np.complex128)
    for _ in range(n_qubits):
        result = np.kron(result, single)
    return result


def _basis_weighted_angles(
    reference: GateOp,
    candidate: GateOp,
    states: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
    basis_transform: np.ndarray,
) -> np.ndarray:
    wire = int(reference.wires[0])
    out_reference = _apply_one_qubit(states, _rot_matrix(reference), wire, n_qubits)
    out_candidate = _apply_one_qubit(states, _rot_matrix(candidate), wire, n_qubits)
    if basis_transform.shape != (2**n_qubits, 2**n_qubits):
        raise ValueError("Invalid basis transform shape")
    transformed_reference = out_reference @ basis_transform.T
    transformed_candidate = out_candidate @ basis_transform.T
    weighted = np.asarray(weights, dtype=np.float64).reshape(1, -1)
    numerator = np.abs(
        np.sum(
            np.conjugate(transformed_reference)
            * weighted
            * transformed_candidate,
            axis=1,
        )
    )
    norm_reference = np.real(
        np.sum(
            np.conjugate(transformed_reference) * weighted * transformed_reference,
            axis=1,
        )
    )
    norm_candidate = np.real(
        np.sum(
            np.conjugate(transformed_candidate) * weighted * transformed_candidate,
            axis=1,
        )
    )
    denominator = np.sqrt(np.maximum(norm_reference * norm_candidate, 0.0))
    if np.any(denominator <= 1e-12):
        raise ValueError("Weighted norm too small")
    return np.arccos(np.clip(numerator / denominator, 0.0, 1.0))


def _state_energy(state: np.ndarray, hamiltonian: np.ndarray) -> float:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    return float(np.real(np.vdot(vector, hamiltonian @ vector)))


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    overlap = float(np.clip(abs(np.vdot(left.reshape(-1), right.reshape(-1))), 0.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - overlap * overlap)))


def _finite_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, draws: int = 20000
) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _sign_flip_p(values: Sequence[float]) -> float:
    array = np.asarray([value for value in values if abs(value) > 1e-12], dtype=np.float64)
    if len(array) == 0:
        return 1.0
    observed = abs(float(array.mean()))
    means = []
    for mask in range(2 ** len(array)):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(len(array))]
        )
        means.append(abs(float(np.mean(array * signs))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def _method_key(basis: str, q: float) -> str:
    return f"{basis}:q={q:.9f}"


def _symmetric_method_key(basis: str, q_high: float) -> str:
    return f"{basis}:sym_q={q_high:.9f}"


def _evaluate_seed(
    *,
    seed: int,
    source_dir: Path,
    q_values: Sequence[float],
    bases: Sequence[str],
    delta: float,
    target_k: int,
    n_qubits: int,
    J: float,
    h: float,
    device: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    parameters = _source_parameters(source_dir, seed)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        parameters, include_cnot=True, include_identity_rot=True
    )
    pools = _pools_by_position(all_gates)
    initial_state = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    prefixes = _compute_prefix_states_fast(baseline_circuit, initial_state, n_qubits)
    baseline_output = simulate_outputs(
        baseline_circuit, initial_state, n_qubits, device
    )[0]
    hamiltonian = np.asarray(
        qml.matrix(
            tfim_hamiltonian(n_qubits, J=J, h=h), wire_order=range(n_qubits)
        ),
        dtype=np.complex128,
    )
    baseline_energy = _state_energy(baseline_output, hamiltonian)
    identity_transform = np.eye(2**n_qubits, dtype=np.complex128)
    transforms = {
        "z": identity_transform,
        "x": _hadamard_matrix(n_qubits),
    }
    requested = set(bases)
    if not requested <= transforms.keys():
        raise ValueError(f"Unsupported bases: {sorted(requested - transforms.keys())}")
    epsilon = epsilon_from_delta(delta, 1.0)

    physical_scores: Dict[int, float] = {}
    q_scores: Dict[str, Dict[int, float]] = {
        _method_key(basis, q): {} for basis in bases for q in q_values
    }
    position_rows: List[Dict[str, Any]] = []
    true_impacts: Dict[int, Dict[str, float]] = {}

    for gate in baseline_circuit:
        if gate.gate_type != "Rot":
            continue
        position = int(gate.pos_id)
        identity = _identity_candidate(pools[position])
        physical_angles = _basis_weighted_angles(
            gate,
            identity,
            prefixes[position],
            np.ones(2**n_qubits, dtype=np.float64),
            n_qubits,
            identity_transform,
        )
        physical_scores[position] = float(np.max(physical_angles))
        deleted_output = simulate_outputs(
            _compact_without(baseline_circuit, [position]),
            initial_state,
            n_qubits,
            device,
        )[0]
        deleted_energy = _state_energy(deleted_output, hamiltonian)
        true_impacts[position] = {
            "energy": abs(deleted_energy - baseline_energy),
            "trace": _trace_distance(baseline_output, deleted_output),
        }
        for basis in bases:
            transform = transforms[basis]
            for q in q_values:
                weights, _ = build_Gq_diag_popcount(q, n_qubits)
                angles = _basis_weighted_angles(
                    gate,
                    identity,
                    prefixes[position],
                    weights,
                    n_qubits,
                    transform,
                )
                key = _method_key(basis, q)
                score = float(np.max(angles))
                q_scores[key][position] = score
                position_rows.append(
                    {
                        "seed": seed,
                        "basis": basis,
                        "q": q,
                        "position": position,
                        "wire": int(gate.wires[0]),
                        "layer": int(gate.layer),
                        "q_score": score,
                        "physical_score": physical_scores[position],
                        "abs_single_energy_drift": true_impacts[position]["energy"],
                        "single_trace_distance": true_impacts[position]["trace"],
                    }
                )

    eligible = sorted(
        position for position, score in physical_scores.items() if score <= epsilon + 1e-12
    )
    matched_k = min(target_k, len(eligible))
    positions = sorted(true_impacts)
    energy_target = [true_impacts[position]["energy"] for position in positions]
    trace_target = [true_impacts[position]["trace"] for position in positions]
    method_rows: List[Dict[str, Any]] = []
    for basis in bases:
        for q in q_values:
            key = _method_key(basis, q)
            scores = q_scores[key]
            order = sorted(positions, key=lambda position: (scores[position], position))
            eligible_order = [position for position in order if position in eligible]
            selected = eligible_order[:matched_k]
            deleted_output = simulate_outputs(
                _compact_without(baseline_circuit, selected),
                initial_state,
                n_qubits,
                device,
            )[0]
            deleted_energy = _state_energy(deleted_output, hamiltonian)
            top2 = order[:2]
            top5 = order[:5]
            method_rows.append(
                {
                    "seed": seed,
                    "basis": basis,
                    "q": q,
                    "method_key": key,
                    "baseline_energy": baseline_energy,
                    "physical_delta": delta,
                    "physical_epsilon": epsilon,
                    "physical_eligible_count": len(eligible),
                    "target_k": target_k,
                    "matched_k": matched_k,
                    "selected_positions": ",".join(str(value) for value in selected),
                    "spearman_single_energy": _finite_spearman(
                        [scores[position] for position in positions], energy_target
                    ),
                    "spearman_single_trace": _finite_spearman(
                        [scores[position] for position in positions], trace_target
                    ),
                    "top2_mean_single_energy": float(
                        np.mean([true_impacts[position]["energy"] for position in top2])
                    ),
                    "top5_mean_single_energy": float(
                        np.mean([true_impacts[position]["energy"] for position in top5])
                    ),
                    "matched_k_abs_energy_drift": abs(deleted_energy - baseline_energy),
                    "matched_k_trace_distance": _trace_distance(
                        baseline_output, deleted_output
                    ),
                }
            )

        # Encoding-invariant diagnostic: exchanging the computational labels
        # |0> and |1> maps q to 1/q.  The max score removes that arbitrary
        # orientation while preserving the same pre-fixed |log(q)| grid.
        q_lookup = {round(math.log(q), 12): float(q) for q in q_values}
        for log_q, q_high in sorted(q_lookup.items()):
            if log_q <= 1e-12 or round(-log_q, 12) not in q_lookup:
                continue
            q_low = q_lookup[round(-log_q, 12)]
            high_scores = q_scores[_method_key(basis, q_high)]
            low_scores = q_scores[_method_key(basis, q_low)]
            scores = {
                position: max(high_scores[position], low_scores[position])
                for position in positions
            }
            key = _symmetric_method_key(basis, q_high)
            order = sorted(positions, key=lambda position: (scores[position], position))
            eligible_order = [position for position in order if position in eligible]
            selected = eligible_order[:matched_k]
            deleted_output = simulate_outputs(
                _compact_without(baseline_circuit, selected),
                initial_state,
                n_qubits,
                device,
            )[0]
            deleted_energy = _state_energy(deleted_output, hamiltonian)
            top2 = order[:2]
            top5 = order[:5]
            method_rows.append(
                {
                    "seed": seed,
                    "basis": basis,
                    "q": q_high,
                    "q_reciprocal": q_low,
                    "symmetric_q_score": True,
                    "method_key": key,
                    "baseline_energy": baseline_energy,
                    "physical_delta": delta,
                    "physical_epsilon": epsilon,
                    "physical_eligible_count": len(eligible),
                    "target_k": target_k,
                    "matched_k": matched_k,
                    "selected_positions": ",".join(str(value) for value in selected),
                    "spearman_single_energy": _finite_spearman(
                        [scores[position] for position in positions], energy_target
                    ),
                    "spearman_single_trace": _finite_spearman(
                        [scores[position] for position in positions], trace_target
                    ),
                    "top2_mean_single_energy": float(
                        np.mean([true_impacts[position]["energy"] for position in top2])
                    ),
                    "top5_mean_single_energy": float(
                        np.mean([true_impacts[position]["energy"] for position in top5])
                    ),
                    "matched_k_abs_energy_drift": abs(deleted_energy - baseline_energy),
                    "matched_k_trace_distance": _trace_distance(
                        baseline_output, deleted_output
                    ),
                }
            )
    return position_rows, method_rows


def _aggregate(
    rows: Sequence[Mapping[str, Any]], seeds: Iterable[int], *, bootstrap_seed: int
) -> List[Dict[str, Any]]:
    seed_set = set(int(seed) for seed in seeds)
    selected = [row for row in rows if int(row["seed"]) in seed_set]
    keys = sorted({str(row["method_key"]) for row in selected})
    metrics = [
        "spearman_single_energy",
        "spearman_single_trace",
        "top2_mean_single_energy",
        "top5_mean_single_energy",
        "matched_k_abs_energy_drift",
        "matched_k_trace_distance",
    ]
    output: List[Dict[str, Any]] = []
    for method_index, key in enumerate(keys):
        group = sorted(
            [row for row in selected if str(row["method_key"]) == key],
            key=lambda row: int(row["seed"]),
        )
        base: Dict[str, Any] = {
            "method_key": key,
            "basis": group[0]["basis"],
            "q": float(group[0]["q"]),
            "seeds": len(group),
            "selection_changed_vs_q1": 0,
        }
        q1 = {
            int(row["seed"]): str(row["selected_positions"])
            for row in selected
            if str(row["basis"]) == "z" and math.isclose(float(row["q"]), 1.0)
        }
        base["selection_changed_vs_q1"] = sum(
            str(row["selected_positions"]) != q1[int(row["seed"])] for row in group
        )
        for metric_index, metric in enumerate(metrics):
            mean, low, high = _bootstrap_mean_ci(
                [float(row[metric]) for row in group],
                seed=bootstrap_seed + 100 * method_index + metric_index,
            )
            base[f"{metric}_mean"] = mean
            base[f"{metric}_ci_low"] = low
            base[f"{metric}_ci_high"] = high
        output.append(base)
    return output


def _comparison(
    rows: Sequence[Mapping[str, Any]], method_key: str, seeds: Iterable[int]
) -> Dict[str, Any]:
    seed_set = set(int(seed) for seed in seeds)
    method = {
        int(row["seed"]): row
        for row in rows
        if int(row["seed"]) in seed_set and str(row["method_key"]) == method_key
    }
    q1 = {
        int(row["seed"]): row
        for row in rows
        if int(row["seed"]) in seed_set
        and str(row["basis"]) == "z"
        and math.isclose(float(row["q"]), 1.0)
    }
    differences = [
        float(method[seed]["matched_k_abs_energy_drift"])
        - float(q1[seed]["matched_k_abs_energy_drift"])
        for seed in sorted(seed_set)
    ]
    mean, low, high = _bootstrap_mean_ci(differences, seed=20260813)
    return {
        "method_key": method_key,
        "baseline": "z:q=1",
        "heldout_seeds": sorted(seed_set),
        "paired_energy_difference_mean": mean,
        "paired_energy_difference_ci_low": low,
        "paired_energy_difference_ci_high": high,
        "wins": sum(value < -1e-12 for value in differences),
        "losses": sum(value > 1e-12 for value in differences),
        "ties": sum(abs(value) <= 1e-12 for value in differences),
        "exact_sign_flip_p": _sign_flip_p(differences),
        "differences": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-registered q magnitude and geometry-basis alignment audit on TFIM."
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--calibration_seeds", nargs="+", type=int, default=list(range(42, 47)))
    parser.add_argument("--test_seeds", nargs="+", type=int, default=list(range(47, 52)))
    parser.add_argument(
        "--q_values",
        nargs="+",
        type=float,
        default=[math.exp(value) for value in (-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6)],
    )
    parser.add_argument("--bases", nargs="+", default=["z", "x"])
    parser.add_argument("--delta", type=float, default=1.2)
    parser.add_argument("--target_k", type=int, default=5)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--device", default="default.qubit")
    args = parser.parse_args()

    seeds = sorted(set(args.seeds))
    calibration_seeds = sorted(set(args.calibration_seeds))
    test_seeds = sorted(set(args.test_seeds))
    if set(calibration_seeds) & set(test_seeds):
        raise ValueError("Calibration and test seeds must be disjoint")
    if set(calibration_seeds + test_seeds) != set(seeds):
        raise ValueError("Calibration and test seeds must partition --seeds")
    q_values = sorted(set(float(value) for value in args.q_values))
    if not any(math.isclose(value, 1.0) for value in q_values):
        raise ValueError("q grid must contain q=1")

    output_dir = args.output_dir.resolve()
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    all_position_rows: List[Dict[str, Any]] = []
    all_method_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        position_rows, method_rows = _evaluate_seed(
            seed=seed,
            source_dir=args.source_dir.resolve(),
            q_values=q_values,
            bases=args.bases,
            delta=args.delta,
            target_k=args.target_k,
            n_qubits=args.n_qubits,
            J=args.J,
            h=args.h,
            device=args.device,
        )
        _atomic_json(
            rows_dir / f"tfim_seed{seed}.json",
            {"seed": seed, "position_rows": position_rows, "method_rows": method_rows},
        )
        all_position_rows.extend(position_rows)
        all_method_rows.extend(method_rows)
        print(f"[OK] seed={seed} methods={len(method_rows)}", flush=True)

    calibration_summary = _aggregate(
        all_method_rows, calibration_seeds, bootstrap_seed=20260813
    )
    test_summary = _aggregate(all_method_rows, test_seeds, bootstrap_seed=20261813)
    q1_key = _method_key("z", 1.0)
    eligible_calibration = [
        row
        for row in calibration_summary
        if not math.isclose(float(row["q"]), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ]
    selected = min(
        eligible_calibration,
        key=lambda row: (
            float(row["matched_k_abs_energy_drift_mean"]),
            abs(math.log(float(row["q"]))),
            str(row["basis"]),
        ),
    )
    comparison = _comparison(all_method_rows, str(selected["method_key"]), test_seeds)

    _atomic_csv(output_dir / "position_level.csv", all_position_rows)
    _atomic_csv(output_dir / "seed_method_level.csv", all_method_rows)
    _atomic_csv(output_dir / "calibration_summary.csv", calibration_summary)
    _atomic_csv(output_dir / "test_summary.csv", test_summary)
    payload = {
        "protocol": {
            "design": "symmetric direct-q grid x pre-fixed Z/X Hamming bases",
            "q_values": q_values,
            "bases": args.bases,
            "delta": args.delta,
            "target_k": args.target_k,
            "calibration_seeds": calibration_seeds,
            "test_seeds": test_seeds,
            "primary_metric": "matched-K one-shot absolute TFIM energy drift",
            "selection_rule": "lowest calibration-seed mean primary metric; tie-break by |log(q)| then basis",
            "no_finetuning": True,
            "interpretation": "X basis is a mechanism-alignment diagnostic, not the original paper metric",
        },
        "selected_on_calibration": selected,
        "heldout_comparison_vs_q1": comparison,
    }
    _atomic_json(output_dir / "RESULTS.json", payload)
    summary = "# TFIM q 大小与几何基底对齐验证\n\n"
    summary += f"- 校准种子：{calibration_seeds}；独立测试种子：{test_seeds}。\n"
    summary += f"- q 网格：{[round(value, 6) for value in q_values]}。\n"
    summary += f"- 基底：{args.bases}；delta={args.delta}；目标 K={args.target_k}；不微调。\n"
    summary += f"- 校准选中：{selected['method_key']}，校准 matched-K 能量漂移均值 {selected['matched_k_abs_energy_drift_mean']:.6f}。\n"
    summary += "- 独立测试相对原始 Z-basis q=1（负值有利）："
    summary += f"{comparison['paired_energy_difference_mean']:+.6f} "
    summary += f"[{comparison['paired_energy_difference_ci_low']:+.6f}, {comparison['paired_energy_difference_ci_high']:+.6f}]；"
    summary += f"W/L/T={comparison['wins']}/{comparison['losses']}/{comparison['ties']}；"
    summary += f"exact p={comparison['exact_sign_flip_p']:.4f}。\n\n"
    summary += "X-Hamming 几何属于机制对齐诊断；只有独立测试稳定改善，才值得作为方法扩展进入后续微调实验。\n"
    (output_dir / "RESULTS_SUMMARY_CN.md").write_text(summary, encoding="utf-8")
    print(f"Saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
