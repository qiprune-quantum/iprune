from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml

from iprune.experiments.protocol import build_random_baseline_circuit
from iprune.experiments.tasks_classification import (
    ToyVisionConfig,
    build_bas_splits,
    build_fashion_sandal_boot_splits,
    build_mnist49_splits,
    sample_task_ensemble_from_validation,
)
from iprune.experiments.tasks_vqe import VQEConfig, sample_intermediate_states_from_vqe
from iprune.gates import GateOp, build_hea_gates
from iprune.qgeometry import build_Gq_diag_popcount, epsilon_from_delta, lambda_from_gamma, q_from_lambda
from scripts.experiments.run_baseline_comparison import _signature


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CHECKPOINTS = [
    PROJECT_DIR
    / "results"
    / "supplementary_upgrade_20260805"
    / "fair_baselines_seed42"
    / "fair_baseline_checkpoint.json",
    PROJECT_DIR
    / "results"
    / "supplementary_upgrade_20260805"
    / "fair_baselines_seed43_46"
    / "fair_baseline_checkpoint.json",
]
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "q_sensitivity_upgrade_20260810"
    / "selection_diagnostic"
)
CONFIG_KEY_FIELDS = (
    "task",
    "dataset",
    "seed",
    "delta",
    "rot_pool_sigma",
)


def _token(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def _config_id(config: Mapping[str, Any]) -> str:
    return (
        f"{config['task']}_{config['dataset']}_seed{config['seed']}_"
        f"delta{_token(float(config['delta']))}_"
        f"sigma{_token(float(config['rot_pool_sigma']))}"
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int_positions(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(part) for part in value.replace(",", " ").split()]
    return [int(position) for position in value]


def load_source_configurations(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Load one q-iPrune and one q=1 row for each historical fair config."""
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for source_path in paths:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            method = str(row.get("baseline_type"))
            if method not in {"qiprune_positionaware", "q1_positionaware_matched_selected_L"}:
                continue
            key = tuple(row[field] for field in CONFIG_KEY_FIELDS)
            entry = grouped.setdefault(key, {"source_checkpoint": str(source_path.resolve())})
            if method in entry:
                raise ValueError(f"Duplicate {method} row for configuration {key}")
            entry[method] = row

    configs: List[Dict[str, Any]] = []
    for key, pair in grouped.items():
        if "qiprune_positionaware" not in pair or "q1_positionaware_matched_selected_L" not in pair:
            raise ValueError(f"Incomplete q-iPrune/q=1 pair for configuration {key}")
        q_row = pair["qiprune_positionaware"]
        q1_row = pair["q1_positionaware_matched_selected_L"]
        config = {
            field: q_row[field]
            for field in (
                "task",
                "dataset",
                "seed",
                "delta",
                "rot_pool_sigma",
                "n_qubits",
                "depth",
                "M",
                "gamma",
                "alpha",
                "qbeta",
                "rot_pool_k",
                "include_cnot",
            )
        }
        config.update(
            {
                "source_checkpoint": pair["source_checkpoint"],
                "source_baseline_signature": str(q_row["baseline_circuit_signature"]),
                "source_qiprune_signature": str(q_row["selected_circuit_signature"]),
                "source_q1_signature": str(q1_row["selected_circuit_signature"]),
                "source_qiprune_positions": _as_int_positions(q_row.get("replaced_positions")),
                "source_q1_positions": _as_int_positions(q1_row.get("replaced_positions")),
                "source_selected_k": int(q_row["selected_replacement_count"]),
            }
        )
        configs.append(config)

    return sorted(
        configs,
        key=lambda item: (
            str(item["task"]),
            str(item["dataset"]),
            int(item["seed"]),
            float(item["delta"]),
            float(item["rot_pool_sigma"]),
        ),
    )


def _rank_positions(scores: Mapping[int, float]) -> List[int]:
    return sorted(scores, key=lambda position: (float(scores[position]), int(position)))


def _rank_correlations(reference_order: Sequence[int], order: Sequence[int]) -> Tuple[float, float]:
    if list(reference_order) == list(order) and len(reference_order) < 2:
        return 1.0, 1.0
    if set(reference_order) != set(order):
        raise ValueError("Rank correlation orders must contain the same positions")
    n = len(reference_order)
    if n < 2:
        return 1.0, 1.0

    reference_rank = {position: rank for rank, position in enumerate(reference_order)}
    rank = {position: index for index, position in enumerate(order)}
    squared_difference = sum(
        (reference_rank[position] - rank[position]) ** 2 for position in reference_order
    )
    spearman = 1.0 - 6.0 * squared_difference / (n * (n * n - 1))

    discordant = 0
    total_pairs = n * (n - 1) // 2
    for left in range(n):
        for right in range(left + 1, n):
            if rank[reference_order[left]] > rank[reference_order[right]]:
                discordant += 1
    kendall = 1.0 - 2.0 * discordant / total_pairs
    return float(spearman), float(kendall)


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return 1.0 if not union else float(len(left_set & right_set) / len(union))


def _boundary_metrics(scores: Mapping[int, float], order: Sequence[int], k: int) -> Dict[str, Any]:
    if k < 0 or k > len(order):
        raise ValueError(f"K={k} is outside ranking length {len(order)}")
    kth_score = float(scores[order[k - 1]]) if k else None
    next_score = float(scores[order[k]]) if k < len(order) else None
    margin = None
    if kth_score is not None and next_score is not None:
        margin = float(next_score - kth_score)
    return {
        "matched_k_score": kth_score,
        "matched_k_next_score": next_score,
        "matched_k_boundary_margin": margin,
    }


def _rot_matrix(gate: GateOp) -> np.ndarray:
    if gate.gate_type != "Rot" or gate.params is None or len(gate.wires) != 1:
        raise ValueError("Selection diagnostics require one-qubit Rot candidate pools")
    return np.asarray(qml.Rot.compute_matrix(*gate.params), dtype=np.complex128)


def _apply_one_qubit(
    states: np.ndarray, matrix: np.ndarray, wire: int, n_qubits: int
) -> np.ndarray:
    shaped = np.asarray(states, dtype=np.complex128).reshape(
        (states.shape[0],) + (2,) * n_qubits
    )
    moved = np.moveaxis(shaped, 1 + wire, -1)
    transformed = np.einsum("ab,...b->...a", matrix, moved)
    restored = np.moveaxis(transformed, -1, 1 + wire)
    return restored.reshape(states.shape)


def _weighted_angles(
    reference: GateOp,
    candidate: GateOp,
    states: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    wire = int(reference.wires[0])
    out_reference = _apply_one_qubit(states, _rot_matrix(reference), wire, n_qubits)
    out_candidate = _apply_one_qubit(states, _rot_matrix(candidate), wire, n_qubits)
    weighted = np.asarray(weights, dtype=np.float64).reshape(1, -1)
    numerator = np.abs(
        np.sum(np.conjugate(out_reference) * weighted * out_candidate, axis=1)
    )
    norm_reference = np.real(
        np.sum(np.conjugate(out_reference) * weighted * out_reference, axis=1)
    )
    norm_candidate = np.real(
        np.sum(np.conjugate(out_candidate) * weighted * out_candidate, axis=1)
    )
    denominator = np.sqrt(np.maximum(norm_reference * norm_candidate, 0.0))
    if np.any(denominator <= 1e-12):
        raise ValueError("q-norm too small while scoring a candidate position")
    return np.arccos(np.clip(numerator / denominator, 0.0, 1.0))


def _apply_cnot(
    states: np.ndarray, control: int, target: int, n_qubits: int
) -> np.ndarray:
    if control == target:
        raise ValueError("CNOT control and target must differ")
    indices = np.arange(2**n_qubits, dtype=np.int64)
    control_mask = 1 << (n_qubits - 1 - control)
    target_mask = 1 << (n_qubits - 1 - target)
    permutation = indices.copy()
    active = (indices & control_mask) != 0
    permutation[active] ^= target_mask
    return np.asarray(states, dtype=np.complex128)[:, permutation]


def _compute_prefix_states_fast(
    circuit: Sequence[GateOp], task_states: np.ndarray, n_qubits: int
) -> List[np.ndarray]:
    """Propagate the whole task ensemble once and retain each circuit prefix."""
    current = np.asarray(task_states, dtype=np.complex128).copy()
    prefixes = [current.copy()]
    for index, gate in enumerate(circuit):
        if int(gate.pos_id) != index:
            raise ValueError("Circuit positions must be consecutive for prefix propagation")
        if gate.gate_type == "Rot":
            current = _apply_one_qubit(
                current, _rot_matrix(gate), int(gate.wires[0]), n_qubits
            )
        elif gate.gate_type == "CNOT":
            current = _apply_cnot(
                current, int(gate.wires[0]), int(gate.wires[1]), n_qubits
            )
        else:
            raise ValueError(f"Unsupported gate type in prefix propagation: {gate.gate_type}")
        prefixes.append(current.copy())
    return prefixes


def _replace_positions(
    baseline: Sequence[GateOp], pools: Mapping[int, Sequence[GateOp]], positions: Iterable[int]
) -> List[GateOp]:
    selected = list(baseline)
    for position in positions:
        selected[int(position)] = pools[int(position)][0]
    return selected


def _load_task_states(config: Mapping[str, Any], args: argparse.Namespace) -> np.ndarray:
    if config["task"] == "vqe":
        vcfg = VQEConfig(
            n_qubits=int(config["n_qubits"]),
            depth=args.vqe_depth,
            steps=args.vqe_steps,
            lr=args.vqe_lr,
            seed=int(config["seed"]),
            J=args.J,
            h=args.h,
        )
        return sample_intermediate_states_from_vqe(
            vcfg, M=int(config["M"]), device=args.device
        )

    tcfg = ToyVisionConfig(
        n_qubits=int(config["n_qubits"]),
        data_dir=str(args.data_dir),
        seed=int(config["seed"]),
        download=args.allow_download,
        n_train_pool=args.n_train_pool,
        n_val_pool=args.n_val_pool,
    )
    if config["dataset"] == "mnist49":
        _, _, validation_states, validation_labels = build_mnist49_splits(tcfg)
    elif config["dataset"] == "fashion_sb":
        _, _, validation_states, validation_labels = build_fashion_sandal_boot_splits(tcfg)
    elif config["dataset"] == "bas":
        _, _, validation_states, validation_labels = build_bas_splits(
            tcfg, H=args.bas_h, W=args.bas_w
        )
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    task_states, _ = sample_task_ensemble_from_validation(
        validation_states,
        validation_labels,
        M=int(config["M"]),
        seed=int(config["seed"]),
    )
    return task_states


def _build_score_bundle(
    config: Mapping[str, Any],
    args: argparse.Namespace,
    qbetas: Sequence[float],
    task_states: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if task_states is None:
        task_states = _load_task_states(config, args)
    all_gates = build_hea_gates(
        int(config["n_qubits"]),
        int(config["depth"]),
        seed=int(config["seed"]),
        rot_pool_k=int(config["rot_pool_k"]),
        rot_pool_sigma=float(config["rot_pool_sigma"]),
        include_cnot=bool(config["include_cnot"]),
    )
    baseline = build_random_baseline_circuit(all_gates, seed=int(config["seed"]) + 999)
    baseline_signature = _signature(baseline)
    if baseline_signature != config["source_baseline_signature"]:
        raise RuntimeError(
            "Reconstructed baseline signature differs from source: "
            f"{baseline_signature} != {config['source_baseline_signature']}"
        )
    prefixes = _compute_prefix_states_fast(
        baseline, task_states, int(config["n_qubits"])
    )
    pools: Dict[int, List[GateOp]] = {}
    for gate in all_gates:
        pools.setdefault(int(gate.pos_id), []).append(gate)

    lam = lambda_from_gamma(float(config["gamma"]), float(config["alpha"]))
    q_geometries = {}
    for qbeta in qbetas:
        q = q_from_lambda(lam, float(qbeta))
        weights, Mq = build_Gq_diag_popcount(q, int(config["n_qubits"]))
        q_geometries[float(qbeta)] = (float(q), weights, float(Mq))

    position_rows: Dict[float, List[Dict[str, Any]]] = {
        float(qbeta): [] for qbeta in qbetas
    }
    for position, baseline_gate in enumerate(baseline):
        pool = pools[position]
        if len(pool) <= 1 or baseline_gate.gate_type != "Rot":
            continue
        reference = pool[0]
        eligible = baseline_gate is not reference
        physical_angles = _weighted_angles(
            reference,
            baseline_gate,
            prefixes[position],
            np.ones(2 ** int(config["n_qubits"]), dtype=np.float64),
            int(config["n_qubits"]),
        )
        baseline_candidate_index = next(
            index for index, gate in enumerate(pool) if gate is baseline_gate
        )
        for qbeta in qbetas:
            q, weights, _ = q_geometries[float(qbeta)]
            q_angles = _weighted_angles(
                reference,
                baseline_gate,
                prefixes[position],
                weights,
                int(config["n_qubits"]),
            )
            position_rows[float(qbeta)].append(
                {
                    "position": int(position),
                    "layer": int(baseline_gate.layer),
                    "wires": [int(wire) for wire in baseline_gate.wires],
                    "eligible_reference_replacement": bool(eligible),
                    "baseline_candidate_index": int(baseline_candidate_index),
                    "qbeta": float(qbeta),
                    "q": float(q),
                    "q_score_mean": float(np.mean(q_angles)),
                    "q_score_max": float(np.max(q_angles)),
                    "physical_score_mean": float(np.mean(physical_angles)),
                    "physical_score_max": float(np.max(physical_angles)),
                }
            )

    return {
        "lambda": float(lam),
        "baseline": baseline,
        "pools": pools,
        "baseline_signature": baseline_signature,
        "position_rows": position_rows,
        "q_geometries": q_geometries,
    }


def _diagnose_config(
    config: Mapping[str, Any], score_bundle: Mapping[str, Any], qbetas: Sequence[float]
) -> Dict[str, Any]:
    baseline = score_bundle["baseline"]
    pools = score_bundle["pools"]
    k = int(config["source_selected_k"])
    epsilon_q = epsilon_from_delta(float(config["delta"]), 1.0)

    per_q_scores: Dict[float, Dict[int, float]] = {}
    per_q_rows: Dict[float, List[Dict[str, Any]]] = {}
    for qbeta in qbetas:
        rows = [dict(row) for row in score_bundle["position_rows"][float(qbeta)]]
        eligible_rows = [row for row in rows if row["eligible_reference_replacement"]]
        per_q_scores[float(qbeta)] = {
            int(row["position"]): float(row["q_score_mean"]) for row in eligible_rows
        }
        per_q_rows[float(qbeta)] = rows

    if 0.0 not in per_q_scores:
        raise ValueError("qbeta=0 is required as the q=1 reference")
    q1_order = _rank_positions(per_q_scores[0.0])
    if k > len(q1_order):
        raise ValueError(f"Source K={k} exceeds {len(q1_order)} eligible positions")
    q1_matched = q1_order[:k]

    summaries: List[Dict[str, Any]] = []
    all_position_rows: List[Dict[str, Any]] = []
    current_reproduction: Dict[str, Any] = {}
    for qbeta in qbetas:
        qbeta = float(qbeta)
        rows = per_q_rows[qbeta]
        scores = per_q_scores[qbeta]
        order = _rank_positions(scores)
        matched = order[:k]
        native_q = [
            int(row["position"])
            for row in rows
            if row["eligible_reference_replacement"]
            and float(row["q_score_mean"]) <= epsilon_q
        ]
        native_dual = [
            int(row["position"])
            for row in rows
            if row["eligible_reference_replacement"]
            and float(row["q_score_mean"]) <= epsilon_q
            and float(row["physical_score_max"]) <= epsilon_q
        ]
        q_pass_physical_fail = [
            int(row["position"])
            for row in rows
            if row["eligible_reference_replacement"]
            and float(row["q_score_mean"]) <= epsilon_q
            and float(row["physical_score_max"]) > epsilon_q
        ]
        physical_pass_q_fail = [
            int(row["position"])
            for row in rows
            if row["eligible_reference_replacement"]
            and float(row["physical_score_max"]) <= epsilon_q
            and float(row["q_score_mean"]) > epsilon_q
        ]
        spearman, kendall = _rank_correlations(q1_order, order)
        matched_signature = _signature(_replace_positions(baseline, pools, matched))
        native_signature = _signature(_replace_positions(baseline, pools, native_q))
        threshold_margins = [abs(float(score) - epsilon_q) for score in scores.values()]

        summary = {
            "qbeta": qbeta,
            "q": float(score_bundle["q_geometries"][qbeta][0]),
            "lambda": float(score_bundle["lambda"]),
            "Mq": float(score_bundle["q_geometries"][qbeta][2]),
            "epsilon_q": float(epsilon_q),
            "eligible_positions": len(order),
            "matched_k": k,
            "matched_positions": [int(position) for position in matched],
            "matched_signature": matched_signature,
            "native_q_threshold_count": len(native_q),
            "native_q_threshold_positions": sorted(native_q),
            "native_q_threshold_signature": native_signature,
            "native_dual_certificate_count": len(native_dual),
            "native_dual_certificate_positions": sorted(native_dual),
            "q_pass_physical_fail_count": len(q_pass_physical_fail),
            "q_pass_physical_fail_positions": sorted(q_pass_physical_fail),
            "physical_pass_q_fail_count": len(physical_pass_q_fail),
            "physical_pass_q_fail_positions": sorted(physical_pass_q_fail),
            "matched_jaccard_vs_q1": _jaccard(matched, q1_matched),
            "matched_different_positions_vs_q1": len(set(matched) ^ set(q1_matched)),
            "spearman_vs_q1": spearman,
            "kendall_vs_q1": kendall,
            "rank_changed_positions_vs_q1": sum(
                left != right for left, right in zip(q1_order, order)
            ),
            "nearest_threshold_margin": min(threshold_margins) if threshold_margins else None,
            **_boundary_metrics(scores, order, k),
        }
        summaries.append(summary)

        matched_set = set(matched)
        q1_set = set(q1_matched)
        native_set = set(native_q)
        dual_set = set(native_dual)
        rank = {position: index + 1 for index, position in enumerate(order)}
        q1_rank = {position: index + 1 for index, position in enumerate(q1_order)}
        for row in rows:
            position = int(row["position"])
            row.update(
                {
                    "epsilon_q": float(epsilon_q),
                    "rank": rank.get(position),
                    "q1_rank": q1_rank.get(position),
                    "matched_k_selected": position in matched_set,
                    "q1_matched_k_selected": position in q1_set,
                    "native_q_threshold_selected": position in native_set,
                    "native_dual_certificate_selected": position in dual_set,
                }
            )
            all_position_rows.append(row)

        if math.isclose(qbeta, float(config["qbeta"]), rel_tol=0.0, abs_tol=1e-12):
            current_reproduction = {
                "source_qiprune_and_q1_identical": (
                    sorted(config["source_qiprune_positions"])
                    == sorted(config["source_q1_positions"])
                    and config["source_qiprune_signature"] == config["source_q1_signature"]
                ),
                "computed_native_current_matches_source_qiprune_positions": (
                    sorted(native_q) == sorted(config["source_qiprune_positions"])
                ),
                "computed_native_current_matches_source_qiprune_signature": (
                    native_signature == config["source_qiprune_signature"]
                ),
                "computed_matched_current_matches_source_q1_positions": (
                    sorted(matched) == sorted(config["source_q1_positions"])
                ),
                "computed_matched_current_matches_source_q1_signature": (
                    matched_signature == config["source_q1_signature"]
                ),
            }

    if not current_reproduction:
        raise ValueError(
            f"Historical qbeta={config['qbeta']} must be included in the diagnostic grid"
        )
    current_reproduction["all_checks_pass"] = all(current_reproduction.values())

    return {
        "meta": {
            "experiment_family": "q_sensitivity_upgrade_20260810",
            "protocol": "historical_fair_candidate_selection_q_sensitivity",
            "selection_modes": {
                "matched_k": "rank eligible baseline-to-reference replacements by q-score and keep historical K fixed",
                "native_q_threshold": "candidate protocol q-score mean <= epsilon_q",
                "native_dual_certificate": "diagnostic only: q-score mean and physical max <= epsilon_q",
            },
            "physical_certificate_note": "Historical candidate-pool q-iPrune used the q threshold only; the dual-certificate track is reported separately for diagnosis.",
            "source_checkpoint": config["source_checkpoint"],
            "config_id": _config_id(config),
        },
        "config": dict(config),
        "reproduction": current_reproduction,
        "q_summaries": summaries,
        "position_rows": all_position_rows,
    }


def _summary_rows(rows_dir: Path) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path in sorted(rows_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        reproduction = payload["reproduction"]
        for q_summary in payload["q_summaries"]:
            summaries.append(
                {
                    "config_id": payload["meta"]["config_id"],
                    "task": config["task"],
                    "dataset": config["dataset"],
                    "seed": config["seed"],
                    "delta": config["delta"],
                    "rot_pool_sigma": config["rot_pool_sigma"],
                    "qbeta": q_summary["qbeta"],
                    "q": q_summary["q"],
                    "matched_k": q_summary["matched_k"],
                    "native_q_threshold_count": q_summary["native_q_threshold_count"],
                    "native_dual_certificate_count": q_summary["native_dual_certificate_count"],
                    "matched_jaccard_vs_q1": q_summary["matched_jaccard_vs_q1"],
                    "matched_different_positions_vs_q1": q_summary[
                        "matched_different_positions_vs_q1"
                    ],
                    "spearman_vs_q1": q_summary["spearman_vs_q1"],
                    "kendall_vs_q1": q_summary["kendall_vs_q1"],
                    "matched_k_boundary_margin": q_summary["matched_k_boundary_margin"],
                    "nearest_threshold_margin": q_summary["nearest_threshold_margin"],
                    "q_pass_physical_fail_count": q_summary[
                        "q_pass_physical_fail_count"
                    ],
                    "physical_pass_q_fail_count": q_summary[
                        "physical_pass_q_fail_count"
                    ],
                    "historical_reproduction_pass": reproduction["all_checks_pass"],
                }
            )
    return summaries


def _checkpoint(output_dir: Path, meta: Mapping[str, Any], started: float) -> None:
    summary_rows = _summary_rows(output_dir / "rows")
    errors = []
    for path in sorted((output_dir / "errors").glob("*.json")):
        errors.append(json.loads(path.read_text(encoding="utf-8")))
    completed_configs = len({row["config_id"] for row in summary_rows})
    reproduction_configs = {
        row["config_id"] for row in summary_rows if row["historical_reproduction_pass"]
    }
    payload = {
        "meta": dict(meta),
        "completed_configs": completed_configs,
        "completed_q_rows": len(summary_rows),
        "historical_reproduction_passed_configs": len(reproduction_configs),
        "failed_configs": len(errors),
        "errors": errors,
        "elapsed_sec": time.time() - started,
        "results": summary_rows,
    }
    _atomic_json(output_dir / "checkpoint.json", payload)
    _atomic_csv(output_dir / "checkpoint.csv", summary_rows)


def _filter_configs(
    configs: Sequence[Dict[str, Any]], args: argparse.Namespace
) -> List[Dict[str, Any]]:
    selected = []
    for config in configs:
        if args.tasks and config["task"] not in args.tasks:
            continue
        if args.datasets and config["dataset"] not in args.datasets:
            continue
        if args.seeds and int(config["seed"]) not in args.seeds:
            continue
        if args.deltas and float(config["delta"]) not in args.deltas:
            continue
        if args.sigmas and float(config["rot_pool_sigma"]) not in args.sigmas:
            continue
        selected.append(config)
    return selected[: args.max_configs] if args.max_configs else selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable q-sensitivity selection-only diagnostic for the 160 fair configurations."
    )
    parser.add_argument(
        "--source_checkpoints",
        nargs="+",
        type=Path,
        default=DEFAULT_SOURCE_CHECKPOINTS,
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qbetas", nargs="+", type=float, default=[0, 1, 3, 6, 10, 20])
    parser.add_argument("--tasks", nargs="+", choices=["classification", "vqe"])
    parser.add_argument(
        "--datasets", nargs="+", choices=["mnist49", "fashion_sb", "bas", "tfim"]
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--deltas", nargs="+", type=float)
    parser.add_argument("--sigmas", nargs="+", type=float)
    parser.add_argument("--max_configs", type=int)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data")
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

    qbetas = sorted(set(float(value) for value in args.qbetas))
    if 0.0 not in qbetas or 1.0 not in qbetas:
        raise ValueError("The qbeta grid must contain both 0 (q=1) and 1 (historical point)")
    source_paths = [path.resolve() for path in args.source_checkpoints]
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    rows_dir = output_dir / "rows"
    errors_dir = output_dir / "errors"
    rows_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    all_configs = load_source_configurations(source_paths)
    configs = _filter_configs(all_configs, args)
    meta = {
        "experiment_family": "q_sensitivity_upgrade_20260810",
        "runner": str(Path(__file__).resolve()),
        "source_checkpoints": [
            {"path": str(path), "sha256": _file_sha256(path)} for path in source_paths
        ],
        "source_configurations": len(all_configs),
        "selected_configurations": len(configs),
        "qbetas": qbetas,
        "fixed_lambda_mapping": "lambda=clip(1-alpha*gamma,0,1); q=exp(qbeta*(1-lambda))",
        "checkpoint_semantics": "one immutable JSON per completed historical configuration; checkpoint CSV is rebuilt from row JSON files",
    }
    manifest = output_dir / "manifest.json"
    if not manifest.exists():
        _atomic_json(manifest, meta)

    started = time.time()
    score_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    task_state_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
    for index, config in enumerate(configs, start=1):
        config_id = _config_id(config)
        output_path = rows_dir / f"{config_id}.json"
        if output_path.exists():
            print(f"[SKIP {index}/{len(configs)}] {config_id}", flush=True)
            continue
        print(f"[RUN  {index}/{len(configs)}] {config_id}", flush=True)
        try:
            cache_key = (
                config["task"],
                config["dataset"],
                int(config["seed"]),
                float(config["rot_pool_sigma"]),
                int(config["n_qubits"]),
                int(config["depth"]),
                int(config["M"]),
                tuple(qbetas),
            )
            if cache_key not in score_cache:
                task_key = (
                    config["task"],
                    config["dataset"],
                    int(config["seed"]),
                    int(config["n_qubits"]),
                    int(config["M"]),
                )
                if task_key not in task_state_cache:
                    task_state_cache[task_key] = _load_task_states(config, args)
                score_cache[cache_key] = _build_score_bundle(
                    config, args, qbetas, task_state_cache[task_key]
                )
            payload = _diagnose_config(config, score_cache[cache_key], qbetas)
            _atomic_json(output_path, payload)
            error_path = errors_dir / f"{config_id}.json"
            if error_path.exists():
                error_path.unlink()
            print(
                f"[DONE {index}/{len(configs)}] {config_id} "
                f"reproduce={payload['reproduction']['all_checks_pass']}",
                flush=True,
            )
        except Exception as error:
            _atomic_json(
                errors_dir / f"{config_id}.json",
                {
                    "config_id": config_id,
                    "config": config,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[FAIL {index}/{len(configs)}] {config_id}: {error}", flush=True)
            _checkpoint(output_dir, meta, started)
            if not args.continue_on_error:
                raise
        _checkpoint(output_dir, meta, started)

    _checkpoint(output_dir, meta, started)
    checkpoint = json.loads((output_dir / "checkpoint.json").read_text(encoding="utf-8"))
    print(
        f"Completed {checkpoint['completed_configs']}/{len(configs)} configs; "
        f"historical reproduction passed "
        f"{checkpoint['historical_reproduction_passed_configs']}/"
        f"{checkpoint['completed_configs']}. Checkpoint: {output_dir / 'checkpoint.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
