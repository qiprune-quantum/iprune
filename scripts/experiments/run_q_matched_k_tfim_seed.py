from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters
from iprune.experiments.vqe_finetune import fine_tune_vqe_circuit
from iprune.gates import GateOp, build_hea_gates_from_parameters, is_identity_rot
from iprune.qgeometry import epsilon_from_delta
from scripts.experiments.run_q_sensitivity_selection_diagnostic import (
    _apply_cnot,
    _apply_one_qubit,
    _compute_prefix_states_fast,
    _jaccard,
    _rank_correlations,
    _rank_positions,
    _rot_matrix,
    _weighted_angles,
)
from scripts.experiments.run_vqe_qiprune_vs_gradient_seed import _evaluate
from scripts.legacy.sweep_all_v2 import _circuit_signature


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CHECKPOINT = (
    PROJECT_DIR
    / "results"
    / "supplementary_upgrade_20260807"
    / "tfim_physical_baselines_seed42_46"
    / "physical_gate_baselines_checkpoint.json"
)
SOURCE_METHOD = "qiprune_true_identity"


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
        if int(row["seed"]) == seed and row["method"] == SOURCE_METHOD
    }
    missing = [delta for delta in deltas if float(delta) not in rows]
    if missing:
        raise ValueError(f"Missing source rows for seed={seed}, deltas={missing}")
    return rows


def _pools_by_position(all_gates: Sequence[GateOp]) -> Dict[int, List[GateOp]]:
    pools: Dict[int, List[GateOp]] = {}
    for gate in all_gates:
        pools.setdefault(int(gate.pos_id), []).append(gate)
    return pools


def _identity_candidate(pool: Sequence[GateOp]) -> GateOp:
    identities = [gate for gate in pool if is_identity_rot(gate)]
    if len(identities) != 1:
        raise ValueError("Each trainable Rot position must have exactly one identity candidate")
    return identities[0]


def _compact_without(circuit: Sequence[GateOp], positions: Sequence[int]) -> List[GateOp]:
    deleted = set(int(position) for position in positions)
    return [gate for gate in circuit if int(gate.pos_id) not in deleted]


def _selected_with_identities(
    all_gates: Sequence[GateOp],
    baseline_circuit: Sequence[GateOp],
    positions: Sequence[int],
) -> List[GateOp]:
    pools = _pools_by_position(all_gates)
    selected = list(baseline_circuit)
    for position in positions:
        selected[int(position)] = _identity_candidate(pools[int(position)])
    return selected


def _audit_physical_selected_path(
    *,
    all_gates: Sequence[GateOp],
    baseline_circuit: Sequence[GateOp],
    deleted_positions: Sequence[int],
    initial_state: np.ndarray,
    n_qubits: int,
    epsilon_q: float,
) -> Dict[str, Any]:
    selected = _selected_with_identities(all_gates, baseline_circuit, deleted_positions)
    prefixes = _compute_prefix_states_fast(selected, initial_state, n_qubits)
    pools = _pools_by_position(all_gates)
    scores = {}
    for position in deleted_positions:
        baseline_gate = baseline_circuit[int(position)]
        identity = _identity_candidate(pools[int(position)])
        angles = _weighted_angles(
            baseline_gate,
            identity,
            prefixes[int(position)],
            np.ones(2**n_qubits, dtype=np.float64),
            n_qubits,
        )
        scores[int(position)] = float(np.max(angles))
    violations = sorted(
        position for position, score in scores.items() if score > epsilon_q + 1e-12
    )
    return {
        "physical_selected_path_scores": scores,
        "physical_selected_path_max": max(scores.values()) if scores else 0.0,
        "physical_selected_path_violations": violations,
        "physical_selected_path_certificate_pass": not violations,
    }


def _fast_native_deleted_positions(
    *,
    all_gates: Sequence[GateOp],
    baseline_circuit: Sequence[GateOp],
    task_states: np.ndarray,
    Gq_diag: np.ndarray,
    epsilon_q: float,
    n_qubits: int,
) -> List[int]:
    """Equivalent sequential identity pruning without constructing full gate matrices."""
    pools = _pools_by_position(all_gates)
    current = np.asarray(task_states, dtype=np.complex128).copy()
    deleted: List[int] = []
    for gate in baseline_circuit:
        selected = gate
        if gate.gate_type == "Rot":
            identity = _identity_candidate(pools[int(gate.pos_id)])
            q_angles = _weighted_angles(
                gate, identity, current, Gq_diag, n_qubits
            )
            physical_angles = _weighted_angles(
                gate,
                identity,
                current,
                np.ones(2**n_qubits, dtype=np.float64),
                n_qubits,
            )
            if (
                float(np.max(q_angles)) <= epsilon_q
                and float(np.max(physical_angles)) <= epsilon_q
            ):
                selected = identity
                deleted.append(int(gate.pos_id))
        if selected.gate_type == "Rot":
            current = _apply_one_qubit(
                current,
                _rot_matrix(selected),
                int(selected.wires[0]),
                n_qubits,
            )
        elif selected.gate_type == "CNOT":
            current = _apply_cnot(
                current,
                int(selected.wires[0]),
                int(selected.wires[1]),
                n_qubits,
            )
        else:
            raise ValueError(f"Unsupported gate type: {selected.gate_type}")
    return deleted


def _score_positions(
    *,
    all_gates: Sequence[GateOp],
    baseline_circuit: Sequence[GateOp],
    prefix_circuit: Sequence[GateOp],
    initial_state: np.ndarray,
    n_qubits: int,
    qbetas: Sequence[float],
    gamma: float,
    alpha: float,
) -> Tuple[Dict[float, Dict[int, float]], Dict[int, float], List[Dict[str, Any]]]:
    prefixes = _compute_prefix_states_fast(prefix_circuit, initial_state, n_qubits)
    pools = _pools_by_position(all_gates)
    q_scores: Dict[float, Dict[int, float]] = {float(qbeta): {} for qbeta in qbetas}
    physical_scores: Dict[int, float] = {}
    rows: List[Dict[str, Any]] = []

    geometries = {}
    for qbeta in qbetas:
        cfg = PaperProtocolConfig(
            n_qubits=n_qubits,
            gamma=gamma,
            alpha=alpha,
            qbeta=float(qbeta),
            delta=1.0,
        )
        lam, q, weights, Mq, _ = make_q_geometry(cfg)
        geometries[float(qbeta)] = (lam, q, weights, Mq)

    for gate in baseline_circuit:
        if gate.gate_type != "Rot":
            continue
        position = int(gate.pos_id)
        identity = _identity_candidate(pools[position])
        physical_angles = _weighted_angles(
            gate,
            identity,
            prefixes[position],
            np.ones(2**n_qubits, dtype=np.float64),
            n_qubits,
        )
        physical_max = float(np.max(physical_angles))
        physical_scores[position] = physical_max
        for qbeta in qbetas:
            lam, q, weights, Mq = geometries[float(qbeta)]
            q_angles = _weighted_angles(
                gate, identity, prefixes[position], weights, n_qubits
            )
            q_max = float(np.max(q_angles))
            q_scores[float(qbeta)][position] = q_max
            rows.append(
                {
                    "position": position,
                    "layer": int(gate.layer),
                    "wires": [int(wire) for wire in gate.wires],
                    "qbeta": float(qbeta),
                    "q": float(q),
                    "lambda": float(lam),
                    "Mq": float(Mq),
                    "q_score_mean": float(np.mean(q_angles)),
                    "q_score_max": q_max,
                    "physical_score_mean": float(np.mean(physical_angles)),
                    "physical_score_max": physical_max,
                }
            )
    return q_scores, physical_scores, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-seed TFIM matched-K q-geometry contribution pilot."
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--source_checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deltas", nargs="+", type=float, default=[1.2, 1.5])
    parser.add_argument("--qbetas", nargs="+", type=float, default=[0, 1, 3, 6, 10, 20])
    parser.add_argument("--selection_only", action="store_true")
    parser.add_argument("--allow_new_seed", action="store_true")
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqe_depth", type=int, default=6)
    parser.add_argument("--vqe_steps", type=int, default=40)
    parser.add_argument("--vqe_lr", type=float, default=0.1)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--finetune_lr", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    args = parser.parse_args()

    qbetas = sorted(set(float(value) for value in args.qbetas))
    if 0.0 not in qbetas or 1.0 not in qbetas:
        raise ValueError("qbetas must include 0 and the historical qbeta=1 point")
    source_checkpoint = args.source_checkpoint.resolve()
    source_origin = "historical_checkpoint"
    try:
        source_rows = _load_source_rows(source_checkpoint, args.seed, args.deltas)
    except ValueError:
        if not args.allow_new_seed:
            raise
        source_rows = {}
        source_origin = "fresh_qbeta1_native_threshold"

    vcfg = VQEConfig(
        n_qubits=args.n_qubits,
        depth=args.vqe_depth,
        steps=args.vqe_steps,
        lr=args.vqe_lr,
        seed=args.seed,
        J=args.J,
        h=args.h,
    )
    optimized_params, baseline_history = optimize_vqe_parameters(vcfg, device=args.device)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        optimized_params, include_cnot=True, include_identity_rot=True
    )
    baseline_signature = _circuit_signature(baseline_circuit)
    if source_rows:
        source_signatures = {
            str(source_rows[float(delta)]["baseline_circuit_signature"])
            for delta in args.deltas
        }
        if source_signatures != {baseline_signature}:
            raise RuntimeError(
                f"Reconstructed TFIM circuit signature mismatch: {baseline_signature} vs {source_signatures}"
            )

    initial_state = np.zeros((1, 2**args.n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    baseline_state = simulate_outputs(
        baseline_circuit, initial_state, args.n_qubits, args.device
    )
    energy_original = float(baseline_history[-1])
    reference_source = source_rows.get(float(args.deltas[0]))
    if reference_source and not math.isclose(
            energy_original,
            float(reference_source["energy_original_optimized"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("Reconstructed optimized energy differs from historical source")

    if not source_rows:
        qbeta1_cfg = PaperProtocolConfig(
            device=args.device,
            n_qubits=args.n_qubits,
            depth=args.vqe_depth,
            M=1,
            seed=args.seed,
            gamma=args.gamma,
            alpha=args.alpha,
            qbeta=1.0,
            delta=1.0,
            rot_pool_k=1,
            rot_pool_sigma=0.0,
            include_cnot=True,
        )
        _, _, qbeta1_weights, _, _ = make_q_geometry(qbeta1_cfg)
        for delta in args.deltas:
            source_rows[float(delta)] = {
                "matched_delete_count": len(
                    _fast_native_deleted_positions(
                        all_gates=all_gates,
                        baseline_circuit=baseline_circuit,
                        task_states=initial_state,
                        Gq_diag=qbeta1_weights,
                        epsilon_q=epsilon_from_delta(float(delta), 1.0),
                        n_qubits=args.n_qubits,
                    )
                ),
                "baseline_circuit_signature": baseline_signature,
                "energy_original_optimized": energy_original,
            }
        reference_source = source_rows[float(args.deltas[0])]

    control = None
    control_state = None
    energy_control = None
    if not args.selection_only:
        control = fine_tune_vqe_circuit(
            circuit=baseline_circuit,
            n_qubits=args.n_qubits,
            J=args.J,
            h=args.h,
            steps=args.finetune_steps,
            lr=args.finetune_lr,
            device=args.device,
        )
        control_state = control.final_state.reshape(1, -1)
        energy_control = float(control.energy_history[-1])
        if source_origin == "historical_checkpoint" and not math.isclose(
            energy_control,
            float(reference_source["energy_full_control_after_extra_steps"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("Reconstructed full-control energy differs from historical source")

    evaluation_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    result_rows: List[Dict[str, Any]] = []
    all_score_rows: List[Dict[str, Any]] = []

    for delta in args.deltas:
        source = source_rows[float(delta)]
        source_target_k = int(source["matched_delete_count"])
        epsilon_q = epsilon_from_delta(float(delta), 1.0)
        q_scores, physical_scores, delta_score_rows = _score_positions(
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            prefix_circuit=baseline_circuit,
            initial_state=initial_state,
            n_qubits=args.n_qubits,
            qbetas=qbetas,
            gamma=args.gamma,
            alpha=args.alpha,
        )
        for score_row in delta_score_rows:
            score_row["delta"] = float(delta)
        all_score_rows.extend(delta_score_rows)
        q1_order = _rank_positions(q_scores[0.0])
        baseline_prefix_physical_eligible = {
            position
            for position, score in physical_scores.items()
            if score <= epsilon_q + 1e-12
        }
        matched_k = min(source_target_k, len(baseline_prefix_physical_eligible))
        q1_feasible_order = [
            position
            for position in q1_order
            if position in baseline_prefix_physical_eligible
        ]
        q1_selected = q1_feasible_order[:matched_k]

        for qbeta in qbetas:
            cfg = PaperProtocolConfig(
                device=args.device,
                n_qubits=args.n_qubits,
                depth=args.vqe_depth,
                M=1,
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
            order = [
                position
                for position in _rank_positions(q_scores[qbeta])
                if position in baseline_prefix_physical_eligible
            ]
            deleted = sorted(order[:matched_k])
            physical_audit = _audit_physical_selected_path(
                all_gates=all_gates,
                baseline_circuit=baseline_circuit,
                deleted_positions=deleted,
                initial_state=initial_state,
                n_qubits=args.n_qubits,
                epsilon_q=epsilon_q,
            )
            native_deleted = _fast_native_deleted_positions(
                all_gates=all_gates,
                baseline_circuit=baseline_circuit,
                task_states=initial_state,
                Gq_diag=Gq_diag,
                epsilon_q=epsilon_q,
                n_qubits=args.n_qubits,
            )
            spearman, kendall = _rank_correlations(q1_order, _rank_positions(q_scores[qbeta]))

            metrics: Dict[str, Any] = {}
            reused = False
            if not args.selection_only:
                cache_key = tuple(deleted)
                if cache_key not in evaluation_cache:
                    evaluation_cache[cache_key] = _evaluate(
                        method=f"qbeta_{qbeta:g}_matched_k",
                        compact_circuit=_compact_without(baseline_circuit, deleted),
                        deleted_pos_ids=deleted,
                        baseline_circuit=baseline_circuit,
                        baseline_state=baseline_state,
                        control_state=control_state,
                        energy_original=energy_original,
                        energy_control=energy_control,
                        energy_initial=float(baseline_history[0]),
                        n_qubits=args.n_qubits,
                        J=args.J,
                        h=args.h,
                        finetune_steps=args.finetune_steps,
                        finetune_lr=args.finetune_lr,
                        device=args.device,
                    )
                else:
                    reused = True
                metrics = dict(evaluation_cache[cache_key])
                metrics["method"] = f"qbeta_{qbeta:g}_matched_k"

            row = {
                **metrics,
                "experiment_family": "q_sensitivity_upgrade_20260810",
                "experiment": "tfim_q_geometry_matched_k_pilot",
                "selection_only": bool(args.selection_only),
                "task": "vqe",
                "dataset": "tfim",
                "seed": args.seed,
                "delta": float(delta),
                "qbeta": float(qbeta),
                "q": float(q),
                "lambda": float(lam),
                "Mq": float(Mq),
                "epsilon_q": float(epsilon_q),
                "matched_delete_count": matched_k,
                "source_target_delete_count": source_target_k,
                "physical_certificate_k_shortfall": source_target_k - matched_k,
                "deleted_pos_ids": deleted,
                "native_threshold_delete_count": len(native_deleted),
                "native_threshold_deleted_pos_ids": native_deleted,
                "baseline_prefix_physical_eligible_positions": len(
                    baseline_prefix_physical_eligible
                ),
                "fixed_baseline_physical_certificate_pass": all(
                    position in baseline_prefix_physical_eligible
                    for position in deleted
                ),
                **physical_audit,
                "matched_jaccard_vs_q1": _jaccard(deleted, q1_selected),
                "matched_different_positions_vs_q1": len(
                    set(deleted) ^ set(q1_selected)
                ),
                "spearman_vs_q1": spearman,
                "kendall_vs_q1": kendall,
                "evaluation_reused_for_identical_selection": reused,
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
                    else f" abs_dE={row['abs_dE_finetuned_vs_original']:.6f}"
                ),
                flush=True,
            )

    _atomic_new_json(
        args.output_json,
        {
            "meta": {
                "experiment_family": "q_sensitivity_upgrade_20260810",
                "protocol": "same_optimized_tfim_checkpoint_same_K_same_fixed_physical_eligible_set_same_finetune_budget",
                "selection_score": "maximum q-weighted angle to identity on fixed baseline prefix states",
                "physical_certificate": "all q variants rank within the same positions whose maximum physical angle on fixed baseline prefix states is <= epsilon_q; K is capped at the size of that common feasible set",
                "source_checkpoint": str(source_checkpoint),
                "source_origin": source_origin,
                "seed": args.seed,
                "deltas": args.deltas,
                "qbetas": qbetas,
                "selection_only": bool(args.selection_only),
                "expected_rows": len(args.deltas) * len(qbetas),
            },
            "optimized_parameters": optimized_params.tolist(),
            "score_rows": all_score_rows,
            "results": result_rows,
        },
    )
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
