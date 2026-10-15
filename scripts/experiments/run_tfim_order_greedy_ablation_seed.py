from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters_with_diagnostics
from iprune.experiments.vqe_finetune import fine_tune_vqe_circuit
from iprune.gates import GateOp, build_hea_gates_from_parameters, is_identity_rot
from iprune.pennylane_unitary import unitary_of_single_gate
from iprune.pruning_true import (
    audit_fixed_identity_deletion_path,
    q_iPrune_fixed_k_identity,
    q_iPrune_iterative_identity_audited,
)
from iprune.qgeometry import task_conditioned_output_angles


def _unitary(gate: GateOp, n_qubits: int) -> np.ndarray:
    return unitary_of_single_gate(gate.apply, n_qubits=n_qubits, device_name="default.qubit")


def _without(circuit: list[GateOp], positions: set[int]) -> list[GateOp]:
    return [gate for gate in circuit if int(gate.pos_id) not in positions]


def _prefix_states(
    circuit: list[GateOp], initial: np.ndarray, n_qubits: int
) -> list[np.ndarray]:
    states: list[np.ndarray] = []
    current = initial.copy()
    for gate in circuit:
        states.append(current.copy())
        current = (_unitary(gate, n_qubits) @ current.T).T
    return states


def _angle_scores(
    circuit: list[GateOp],
    id_by_pos: dict[int, GateOp],
    states_by_pos: list[np.ndarray],
    weights: np.ndarray,
    n_qubits: int,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for index, gate in enumerate(circuit):
        position = int(gate.pos_id)
        if gate.gate_type != "Rot" or position not in id_by_pos:
            continue
        left = _unitary(gate, n_qubits)
        right = _unitary(id_by_pos[position], n_qubits)
        q_angles, _ = task_conditioned_output_angles(
            left, right, states_by_pos[index], weights
        )
        scores[position] = float(np.max(q_angles))
    return scores


def backward_dynamic_select(
    *,
    baseline_circuit: list[GateOp],
    all_gates: list[GateOp],
    n_qubits: int,
    task_states: np.ndarray,
    weights: np.ndarray,
    deletion_budget: int,
) -> list[int]:
    """Backward-context scoring: scan from the output end and retain the
    unitary suffix of untouched gates while scoring each rotation."""
    id_by_pos = {int(gate.pos_id): gate for gate in all_gates if is_identity_rot(gate)}
    dim = 2**n_qubits
    suffix = np.eye(dim, dtype=np.complex128)
    prefix = _prefix_states(list(baseline_circuit), task_states, n_qubits)
    scores: dict[int, float] = {}
    for index in range(len(baseline_circuit) - 1, -1, -1):
        gate = baseline_circuit[index]
        position = int(gate.pos_id)
        if gate.gate_type == "Rot" and position in id_by_pos:
            left = suffix @ _unitary(gate, n_qubits)
            right = suffix
            q_angles, _ = task_conditioned_output_angles(
                left, right, prefix[index], weights
            )
            scores[position] = float(np.max(q_angles))
        suffix = suffix @ _unitary(gate, n_qubits)
    ranked = sorted(scores, key=lambda position: (scores[position], position))
    return ranked[:deletion_budget]


def global_greedy_select(
    *,
    baseline_circuit: list[GateOp],
    all_gates: list[GateOp],
    n_qubits: int,
    task_states: np.ndarray,
    weights: np.ndarray,
    deletion_budget: int,
) -> list[int]:
    id_by_pos = {int(gate.pos_id): gate for gate in all_gates if is_identity_rot(gate)}
    current = list(baseline_circuit)
    selected: list[int] = []
    for _ in range(deletion_budget):
        prefixes = _prefix_states(current, task_states, n_qubits)
        scores = _angle_scores(current, id_by_pos, prefixes, weights, n_qubits)
        if not scores:
            break
        best = min(scores, key=lambda position: (scores[position], position))
        selected.append(int(best))
        current = [gate for gate in current if int(gate.pos_id) != int(best)]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched-K order-and-greedy pruning ablation for TFIM."
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--n_qubits", type=int, default=4)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)

    n_qubits = args.n_qubits
    J, h = 1.0, 1.0
    optimized = optimize_vqe_parameters_with_diagnostics(
        VQEConfig(
            n_qubits=n_qubits,
            depth=args.depth,
            steps=args.steps,
            lr=0.1,
            seed=args.seed,
            J=J,
            h=h,
            min_steps=200,
            convergence_window=20,
            convergence_tol=5e-5,
        )
    )
    if not optimized.converged:
        raise RuntimeError("TFIM baseline did not meet the frozen convergence rule.")
    all_gates, baseline = build_hea_gates_from_parameters(
        optimized.params, include_cnot=True, include_identity_rot=True
    )
    initial = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial[0, 0] = 1.0
    control = fine_tune_vqe_circuit(
        circuit=baseline,
        n_qubits=n_qubits,
        J=J,
        h=h,
        steps=args.finetune_steps,
        lr=0.05,
        device="default.qubit",
    )
    control_energy = float(control.energy_history[-1])

    rows: list[dict[str, Any]] = []
    for delta in args.deltas:
        cfg = PaperProtocolConfig(
            device="default.qubit",
            n_qubits=n_qubits,
            depth=args.depth,
            M=1,
            seed=args.seed,
            gamma=0.0,
            alpha=0.6,
            qbeta=0.0,
            delta=float(delta),
            rot_pool_k=1,
            rot_pool_sigma=0.0,
            include_cnot=True,
        )
        _lam, q, weights, _Mq, epsilon = make_q_geometry(cfg)
        forward = q_iPrune_iterative_identity_audited(
            all_gates=all_gates,
            baseline_circuit=baseline,
            n_qubits=n_qubits,
            device_name="default.qubit",
            task_states=initial,
            Gq_diag=weights,
            q_epsilon=epsilon,
            physical_epsilon=epsilon,
        )
        k_budget = len(forward.accepted_positions)
        static = q_iPrune_fixed_k_identity(
            all_gates=all_gates,
            baseline_circuit=baseline,
            n_qubits=n_qubits,
            device_name="default.qubit",
            task_states=initial,
            Gq_diag=weights,
            physical_epsilon=epsilon,
            deletion_budget=k_budget,
            require_full_budget=False,
        )
        backward = backward_dynamic_select(
            baseline_circuit=baseline,
            all_gates=all_gates,
            n_qubits=n_qubits,
            task_states=initial,
            weights=weights,
            deletion_budget=k_budget,
        )
        greedy = global_greedy_select(
            baseline_circuit=baseline,
            all_gates=all_gates,
            n_qubits=n_qubits,
            task_states=initial,
            weights=weights,
            deletion_budget=k_budget,
        )

        strategies = {
            "forward_dynamic": forward.accepted_positions,
            "static_frozen": static.selected_positions,
            "backward_dynamic": backward,
            "global_greedy": greedy,
        }
        for method, positions in strategies.items():
            audit = audit_fixed_identity_deletion_path(
                all_gates=all_gates,
                baseline_circuit=baseline,
                selected_positions=sorted(positions),
                n_qubits=n_qubits,
                device_name="default.qubit",
                task_states=initial,
                Gq_diag=weights,
                physical_epsilon=epsilon,
            )
            compact = _without(list(baseline), set(int(p) for p in positions))
            tuned = fine_tune_vqe_circuit(
                circuit=compact,
                n_qubits=n_qubits,
                J=J,
                h=h,
                steps=args.finetune_steps,
                lr=0.05,
                device="default.qubit",
            )
            rows.append(
                {
                    "seed": args.seed,
                    "delta": float(delta),
                    "q": float(q),
                    "epsilon": float(epsilon),
                    "method": method,
                    "matched_delete_count": len(positions),
                    "deleted_pos_ids": sorted(int(p) for p in positions),
                    "energy_gap_finetuned_vs_control": float(
                        tuned.energy_history[-1] - control_energy
                    ),
                    "actual_trace_distance_max": float(audit.actual_trace_distance_max),
                    "path_trace_sum_max": float(audit.path_trace_sum_max),
                    "physical_threshold_violations": int(
                        audit.physical_threshold_violations
                    ),
                    "path_bound_violations": int(audit.path_bound_violations),
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "protocol": "tfim_matched_k_order_greedy_ablation",
            "seed": args.seed,
            "n_qubits": n_qubits,
            "vqe_converged": bool(optimized.converged),
            "vqe_steps_run": int(optimized.steps_run),
            "deltas": args.deltas,
        },
        "results": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
