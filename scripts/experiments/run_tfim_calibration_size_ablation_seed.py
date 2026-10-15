from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry
from iprune.experiments.tasks_vqe import (
    VQEConfig,
    optimize_vqe_parameters_with_diagnostics,
)
from iprune.experiments.vqe_finetune import fine_tune_vqe_circuit
from iprune.gates import build_hea_gates_from_parameters, is_identity_rot
from iprune.pennylane_unitary import unitary_of_single_gate
from iprune.qgeometry import task_conditioned_output_angles
from iprune.pruning_true import (
    audit_fixed_identity_deletion_path,
    q_iPrune_iterative_identity_audited,
)


def select_lowest_q_angles(
    *,
    all_gates: list,
    baseline_circuit: list,
    n_qubits: int,
    task_states: np.ndarray,
    weights: np.ndarray,
    deletion_budget: int,
) -> list[int]:
    id_by_pos = {int(g.pos_id): g for g in all_gates if is_identity_rot(g)}
    scores: dict[int, float] = {}
    current = task_states.copy()
    for gate in baseline_circuit:
        position = int(gate.pos_id)
        if gate.gate_type == "Rot" and position in id_by_pos:
            left = unitary_of_single_gate(gate.apply, n_qubits=n_qubits, device_name="default.qubit")
            right = unitary_of_single_gate(
                id_by_pos[position].apply, n_qubits=n_qubits, device_name="default.qubit"
            )
            q_angles, _ = task_conditioned_output_angles(left, right, current, weights)
            scores[position] = float(np.max(q_angles))
        current = (
            unitary_of_single_gate(gate.apply, n_qubits=n_qubits, device_name="default.qubit")
            @ current.T
        ).T
    ranked = sorted(scores, key=lambda position: (scores[position], position))
    return ranked[:deletion_budget]


def random_product_states(n_qubits: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 98765)
    states = []
    for _ in range(count):
        angles = rng.uniform(0.0, 2.0 * np.pi, size=(n_qubits, 2))
        single = [
            np.array(
                [np.cos(theta / 2.0), np.sin(theta / 2.0) * np.exp(1j * phi)],
                dtype=np.complex128,
            )
            for theta, phi in angles
        ]
        state = single[0]
        for part in single[1:]:
            state = np.kron(state, part)
        states.append(state)
    return np.stack(states, axis=0).reshape(count, 2**n_qubits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration-size sensitivity for TFIM.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 4, 8])
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
    calibration_states = random_product_states(
        n_qubits=n_qubits,
        count=max(args.sizes),
        seed=args.seed,
    )
    correct = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    correct[0, 0] = 1.0
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
            task_states=correct,
            Gq_diag=weights,
            q_epsilon=epsilon,
            physical_epsilon=epsilon,
        )
        k_budget = len(forward.accepted_positions)
        for size in args.sizes:
            states = calibration_states[:size]
            selected_positions = select_lowest_q_angles(
                all_gates=all_gates,
                baseline_circuit=baseline,
                n_qubits=n_qubits,
                task_states=states,
                weights=weights,
                deletion_budget=k_budget,
            )
            audit = audit_fixed_identity_deletion_path(
                all_gates=all_gates,
                baseline_circuit=baseline,
                selected_positions=selected_positions,
                n_qubits=n_qubits,
                device_name="default.qubit",
                task_states=correct,
                Gq_diag=weights,
                physical_epsilon=epsilon,
            )
            compact = [
                gate for gate in baseline if gate.pos_id not in set(selected_positions)
            ]
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
                    "calibration_size": int(size),
                    "matched_delete_count": len(selected_positions),
                    "deleted_pos_ids": sorted(int(p) for p in selected_positions),
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
            "protocol": "tfim_calibration_size_sensitivity",
            "seed": args.seed,
            "n_qubits": n_qubits,
            "vqe_converged": bool(optimized.converged),
            "vqe_steps_run": int(optimized.steps_run),
            "deltas": args.deltas,
            "calibration_sizes": args.sizes,
        },
        "results": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
