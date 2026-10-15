"""Noisy-execution audit for post-training q=1 TFIM identity deletion.

All VQE optimization and gate selection are noiseless.  Depolarizing noise is
applied only after every retained physical gate at evaluation time, so the test
asks whether deleting rotations trades structural error for lower noise exposure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pennylane as qml

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs
from iprune.experiments.qfi_gate_pruning import diagonal_qfi_gate_saliency_scores
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters_with_diagnostics, tfim_hamiltonian
from iprune.gates import GateOp, build_hea_gates_from_parameters
from iprune.pruning_true import q_iPrune_iterative_identity_audited


def rotations(circuit: list[GateOp]) -> list[int]:
    return [int(gate.pos_id) for gate in circuit if gate.gate_type == "Rot"]


def compact(circuit: list[GateOp], deleted: list[int]) -> list[GateOp]:
    selected = {int(position) for position in deleted}
    return [gate for gate in circuit if int(gate.pos_id) not in selected]


def magnitude(gate: GateOp) -> float:
    return float(np.linalg.norm(np.angle(np.exp(1j * np.asarray(gate.params, dtype=np.float64)))))


def noisy_energy(circuit: list[GateOp], hamiltonian, n_qubits: int, noise_strength: float) -> float:
    device = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(device)
    def run():
        for gate in circuit:
            gate.apply()
            for wire in gate.wires:
                qml.DepolarizingChannel(noise_strength, wires=int(wire))
        return qml.expval(hamiltonian)

    return float(run())


def noiseless_energy(circuit: list[GateOp], hamiltonian, n_qubits: int) -> float:
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def run():
        for gate in circuit:
            gate.apply()
        return qml.expval(hamiltonian)

    return float(run())


def main() -> None:
    parser = argparse.ArgumentParser(description="TFIM q=1 post-training noisy-execution audit.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    parser.add_argument("--noise_strengths", nargs="+", type=float, default=[0.001, 0.005, 0.01])
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqe_depth", type=int, default=4)
    parser.add_argument("--vqe_steps", type=int, default=1800)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "full_unpruned",
            "qiprune_iterative_hybrid_path",
            "diagonal_qfi_quadratic_saliency_matched_k",
            "parameter_magnitude_matched_k",
            "random_matched_k",
            "single_gate_energy_probe_matched_k",
        ],
        help="Subset of selectors to evaluate under the same noisy-execution protocol.",
    )
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    if any(not 0.0 <= p <= 1.0 for p in args.noise_strengths):
        raise ValueError("noise strengths must be in [0, 1].")
    supported_methods = {
        "full_unpruned", "qiprune_iterative_hybrid_path",
        "diagonal_qfi_quadratic_saliency_matched_k",
        "parameter_magnitude_matched_k", "random_matched_k",
        "single_gate_energy_probe_matched_k",
    }
    unknown_methods = set(args.methods) - supported_methods
    if unknown_methods:
        raise ValueError(f"Unsupported methods: {sorted(unknown_methods)}")
    if "qiprune_iterative_hybrid_path" not in args.methods:
        raise ValueError("qiprune_iterative_hybrid_path must be included to define matched deletion counts.")

    n_qubits, J, h = args.n_qubits, args.J, args.h
    optimized = optimize_vqe_parameters_with_diagnostics(VQEConfig(
        n_qubits=n_qubits, depth=args.vqe_depth, steps=args.vqe_steps, lr=0.1,
        seed=args.seed, J=J, h=h, min_steps=200, convergence_window=20,
        convergence_tol=5e-5,
    ))
    if not optimized.converged:
        raise RuntimeError("TFIM baseline did not satisfy the frozen convergence rule.")
    all_gates, baseline = build_hea_gates_from_parameters(
        optimized.params, include_cnot=True, include_identity_rot=True
    )
    hamiltonian = tfim_hamiltonian(n_qubits, J, h)
    baseline_energy = noiseless_energy(baseline, hamiltonian, n_qubits)
    initial = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial[0, 0] = 1.0
    baseline_state = simulate_outputs(baseline, initial, n_qubits, "default.qubit")
    rotation_by_id = {int(gate.pos_id): gate for gate in baseline if gate.gate_type == "Rot"}
    magnitude_rank = None
    if "parameter_magnitude_matched_k" in args.methods:
        magnitude_rank = sorted(rotation_by_id, key=lambda p: (magnitude(rotation_by_id[p]), p))
    random_rank = None
    if "random_matched_k" in args.methods:
        random_rank = list(np.random.default_rng(args.seed + 20260913).permutation(rotations(baseline)))
    qfi_rank, qfi_ms = None, None
    if "diagonal_qfi_quadratic_saliency_matched_k" in args.methods:
        qfi_rank, _, _, qfi_ms = diagonal_qfi_gate_saliency_scores(
            circuit=baseline, task_states=initial, n_qubits=n_qubits, device="default.qubit"
        )
    probe_rank, probe_ms = None, None
    if "single_gate_energy_probe_matched_k" in args.methods:
        started = perf_counter()
        probe_scores = {
            p: abs(noiseless_energy(compact(baseline, [p]), hamiltonian, n_qubits) - baseline_energy)
            for p in rotations(baseline)
        }
        probe_rank = sorted(probe_scores, key=lambda p: (probe_scores[p], p))
        probe_ms = (perf_counter() - started) * 1000.0

    rows: list[dict] = []
    for delta in args.deltas:
        config = PaperProtocolConfig(
            device="default.qubit", n_qubits=n_qubits, depth=args.vqe_depth, M=1,
            seed=args.seed, gamma=0.0, alpha=0.6, qbeta=0.0, delta=float(delta),
            rot_pool_k=1, rot_pool_sigma=0.0, include_cnot=True,
        )
        _, q, weights, _, epsilon = make_q_geometry(config)
        qresult = q_iPrune_iterative_identity_audited(
            all_gates=all_gates, baseline_circuit=baseline, n_qubits=n_qubits,
            device_name="default.qubit", task_states=initial, Gq_diag=weights,
            q_epsilon=epsilon, physical_epsilon=epsilon,
        )
        deleted = list(qresult.accepted_positions)
        selections_all = {
            "full_unpruned": [],
            "qiprune_iterative_hybrid_path": deleted,
            "diagonal_qfi_quadratic_saliency_matched_k": list(qfi_rank[:len(deleted)]) if qfi_rank is not None else None,
            "parameter_magnitude_matched_k": list(magnitude_rank[:len(deleted)]) if magnitude_rank is not None else None,
            "random_matched_k": list(random_rank[:len(deleted)]) if random_rank is not None else None,
            "single_gate_energy_probe_matched_k": list(probe_rank[:len(deleted)]) if probe_rank is not None else None,
        }
        selections = {method: selections_all[method] for method in args.methods}
        for method, positions in selections.items():
            circuit = compact(baseline, positions)
            ideal = noiseless_energy(circuit, hamiltonian, n_qubits)
            for strength in args.noise_strengths:
                value = noisy_energy(circuit, hamiltonian, n_qubits, float(strength))
                rows.append({
                    "experiment": "tfim_q1_posttraining_gate_local_depolarizing_audit",
                    "seed": args.seed, "delta": float(delta), "q": float(q),
                    "noise_channel": "depolarizing", "noise_strength": float(strength),
                    "method": method, "deleted_pos_ids": sorted(int(p) for p in positions),
                    "matched_delete_count": len(positions), "gate_count_after": len(circuit),
                    "energy_noiseless": ideal, "energy_noisy": value,
                    "noise_induced_energy_shift": value - ideal,
                    "excess_over_noiseless_full": value - baseline_energy,
                    "physical_threshold_violations": int(qresult.physical_threshold_violations) if method.startswith("qiprune") else None,
                    "qfi_selection_time_ms": float(qfi_ms) if method.startswith("diagonal") else None,
                    "probe_selection_time_ms": float(probe_ms) if method.startswith("single") else None,
                })
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "meta": {
            "protocol": "noiseless_training_and_selection__posttraining_real_gate_deletion__post_gate_depolarizing_evaluation",
            "seed": args.seed, "vqe_depth": args.vqe_depth, "vqe_steps_run": optimized.steps_run,
            "n_qubits": n_qubits, "J": J, "h": h,
            "vqe_converged": optimized.converged, "vqe_final_window_range": optimized.final_window_range,
            "deltas": args.deltas, "noise_strengths": args.noise_strengths,
            "methods": args.methods,
            "noise_semantics": "deleting_a_gate_removes_its_post_gate_noise_event",
        }, "results": rows,
    }, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
