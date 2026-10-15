"""H2 VQE supplementary benchmark for dynamic q=1 identity deletion.

This is deliberately a supplementary physical benchmark.  It uses
the same post-training, real-rotation deletion and matched-realized-K logic as
the TFIM runner, but does not relabel an H2 result as a TFIM result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs, trace_distance_stats
from iprune.experiments.qfi_gate_pruning import diagonal_qfi_gate_saliency_scores
from iprune.experiments.tasks_vqe import hea_ansatz
from iprune.experiments.vqe_finetune import rebuild_circuit_with_rot_params
from iprune.gates import GateOp, build_hea_gates_from_parameters
from iprune.pruning_true import q_iPrune_iterative_identity_audited
from scripts.legacy.sweep_all_v2 import _logical_resource_metrics


def h2_hamiltonian(bond_length: float):
    coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, bond_length])
    molecule = qml.qchem.Molecule(["H", "H"], coordinates)
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(molecule, method="dhf")
    if int(n_qubits) != 4:
        raise RuntimeError(f"Expected 4-qubit H2 Hamiltonian, got {n_qubits} qubits.")
    return hamiltonian, int(n_qubits)


@dataclass
class OptimizedVQE:
    params: np.ndarray
    history: np.ndarray
    converged: bool
    steps_run: int
    final_window_range: float


def optimize_h2(*, hamiltonian, n_qubits: int, depth: int, seed: int, steps: int, lr: float, min_steps: int, window: int, tolerance: float, device: str) -> OptimizedVQE:
    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(0.0, 2.0 * np.pi, size=(depth, n_qubits, 3)), requires_grad=True)
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def energy(current):
        hea_ansatz(current, n_qubits, depth)
        return qml.expval(hamiltonian)

    optimizer = qml.GradientDescentOptimizer(stepsize=lr)
    history = [float(energy(params))]
    converged = False
    final_range = float("inf")
    for _ in range(steps):
        params = optimizer.step(energy, params)
        history.append(float(energy(params)))
        if len(history) - 1 >= min_steps and len(history) >= window:
            final_range = float(np.ptp(np.asarray(history[-window:], dtype=np.float64)))
            if final_range <= tolerance:
                converged = True
                break
    if not np.isfinite(final_range):
        final_range = float(np.ptp(np.asarray(history[-window:], dtype=np.float64)))
    return OptimizedVQE(np.asarray(params, dtype=np.float64), np.asarray(history), converged, len(history) - 1, final_range)


def compact(circuit: Sequence[GateOp], positions: Sequence[int]) -> List[GateOp]:
    deleted = {int(position) for position in positions}
    return [gate for gate in circuit if int(gate.pos_id) not in deleted]


def rot_positions(circuit: Sequence[GateOp]) -> List[int]:
    return [int(gate.pos_id) for gate in circuit if gate.gate_type == "Rot"]


def magnitude(gate: GateOp) -> float:
    return float(np.linalg.norm(np.angle(np.exp(1j * np.asarray(gate.params, dtype=np.float64)))))


def energy_of_state(state: np.ndarray, matrix: np.ndarray) -> float:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    return float(np.real(np.vdot(vector, matrix @ vector)))


def fine_tune(*, circuit: Sequence[GateOp], hamiltonian, n_qubits: int, steps: int, lr: float, device: str):
    template = list(circuit)
    initial = np.asarray([gate.params for gate in template if gate.gate_type == "Rot"], dtype=np.float64).reshape(-1, 3) if any(g.gate_type == "Rot" for g in template) else np.zeros((0, 3))
    dev = qml.device(device, wires=n_qubits)

    def apply(current) -> None:
        index = 0
        for gate in template:
            if gate.gate_type == "Rot":
                qml.Rot(*current[index], wires=int(gate.wires[0]))
                index += 1
            else:
                qml.CNOT(wires=list(gate.wires))

    if len(initial) == 0:
        @qml.qnode(dev)
        def fixed_energy():
            apply(np.zeros((0, 3)))
            return qml.expval(hamiltonian)
        @qml.qnode(dev)
        def fixed_state():
            apply(np.zeros((0, 3)))
            return qml.state()
        value = float(fixed_energy())
        return template, np.asarray(fixed_state()), np.full(steps + 1, value), 0
    params = pnp.array(initial, requires_grad=True)
    @qml.qnode(dev, interface="autograd")
    def energy(current):
        apply(current)
        return qml.expval(hamiltonian)
    @qml.qnode(dev, interface="autograd")
    def state(current):
        apply(current)
        return qml.state()
    optimizer = qml.GradientDescentOptimizer(stepsize=lr)
    history = [float(energy(params))]
    for _ in range(steps):
        params = optimizer.step(energy, params)
        history.append(float(energy(params)))
    return rebuild_circuit_with_rot_params(template, np.asarray(params)), np.asarray(state(params)), np.asarray(history), len(initial)


def evaluate(*, method: str, positions: Sequence[int], baseline: Sequence[GateOp], base_state: np.ndarray, control_state: np.ndarray, hamiltonian, matrix: np.ndarray, n_qubits: int, original_energy: float, control_energy: float, finetune_steps: int, finetune_lr: float, device: str) -> dict:
    pruned = compact(baseline, positions)
    initial = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial[0, 0] = 1.0
    one_state = simulate_outputs(pruned, initial, n_qubits, device)
    tuned, tuned_state, history, trainable = fine_tune(circuit=pruned, hamiltonian=hamiltonian, n_qubits=n_qubits, steps=finetune_steps, lr=finetune_lr, device=device)
    one_energy = energy_of_state(one_state, matrix)
    tuned_energy = float(history[-1])
    before = _logical_resource_metrics(list(baseline), n_qubits)
    after = _logical_resource_metrics(pruned, n_qubits)
    trace_mean, trace_max = trace_distance_stats(base_state, one_state)
    return {
        "method": method, "deleted_pos_ids": sorted(int(p) for p in positions), "matched_delete_count": len(positions),
        "energy_original_optimized": original_energy, "energy_full_control_after_extra_steps": control_energy,
        "energy_pruned_one_shot": one_energy, "energy_pruned_finetuned": tuned_energy,
        "energy_gap_one_shot_vs_control": one_energy - control_energy,
        "energy_gap_finetuned_vs_control": tuned_energy - control_energy,
        "abs_dE_one_shot_vs_original": abs(one_energy - original_energy),
        "actual_gate_prune_ratio": len(positions) / len(baseline), "gate_count_before": len(baseline), "gate_count_after": len(pruned),
        "logical_depth_before": before["logical_depth"], "logical_depth_after": after["logical_depth"],
        "one_qubit_gate_count_before": before["one_qubit"], "one_qubit_gate_count_after": after["one_qubit"],
        "two_qubit_gate_count_before": before["two_qubit"], "two_qubit_gate_count_after": after["two_qubit"],
        "trainable_rotations_after": trainable, "trace_dist_one_shot_vs_original_mean": float(trace_mean), "trace_dist_one_shot_vs_original_max": float(trace_max),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 matched-K physical deletion pilot.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--bond_length", type=float, default=0.74)
    parser.add_argument("--vqe_steps", type=int, default=1600)
    parser.add_argument("--vqe_lr", type=float, default=0.1)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--finetune_lr", type=float, default=0.05)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    parser.add_argument("--device", default="default.qubit")
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    hamiltonian, n_qubits = h2_hamiltonian(args.bond_length)
    optimized = optimize_h2(hamiltonian=hamiltonian, n_qubits=n_qubits, depth=args.depth, seed=args.seed, steps=args.vqe_steps, lr=args.vqe_lr, min_steps=200, window=20, tolerance=5e-5, device=args.device)
    if not optimized.converged:
        raise RuntimeError("H2 baseline did not meet the frozen convergence rule.")
    all_gates, baseline = build_hea_gates_from_parameters(optimized.params, include_cnot=True, include_identity_rot=True)
    initial = np.zeros((1, 2**n_qubits), dtype=np.complex128); initial[0, 0] = 1.0
    base_state = simulate_outputs(baseline, initial, n_qubits, args.device)
    matrix = np.asarray(qml.matrix(hamiltonian, wire_order=range(n_qubits)), dtype=np.complex128)
    operator_norm = float(np.max(np.abs(np.linalg.eigvalsh(matrix))))
    original_energy = float(optimized.history[-1])
    _, control_state, control_history, _ = fine_tune(circuit=baseline, hamiltonian=hamiltonian, n_qubits=n_qubits, steps=args.finetune_steps, lr=args.finetune_lr, device=args.device)
    control_energy = float(control_history[-1])
    rotations = [gate for gate in baseline if gate.gate_type == "Rot"]
    mag_rank = sorted(rot_positions(baseline), key=lambda p: (magnitude(next(g for g in rotations if g.pos_id == p)), p))
    random_rank = list(np.random.default_rng(args.seed + 20260913).permutation(rot_positions(baseline)))
    started = perf_counter(); qfi_rank, _, _, qfi_ms = diagonal_qfi_gate_saliency_scores(circuit=baseline, task_states=initial, n_qubits=n_qubits, device=args.device); qfi_rank = list(qfi_rank); qfi_ms = float(qfi_ms)
    started = perf_counter(); probe = {p: abs(energy_of_state(simulate_outputs(compact(baseline, [p]), initial, n_qubits, args.device), matrix) - original_energy) for p in rot_positions(baseline)}; probe_rank = sorted(probe, key=lambda p: (probe[p], p)); probe_ms = (perf_counter() - started) * 1000.0
    rows = []
    for delta in args.deltas:
        cfg = PaperProtocolConfig(device=args.device, n_qubits=n_qubits, depth=args.depth, M=1, seed=args.seed, gamma=0.0, alpha=0.6, qbeta=0.0, delta=delta, rot_pool_k=1, rot_pool_sigma=0.0, include_cnot=True)
        _, q, weights, Mq, epsilon = make_q_geometry(cfg)
        qresult = q_iPrune_iterative_identity_audited(all_gates=all_gates, baseline_circuit=baseline, n_qubits=n_qubits, device_name=args.device, task_states=initial, Gq_diag=weights, q_epsilon=epsilon, physical_epsilon=epsilon)
        selections = {"qiprune_iterative_hybrid_path": qresult.accepted_positions, "diagonal_qfi_quadratic_saliency_matched_l": qfi_rank[:len(qresult.accepted_positions)], "parameter_magnitude_matched_l": mag_rank[:len(qresult.accepted_positions)], "random_matched_l": random_rank[:len(qresult.accepted_positions)], "single_gate_energy_probe_matched_l": probe_rank[:len(qresult.accepted_positions)]}
        for method, positions in selections.items():
            row = evaluate(method=method, positions=positions, baseline=baseline, base_state=base_state, control_state=control_state, hamiltonian=hamiltonian, matrix=matrix, n_qubits=n_qubits, original_energy=original_energy, control_energy=control_energy, finetune_steps=args.finetune_steps, finetune_lr=args.finetune_lr, device=args.device)
            if method.startswith("qiprune"):
                observable_bound = 2.0 * operator_norm * min(1.0, qresult.path_trace_sum_max)
                if row["abs_dE_one_shot_vs_original"] > observable_bound + 1e-9:
                    raise AssertionError("H2 one-shot energy drift exceeds the realized-path observable bound.")
            else:
                observable_bound = None
            row.update({"experiment": "h2_vqe_dynamic_q1_physical_deletion_pilot", "seed": args.seed, "delta": delta, "q": q, "epsilon_q": epsilon, "h2_bond_length_angstrom": args.bond_length, "vqe_depth": args.depth, "vqe_steps_run": optimized.steps_run, "vqe_converged": optimized.converged, "vqe_final_window_range": optimized.final_window_range, "physical_threshold_violations": int(qresult.physical_threshold_violations) if method.startswith("qiprune") else None, "path_trace_sum_max": float(qresult.path_trace_sum_max) if method.startswith("qiprune") else None, "actual_trace_distance_max": float(qresult.actual_trace_distance_max) if method.startswith("qiprune") else None, "selection_time_ms": qfi_ms if method.startswith("diagonal") else probe_ms if method.startswith("single") else None})
            row["hamiltonian_operator_norm"] = operator_norm
            row["realized_path_observable_bound"] = observable_bound
            rows.append(row)
    payload = {"meta": {"protocol": "H2_vqe_real_rotation_identity_deletion_matched_realized_K", "seed": args.seed, "bond_length_angstrom": args.bond_length, "methods": sorted({r['method'] for r in rows}), "deltas": args.deltas, "vqe_converged": optimized.converged, "vqe_steps_run": optimized.steps_run, "vqe_final_window_range": optimized.final_window_range}, "results": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
