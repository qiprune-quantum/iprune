from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

import numpy as np
import pennylane as qml

from iprune.experiments.protocol import simulate_outputs
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters_with_diagnostics, tfim_hamiltonian
from iprune.experiments.vqe_finetune import fine_tune_vqe_circuit
from iprune.gates import GateOp, build_hea_gates_from_parameters
from scripts.experiments.run_vqe_qiprune_vs_gradient_seed import _atomic_new_json, _evaluate, _without_positions
from scripts.legacy.sweep_all_v2 import _circuit_signature


def _rotation_positions(circuit: Sequence[GateOp]) -> List[int]:
    return [int(gate.pos_id) for gate in circuit if gate.gate_type == "Rot"]


def _state_energy(state: np.ndarray, hamiltonian: np.ndarray) -> float:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    return float(np.real(np.vdot(vector, hamiltonian @ vector)))


def sequential_objective_deletion(
    *,
    baseline_circuit: Sequence[GateOp],
    initial_state: np.ndarray,
    n_qubits: int,
    delete_count: int,
    hamiltonian: np.ndarray,
    device: str,
) -> tuple[List[GateOp], List[int], List[Dict[str, Any]], float]:
    """Greedily delete the gate giving the lowest current-circuit energy.

    Candidate energies are recomputed after every accepted deletion.  This is
    therefore a dynamic direct-objective baseline, rather than a ranking frozen
    on the original checkpoint.
    """
    current = list(baseline_circuit)
    deleted: List[int] = []
    audit: List[Dict[str, Any]] = []
    started = perf_counter()

    for step in range(delete_count):
        state_before = simulate_outputs(current, initial_state, n_qubits, device)
        energy_before = _state_energy(state_before, hamiltonian)
        candidates: List[tuple[float, int, List[GateOp]]] = []
        for position in _rotation_positions(current):
            candidate_circuit = _without_positions(current, [position])
            candidate_state = simulate_outputs(
                candidate_circuit, initial_state, n_qubits, device
            )
            candidate_energy = _state_energy(candidate_state, hamiltonian)
            candidates.append((candidate_energy, position, candidate_circuit))

        if not candidates:
            raise RuntimeError(
                f"Requested {delete_count} deletions but no rotation remains at step {step}."
            )
        energy_after, chosen_position, chosen_circuit = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        audit.append(
            {
                "step": step + 1,
                "energy_before": energy_before,
                "chosen_position": int(chosen_position),
                "energy_after": float(energy_after),
                "objective_increase": float(energy_after - energy_before),
                "candidate_count": len(candidates),
            }
        )
        deleted.append(int(chosen_position))
        current = chosen_circuit

    return current, sorted(deleted), audit, (perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic sequential direct-energy baseline on saved TFIM protocols."
    )
    parser.add_argument("--source_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--device", default="default.qubit")
    args = parser.parse_args()

    source = json.loads(args.source_json.read_text(encoding="utf-8"))
    q_rows = [
        row
        for row in source["results"]
        if row["method"] == "qiprune_iterative_hybrid_path"
    ]
    if not q_rows:
        raise ValueError("No qiprune_iterative_hybrid_path rows in source JSON.")
    first = q_rows[0]
    seed = int(first["seed"])
    n_qubits = int(first["n_qubits"])
    depth = int(first["vqe_depth"])
    J = float(first.get("J", 1.0))
    h = float(first.get("h", 1.0))

    config = VQEConfig(
        n_qubits=n_qubits,
        depth=depth,
        steps=int(first["vqe_steps"]),
        lr=float(first["vqe_lr"]),
        seed=seed,
        J=J,
        h=h,
        min_steps=200,
        convergence_window=20,
        convergence_tol=5e-5,
    )
    optimization = optimize_vqe_parameters_with_diagnostics(config, device=args.device)
    if not optimization.converged:
        raise RuntimeError(
            f"Reconstructed VQE did not converge for seed {seed}: "
            f"{optimization.final_window_range:.6g}."
        )

    _, baseline_circuit = build_hea_gates_from_parameters(
        optimization.params, include_cnot=True, include_identity_rot=True
    )
    signature = _circuit_signature(list(baseline_circuit))
    expected_signature = str(first["baseline_circuit_signature"])
    if signature != expected_signature:
        raise RuntimeError(
            f"Checkpoint reconstruction mismatch for seed {seed}: "
            f"{signature} != {expected_signature}."
        )

    initial_state = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    baseline_state = simulate_outputs(
        baseline_circuit, initial_state, n_qubits, args.device
    )
    control = fine_tune_vqe_circuit(
        circuit=baseline_circuit,
        n_qubits=n_qubits,
        J=J,
        h=h,
        steps=int(first["finetune_steps"]),
        lr=float(first["finetune_lr"]),
        device=args.device,
    )
    control_state = control.final_state.reshape(1, -1)
    hamiltonian = np.asarray(
        qml.matrix(
            tfim_hamiltonian(n_qubits, J, h), wire_order=range(n_qubits)
        ),
        dtype=np.complex128,
    )

    cache: Dict[int, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    for q_row in sorted(q_rows, key=lambda row: float(row["delta"])):
        delete_count = int(q_row["matched_delete_count"])
        if delete_count not in cache:
            circuit, deleted, audit, selection_time_ms = sequential_objective_deletion(
                baseline_circuit=baseline_circuit,
                initial_state=initial_state,
                n_qubits=n_qubits,
                delete_count=delete_count,
                hamiltonian=hamiltonian,
                device=args.device,
            )
            cache[delete_count] = {
                "circuit": circuit,
                "deleted": deleted,
                "audit": audit,
                "selection_time_ms": selection_time_ms,
            }
        selected = cache[delete_count]
        row = _evaluate(
            method="dynamic_sequential_energy_probe_matched_l",
            compact_circuit=selected["circuit"],
            deleted_pos_ids=selected["deleted"],
            baseline_circuit=baseline_circuit,
            baseline_state=baseline_state,
            control_state=control_state,
            energy_original=float(optimization.energy_history[-1]),
            energy_control=float(control.energy_history[-1]),
            energy_initial=float(optimization.energy_history[0]),
            n_qubits=n_qubits,
            J=J,
            h=h,
            finetune_steps=int(first["finetune_steps"]),
            finetune_lr=float(first["finetune_lr"]),
            device=args.device,
        )
        row.update(
            {
                "experiment": "tfim_dynamic_sequential_objective_baseline",
                "seed": seed,
                "n_qubits": n_qubits,
                "vqe_depth": depth,
                "delta": float(q_row["delta"]),
                "matched_delete_count": delete_count,
                "selection_time_ms": float(selected["selection_time_ms"]),
                "selection_audit": selected["audit"],
            }
        )
        results.append(row)
        print(
            f"[OK] seed={seed} delta={row['delta']:g} K={delete_count} "
            f"gap={row['energy_gap_finetuned_vs_control']:.6f}",
            flush=True,
        )

    payload = {
        "meta": {
            "protocol": (
                "dynamic sequential direct-energy deletion; recompute every "
                "remaining one-gate deletion on the current circuit; choose "
                "minimum energy; match iPrune realized L; same recovery budget"
            ),
            "source_json": str(args.source_json.resolve()),
            "seed": seed,
            "n_qubits": n_qubits,
            "vqe_depth": depth,
            "baseline_circuit_signature": signature,
            "vqe_steps_run": int(optimization.steps_run),
            "vqe_converged": bool(optimization.converged),
        },
        "results": results,
    }
    _atomic_new_json(args.output_json, payload)
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
