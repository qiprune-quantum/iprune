from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pennylane as qml

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs, trace_distance_stats
from iprune.experiments.tasks_vqe import VQEConfig, hea_ansatz, optimize_vqe_parameters_with_diagnostics, tfim_hamiltonian
from iprune.gates import GateOp, build_hea_gates_from_parameters


DEFAULT_SEEDS = list(range(3001, 3013))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-QFI null-space versus task-conditioned iPrune gate-level audit."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=Path("runs/tfim_pruning_transition_formal_dense_20260915"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--vqe-steps", type=int, default=1200)
    parser.add_argument("--finite-difference-step", type=float, default=1e-5)
    parser.add_argument("--qfi-relative-tol", type=float, default=1e-8)
    parser.add_argument("--device", default="default.qubit")
    return parser.parse_args()


def circuit_signature(circuit: Sequence[GateOp]) -> str:
    pieces = []
    for gate in circuit:
        params = "" if gate.params is None else ",".join(f"{value:.17g}" for value in gate.params)
        pieces.append(f"{gate.pos_id}|{gate.gate_type}|{gate.wires}|{params}")
    return hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()[:16]


def without_position(circuit: Sequence[GateOp], position: int) -> list[GateOp]:
    return [gate for gate in circuit if int(gate.pos_id) != int(position)]


def pure_state_angle(left: np.ndarray, right: np.ndarray) -> float:
    overlap = abs(np.vdot(left, right)) / (np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.arccos(np.clip(overlap, 0.0, 1.0)))


def q_weighted_angle(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    numerator = abs(np.vdot(left, weights * right))
    left_norm = float(np.real(np.vdot(left, weights * left)))
    right_norm = float(np.real(np.vdot(right, weights * right)))
    overlap = numerator / np.sqrt(max(left_norm * right_norm, 1e-30))
    return float(np.arccos(np.clip(overlap, 0.0, 1.0)))


def full_pure_state_qfi(
    params: np.ndarray,
    *,
    n_qubits: int,
    depth: int,
    device: str,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    if step <= 0.0:
        raise ValueError("finite-difference step must be positive")
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev)
    def state_qnode(current: np.ndarray):
        hea_ansatz(current, n_qubits, depth)
        return qml.state()

    params = np.asarray(params, dtype=np.float64)
    flat = params.reshape(-1)
    psi = np.asarray(state_qnode(params), dtype=np.complex128)
    derivatives = np.empty((len(flat), len(psi)), dtype=np.complex128)
    for index in range(len(flat)):
        plus = flat.copy()
        minus = flat.copy()
        plus[index] += step
        minus[index] -= step
        psi_plus = np.asarray(state_qnode(plus.reshape(params.shape)), dtype=np.complex128)
        psi_minus = np.asarray(state_qnode(minus.reshape(params.shape)), dtype=np.complex128)
        derivatives[index] = (psi_plus - psi_minus) / (2.0 * step)

    gram = derivatives.conj() @ derivatives.T
    parallel = derivatives.conj() @ psi
    projected = gram - np.outer(parallel, parallel.conj())
    qfi = 4.0 * np.real(projected)
    qfi = 0.5 * (qfi + qfi.T)
    return psi, qfi


def load_reference(reference_dir: Path, seed: int) -> dict[str, Any]:
    path = reference_dir / f"seed{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["meta"].get("status") != "formal":
        raise ValueError(f"Reference is not formal: {path}")
    return payload


def analyze_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    n_qubits = int(args.n_qubits)
    depth = int(args.depth)
    state_dim = 2**n_qubits
    rotation_count = n_qubits * depth
    qfi_parameter_count = rotation_count * 3

    reference = load_reference(args.reference_dir, seed)
    reference_row = reference["results"][0]
    optimization = optimize_vqe_parameters_with_diagnostics(
        VQEConfig(
            n_qubits=n_qubits,
            depth=depth,
            steps=args.vqe_steps,
            lr=0.1,
            seed=seed,
            J=1.0,
            h=1.0,
            min_steps=200,
            convergence_window=20,
            convergence_tol=5e-5,
        ),
        device=args.device,
    )
    if not optimization.converged:
        raise RuntimeError(f"seed {seed} did not converge")
    all_gates, circuit = build_hea_gates_from_parameters(
        optimization.params, include_cnot=True, include_identity_rot=True
    )
    signature = circuit_signature(circuit)
    expected_signature = str(reference_row["baseline_circuit_signature"])
    if signature != expected_signature:
        raise AssertionError(
            f"seed {seed} baseline signature mismatch: {signature} != {expected_signature}"
        )
    if int(optimization.steps_run) != int(reference_row["vqe_steps_run"]):
        raise AssertionError(f"seed {seed} VQE stopping step mismatch")

    initial = np.zeros((1, state_dim), dtype=np.complex128)
    initial[0, 0] = 1.0
    baseline_state = simulate_outputs(circuit, initial, n_qubits, args.device)[0]
    hamiltonian = np.asarray(
        qml.matrix(tfim_hamiltonian(n_qubits, 1.0, 1.0), wire_order=range(n_qubits)),
        dtype=np.complex128,
    )
    baseline_energy = float(np.real(np.vdot(baseline_state, hamiltonian @ baseline_state)))
    if not np.isclose(
        baseline_energy,
        float(reference_row["energy_original_optimized"]),
        atol=1e-10,
    ):
        raise AssertionError(f"seed {seed} baseline energy mismatch")

    started = perf_counter()
    qfi_state, qfi = full_pure_state_qfi(
        optimization.params,
        n_qubits=n_qubits,
        depth=depth,
        device=args.device,
        step=args.finite_difference_step,
    )
    if 1.0 - abs(np.vdot(qfi_state, baseline_state)) > 1e-10:
        raise AssertionError(f"seed {seed} QFI state mismatch")
    eigenvalues, eigenvectors = np.linalg.eigh(qfi)
    max_eigenvalue = float(np.max(eigenvalues))
    tolerance = max(1e-10, args.qfi_relative_tol * max(1.0, max_eigenvalue))
    null_mask = eigenvalues <= tolerance
    null_vectors = eigenvectors[:, null_mask]
    null_projector_diag = np.sum(null_vectors * null_vectors, axis=1)
    ranks = {}
    for relative in (1e-6, 1e-8, 1e-10):
        current_tol = max(1e-10, relative * max(1.0, max_eigenvalue))
        ranks[f"{relative:.0e}"] = int(np.sum(eigenvalues > current_tol))

    cfg = PaperProtocolConfig(
        n_qubits=n_qubits,
        depth=depth,
        M=1,
        seed=seed,
        gamma=0.05,
        alpha=0.6,
        # Retain the earlier q-weighted angle as a diagnostic comparison only;
        # the physical-angle column and main selector use q=1.
        qbeta=1.0,
        delta=1.0,
        rot_pool_k=1,
        rot_pool_sigma=0.0,
        include_cnot=True,
    )
    _lam, q_value, q_weights, _Mq, _epsilon = make_q_geometry(cfg)
    prefix_state = initial.copy()
    gate_rows = []
    rotation_index = 0
    for gate in circuit:
        after_gate = simulate_outputs([gate], prefix_state, n_qubits, args.device)
        if gate.gate_type == "Rot":
            physical_angle = pure_state_angle(after_gate[0], prefix_state[0])
            task_q_angle = q_weighted_angle(after_gate[0], prefix_state[0], q_weights)
            deleted_state = simulate_outputs(
                without_position(circuit, int(gate.pos_id)), initial, n_qubits, args.device
            )[0]
            deleted_energy = float(
                np.real(np.vdot(deleted_state, hamiltonian @ deleted_state))
            )
            trace_distance = trace_distance_stats(
                baseline_state.reshape(1, -1), deleted_state.reshape(1, -1)
            )[1]
            block = slice(3 * rotation_index, 3 * rotation_index + 3)
            gate_null = null_projector_diag[block]
            gate_rows.append(
                {
                    "seed": seed,
                    "position": int(gate.pos_id),
                    "layer": int(gate.layer),
                    "qubit": int(gate.wires[0]),
                    "qfi_null_participation_mean": float(np.mean(gate_null)),
                    "qfi_null_participation_sum": float(np.sum(gate_null)),
                    "qfi_null_coord_0": float(gate_null[0]),
                    "qfi_null_coord_1": float(gate_null[1]),
                    "qfi_null_coord_2": float(gate_null[2]),
                    "iprune_q_angle": task_q_angle,
                    "physical_fs_angle": physical_angle,
                    "single_delete_trace_distance": float(trace_distance),
                    "single_delete_energy_signed": deleted_energy - baseline_energy,
                    "single_delete_energy_abs": abs(deleted_energy - baseline_energy),
                }
            )
            rotation_index += 1
        prefix_state = after_gate
    if len(gate_rows) != rotation_count:
        raise AssertionError(f"seed {seed} expected {rotation_count} rotation gates")

    return {
        "meta": {
            "status": "formal",
            "experiment": "full_qfi_nullspace_vs_task_conditioned_iprune_gatelevel",
            "seed": seed,
            "n_qubits": n_qubits,
            "depth": depth,
            "rotation_gates": rotation_count,
            "qfi_parameters": qfi_parameter_count,
            "qfi_rank": int(np.sum(eigenvalues > tolerance)),
            "qfi_nullity": int(np.sum(null_mask)),
            "qfi_tolerance": tolerance,
            "qfi_relative_tolerance": args.qfi_relative_tol,
            "qfi_rank_sensitivity": ranks,
            "qfi_min_eigenvalue": float(np.min(eigenvalues)),
            "qfi_max_eigenvalue": max_eigenvalue,
            "qfi_smallest_positive_eigenvalue": float(np.min(eigenvalues[~null_mask])),
            "finite_difference_step": args.finite_difference_step,
            "q_geometry_q": q_value,
            "baseline_signature": signature,
            "baseline_energy": baseline_energy,
            "vqe_steps_run": int(optimization.steps_run),
            "runtime_seconds": perf_counter() - started,
        },
        "qfi_eigenvalues": [float(value) for value in eigenvalues],
        "gates": gate_rows,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("meta", {}).get("status") == "formal":
            return
        raise FileExistsError(f"Refusing to overwrite non-formal result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def export_combined(output_dir: Path, seeds: Sequence[int]) -> None:
    rows = []
    metas = []
    for seed in seeds:
        payload = json.loads((output_dir / f"seed{seed}.json").read_text(encoding="utf-8"))
        if payload["meta"].get("status") != "formal":
            raise ValueError(f"seed {seed} is not formal")
        metas.append(payload["meta"])
        rows.extend(payload["gates"])
    fieldnames = list(rows[0])
    with (output_dir / "gate_level_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "formal",
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "gate_rows": len(rows),
        "qfi_ranks": [meta["qfi_rank"] for meta in metas],
        "qfi_nullities": [meta["qfi_nullity"] for meta in metas],
        "rank_sensitivity": [meta["qfi_rank_sensitivity"] for meta in metas],
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(f"runs/qfi_iprune_gatelevel_n{args.n_qubits}_20260916")
    seeds = sorted(set(args.seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        path = args.output_dir / f"seed{seed}.json"
        if path.exists():
            print(f"[seed {seed}] SKIP existing formal result", flush=True)
            continue
        payload = analyze_seed(args, seed)
        atomic_write_json(path, payload)
        meta = payload["meta"]
        print(
            f"[seed {seed}] rank={meta['qfi_rank']} nullity={meta['qfi_nullity']} "
            f"runtime={meta['runtime_seconds']:.2f}s",
            flush=True,
        )
    export_combined(args.output_dir, seeds)
    print(f"Saved gate-level rows to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
