from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pennylane as qml

from iprune.experiments.tasks_vqe import tfim_hamiltonian
from iprune.gates import GateOp, build_hea_gates_from_parameters
from iprune.qgeometry import build_Gq_diag_popcount, epsilon_from_delta, q_from_lambda
from scripts.experiments.run_q_matched_k_tfim_seed import _identity_candidate, _pools_by_position
from scripts.experiments.run_q_sensitivity_selection_diagnostic import (
    _compute_prefix_states_fast,
    _rank_positions,
    _weighted_angles,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = (
    PROJECT_DIR
    / "results"
    / "q_sensitivity_upgrade_20260810"
    / "tfim_matched_k"
    / "rows"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "lambda_noise_grid_20260812" / "tfim"
CHANNELS = ("depolarizing", "amplitude_damping")


def _atomic_new_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite completed row: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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


def _apply_noise(channel: str, strength: float, wires: Sequence[int]) -> None:
    for wire in wires:
        if channel == "depolarizing":
            qml.DepolarizingChannel(strength, wires=int(wire))
        elif channel == "amplitude_damping":
            qml.AmplitudeDamping(strength, wires=int(wire))
        else:
            raise ValueError(channel)


def _noisy_density_runner(
    circuit: Sequence[GateOp],
    n_qubits: int,
    channel: str,
    strength: float,
    noise_semantics: str,
):
    device = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(device, interface=None)
    def run(deleted_positions: Sequence[int]):
        deleted = set(int(position) for position in deleted_positions)
        for gate in circuit:
            if int(gate.pos_id) in deleted:
                if noise_semantics == "fixed_scheduled_noise":
                    _apply_noise(channel, strength, gate.wires)
                continue
            gate.apply()
            _apply_noise(channel, strength, gate.wires)
        return qml.density_matrix(wires=range(n_qubits))

    return run


def _mixed_trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    difference = np.asarray(rho, dtype=np.complex128) - np.asarray(sigma, dtype=np.complex128)
    difference = 0.5 * (difference + difference.conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def _density_energy(rho: np.ndarray, hamiltonian: np.ndarray) -> float:
    return float(np.real(np.trace(np.asarray(rho) @ hamiltonian)))


def _load_source(source_path: Path, deltas: Sequence[float]) -> tuple[np.ndarray, Dict[float, int]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    target_k: Dict[float, int] = {}
    for delta in deltas:
        matches = [
            row
            for row in source["results"]
            if float(row["delta"]) == float(delta) and float(row["qbeta"]) == 0.0
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one q=1 source row for delta={delta}, found {len(matches)}")
        target_k[float(delta)] = int(matches[0]["matched_delete_count"])
    return np.asarray(source["optimized_parameters"], dtype=np.float64), target_k


def _score_grid(
    *,
    all_gates: Sequence[GateOp],
    baseline_circuit: Sequence[GateOp],
    n_qubits: int,
    lambdas: Sequence[float],
    beta: float,
) -> tuple[Dict[float, Dict[int, float]], Dict[int, float]]:
    initial_state = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    prefixes = _compute_prefix_states_fast(baseline_circuit, initial_state, n_qubits)
    pools = _pools_by_position(all_gates)
    physical_weights = np.ones(2**n_qubits, dtype=np.float64)
    q_scores: Dict[float, Dict[int, float]] = {float(lam): {} for lam in lambdas}
    physical_scores: Dict[int, float] = {}
    q_weights = {
        float(lam): build_Gq_diag_popcount(q_from_lambda(float(lam), beta), n_qubits)[0]
        for lam in lambdas
    }
    for gate in baseline_circuit:
        if gate.gate_type != "Rot":
            continue
        position = int(gate.pos_id)
        identity = _identity_candidate(pools[position])
        physical_scores[position] = float(
            np.max(
                _weighted_angles(
                    gate, identity, prefixes[position], physical_weights, n_qubits
                )
            )
        )
        for lam in lambdas:
            q_scores[float(lam)][position] = float(
                np.max(
                    _weighted_angles(
                        gate,
                        identity,
                        prefixes[position],
                        q_weights[float(lam)],
                        n_qubits,
                    )
                )
            )
    return q_scores, physical_scores


def _evaluate_seed(
    *,
    seed: int,
    source_path: Path,
    output_path: Path,
    lambdas: Sequence[float],
    beta: float,
    deltas: Sequence[float],
    channel_strengths: Mapping[str, Sequence[float]],
    n_qubits: int,
    J: float,
    h: float,
    noise_semantics: str,
) -> None:
    parameters, target_k = _load_source(source_path, deltas)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        parameters, include_cnot=True, include_identity_rot=True
    )
    q_scores, physical_scores = _score_grid(
        all_gates=all_gates,
        baseline_circuit=baseline_circuit,
        n_qubits=n_qubits,
        lambdas=lambdas,
        beta=beta,
    )
    hamiltonian = np.asarray(
        qml.matrix(tfim_hamiltonian(n_qubits, J=J, h=h), wire_order=range(n_qubits)),
        dtype=np.complex128,
    )
    selection_rows: List[Dict[str, Any]] = []
    selected_by_delta_lambda: Dict[tuple[float, float], List[int]] = {}
    for delta in deltas:
        epsilon = epsilon_from_delta(float(delta), 1.0)
        eligible = {
            position
            for position, score in physical_scores.items()
            if score <= epsilon + 1e-12
        }
        matched_k = min(target_k[float(delta)], len(eligible))
        for lam in lambdas:
            order = [
                position
                for position in _rank_positions(q_scores[float(lam)])
                if position in eligible
            ]
            deleted = sorted(order[:matched_k])
            selected_by_delta_lambda[(float(delta), float(lam))] = deleted
            selection_rows.append(
                {
                    "seed": seed,
                    "delta": float(delta),
                    "lambda": float(lam),
                    "q": q_from_lambda(float(lam), beta),
                    "target_k": target_k[float(delta)],
                    "matched_k": matched_k,
                    "physical_eligible": len(eligible),
                    "deleted_pos_ids": deleted,
                }
            )

    result_rows: List[Dict[str, Any]] = []
    for channel, strengths in channel_strengths.items():
        for strength in strengths:
            run_density = _noisy_density_runner(
                baseline_circuit,
                n_qubits,
                channel,
                float(strength),
                noise_semantics,
            )
            baseline_density = np.asarray(run_density([]), dtype=np.complex128)
            baseline_energy = _density_energy(baseline_density, hamiltonian)
            density_cache: Dict[tuple[int, ...], np.ndarray] = {}
            for delta in deltas:
                for lam in lambdas:
                    deleted = selected_by_delta_lambda[(float(delta), float(lam))]
                    key = tuple(deleted)
                    if key not in density_cache:
                        density_cache[key] = np.asarray(run_density(deleted), dtype=np.complex128)
                    pruned_density = density_cache[key]
                    pruned_energy = _density_energy(pruned_density, hamiltonian)
                    result_rows.append(
                        {
                            "seed": seed,
                            "channel": channel,
                            "noise_strength": float(strength),
                            "delta": float(delta),
                            "lambda": float(lam),
                            "q": q_from_lambda(float(lam), beta),
                            "matched_k": len(deleted),
                            "gate_count_before": len(baseline_circuit),
                            "gate_count_after": len(baseline_circuit) - len(deleted),
                            "actual_gate_prune_ratio": len(deleted) / max(1, len(baseline_circuit)),
                            "deleted_pos_ids": deleted,
                            "baseline_noisy_energy": baseline_energy,
                            "pruned_noisy_energy": pruned_energy,
                            "abs_energy_drift": abs(pruned_energy - baseline_energy),
                            "trace_distance": _mixed_trace_distance(
                                baseline_density, pruned_density
                            ),
                        }
                    )
            print(
                f"[OK] seed={seed} channel={channel} strength={strength:g}", flush=True
            )

    _atomic_new_json(
        output_path,
        {
            "meta": {
                "experiment_family": "lambda_noise_grid_20260812",
                "protocol": "same_checkpoint_same_K_same_physical_certificate_lambda_grid_true_gate_deletion",
                "seed": seed,
                "source": str(source_path.resolve()),
                "lambdas": list(lambdas),
                "beta": beta,
                "q_mapping": "q(lambda)=exp(beta*(1-lambda))",
                "important_alias": "lambda=1 maps exactly to q=1; these are one control, not two methods",
                "deltas": list(deltas),
                "channel_strengths": {
                    key: list(value) for key, value in channel_strengths.items()
                },
                "noise_placement": "after each scheduled or retained gate on each touched wire",
                "noise_semantics": noise_semantics,
                "deletion_semantics": (
                    "deleting a gate retains its scheduled noise event"
                    if noise_semantics == "fixed_scheduled_noise"
                    else "deleting a gate also removes its associated noise event"
                ),
                "selection_states": "ideal |0000> baseline prefix states",
                "no_finetuning": True,
            },
            "selection_rows": selection_rows,
            "results": result_rows,
        },
    )


def _rebuild(output_dir: Path, seeds: Sequence[int]) -> None:
    payloads = [
        json.loads((output_dir / "rows" / f"tfim_seed{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    selection_rows = [row for payload in payloads for row in payload["selection_rows"]]
    result_rows = [row for payload in payloads for row in payload["results"]]
    _atomic_csv(output_dir / "selection_level.csv", selection_rows)
    _atomic_csv(output_dir / "seed_level.csv", result_rows)
    _atomic_json(
        output_dir / "checkpoint.json",
        {
            "meta": payloads[0]["meta"] | {"seeds": list(seeds)},
            "completed_seeds": list(seeds),
            "selection_rows": len(selection_rows),
            "result_rows": len(result_rows),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TFIM lambda x physical-noise matched-K grid.")
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.2, 0.5, 0.8, 1.0])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--deltas", nargs="+", type=float, default=[1.2, 1.5])
    parser.add_argument(
        "--depolarizing_strengths", nargs="+", type=float, default=[0.001, 0.005, 0.01, 0.05]
    )
    parser.add_argument(
        "--amplitude_damping_strengths", nargs="+", type=float, default=[0.001, 0.01, 0.05]
    )
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument(
        "--noise_semantics",
        choices=["gate_removed_noise_removed", "fixed_scheduled_noise"],
        default="gate_removed_noise_removed",
    )
    args = parser.parse_args()

    lambdas = sorted(set(float(value) for value in args.lambdas))
    if any(value < 0.0 or value > 1.0 for value in lambdas):
        raise ValueError("lambda values must be in [0, 1]")
    if 1.0 not in lambdas:
        raise ValueError("lambda grid must include 1.0 as the exact q=1 control")
    if args.beta <= 0.0:
        raise ValueError("beta must be positive")
    channel_strengths = {
        "depolarizing": sorted(set(float(value) for value in args.depolarizing_strengths)),
        "amplitude_damping": sorted(
            set(float(value) for value in args.amplitude_damping_strengths)
        ),
    }
    if any(
        value < 0.0 or value > 1.0
        for strengths in channel_strengths.values()
        for value in strengths
    ):
        raise ValueError("noise strengths must be in [0, 1]")

    output_dir = args.output_dir.resolve()
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        destination = rows_dir / f"tfim_seed{seed}.json"
        if destination.exists():
            print(f"[SKIP] seed={seed} existing={destination}", flush=True)
            continue
        _evaluate_seed(
            seed=seed,
            source_path=args.source_dir.resolve() / f"tfim_seed{seed}.json",
            output_path=destination,
            lambdas=lambdas,
            beta=args.beta,
            deltas=sorted(set(float(value) for value in args.deltas)),
            channel_strengths=channel_strengths,
            n_qubits=args.n_qubits,
            J=args.J,
            h=args.h,
            noise_semantics=args.noise_semantics,
        )
    _rebuild(output_dir, args.seeds)
    print(f"Saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
