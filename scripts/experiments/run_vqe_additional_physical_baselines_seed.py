from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from iprune.experiments.protocol import simulate_outputs
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters
from iprune.experiments.vqe_finetune import fine_tune_vqe_circuit
from iprune.gates import GateOp, build_hea_gates_from_parameters
from scripts.experiments.run_vqe_qiprune_vs_gradient_seed import _atomic_new_json, _evaluate
from scripts.legacy.sweep_all_v2 import _circuit_signature


QIPRUNE_METHOD = "qiprune_true_identity"


def _load_qiprune_rows(path: Path, seed: int) -> Dict[float, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("results", [])
        if int(row["seed"]) == seed and row["method"] == QIPRUNE_METHOD
    ]
    if not rows:
        raise ValueError(f"No q-iPrune matched-K rows for seed={seed} in {path}")
    return {float(row["delta"]): row for row in rows}


def _rotation_positions(circuit: Sequence[GateOp]) -> List[int]:
    return [int(gate.pos_id) for gate in circuit if gate.gate_type == "Rot"]


def _circular_parameter_magnitude(gate: GateOp) -> float:
    if gate.gate_type != "Rot" or gate.params is None:
        raise ValueError("Parameter-magnitude baseline requires Rot gates")
    parameters = np.asarray(gate.params, dtype=np.float64)
    wrapped = np.angle(np.exp(1j * parameters))
    return float(np.linalg.norm(wrapped))


def _without_positions(circuit: Sequence[GateOp], positions: Sequence[int]) -> List[GateOp]:
    deleted = set(int(position) for position in positions)
    return [gate for gate in circuit if gate.pos_id not in deleted]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched-K random and parameter-magnitude physical gate-deletion baselines."
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 0.9, 1.0, 1.2, 1.5, 2.0])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqe_depth", type=int, default=6)
    parser.add_argument("--vqe_steps", type=int, default=40)
    parser.add_argument("--vqe_lr", type=float, default=0.1)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--finetune_lr", type=float, default=0.05)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--random_seed_offset", type=int, default=20260807)
    args = parser.parse_args()

    source_rows = _load_qiprune_rows(args.source_checkpoint, args.seed)
    missing = [delta for delta in args.deltas if float(delta) not in source_rows]
    if missing:
        raise ValueError(f"Missing source deltas for seed={args.seed}: {missing}")

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
    _, baseline_circuit = build_hea_gates_from_parameters(
        optimized_params, include_cnot=True, include_identity_rot=True
    )
    baseline_signature = _circuit_signature(baseline_circuit)
    source_signatures = {str(row["baseline_circuit_signature"]) for row in source_rows.values()}
    if source_signatures != {baseline_signature}:
        raise RuntimeError(
            "Reconstructed optimized circuit does not match the source matched-K experiment: "
            f"reconstructed={baseline_signature}, source={sorted(source_signatures)}"
        )

    initial_state = np.zeros((1, 2**args.n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    baseline_state = simulate_outputs(
        baseline_circuit, initial_state, args.n_qubits, args.device
    )
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
    energy_original = float(baseline_history[-1])
    energy_control = float(control.energy_history[-1])

    reference_row = next(iter(source_rows.values()))
    if abs(energy_original - float(reference_row["energy_original_optimized"])) > 1e-9:
        raise RuntimeError("Reconstructed optimized energy differs from the source experiment")
    if abs(energy_control - float(reference_row["energy_full_control_after_extra_steps"])) > 1e-9:
        raise RuntimeError("Reconstructed matched control differs from the source experiment")

    rotations = [gate for gate in baseline_circuit if gate.gate_type == "Rot"]
    rot_pos_ids = _rotation_positions(baseline_circuit)

    random_started = perf_counter()
    random_rng = np.random.default_rng(args.seed + args.random_seed_offset)
    random_ranking = [int(value) for value in random_rng.permutation(rot_pos_ids)]
    random_time_ms = (perf_counter() - random_started) * 1000.0

    magnitude_started = perf_counter()
    magnitude_scores = {
        int(gate.pos_id): _circular_parameter_magnitude(gate) for gate in rotations
    }
    magnitude_ranking = sorted(
        rot_pos_ids, key=lambda position: (magnitude_scores[position], position)
    )
    magnitude_time_ms = (perf_counter() - magnitude_started) * 1000.0

    rows: List[Dict[str, Any]] = []
    for delta in args.deltas:
        source = source_rows[float(delta)]
        delete_count = int(source["matched_delete_count"])
        if delete_count > len(rot_pos_ids):
            raise ValueError(f"Matched K={delete_count} exceeds {len(rot_pos_ids)} Rot gates")

        selections = [
            (
                "random_rotation_matched_k",
                sorted(random_ranking[:delete_count]),
                random_time_ms,
                {"random_seed": args.seed + args.random_seed_offset},
            ),
            (
                "parameter_circular_magnitude_matched_k",
                sorted(magnitude_ranking[:delete_count]),
                magnitude_time_ms,
                {
                    "parameter_score": "L2 norm of each Euler angle wrapped to [-pi, pi]",
                    "selected_score_max": (
                        max(magnitude_scores[position] for position in magnitude_ranking[:delete_count])
                        if delete_count
                        else 0.0
                    ),
                },
            ),
        ]
        for method, deleted, selection_time_ms, audit in selections:
            row = _evaluate(
                method=method,
                compact_circuit=_without_positions(baseline_circuit, deleted),
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
            row.update(
                {
                    "experiment_family": "supplementary_upgrade",
                    "experiment": "vqe_additional_physical_baselines_matched_k",
                    "task": "vqe",
                    "dataset": "tfim",
                    "seed": args.seed,
                    "n_qubits": args.n_qubits,
                    "vqe_depth": args.vqe_depth,
                    "vqe_steps": args.vqe_steps,
                    "vqe_lr": args.vqe_lr,
                    "finetune_steps": args.finetune_steps,
                    "finetune_lr": args.finetune_lr,
                    "delta": float(delta),
                    "epsilon_q": float(source["epsilon_q"]),
                    "lambda": float(source["lambda"]),
                    "q": float(source["q"]),
                    "Mq": float(source["Mq"]),
                    "matched_delete_count": delete_count,
                    "matched_reference_method": QIPRUNE_METHOD,
                    "selection_time_ms": selection_time_ms,
                    **audit,
                }
            )
            rows.append(row)
            print(
                f"[OK] seed={args.seed} delta={delta:g} K={delete_count} "
                f"method={method} gap={row['energy_gap_finetuned_vs_control']:.6f}",
                flush=True,
            )

    payload = {
        "meta": {
            "experiment_family": "supplementary_upgrade",
            "protocol": "same_optimized_tfim_circuit_matched_K_random_parameter_same_finetune_budget",
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "source_baseline_signature": baseline_signature,
            "seed": args.seed,
            "deltas": args.deltas,
            "methods": [
                "random_rotation_matched_k",
                "parameter_circular_magnitude_matched_k",
            ],
            "expected_rows": len(args.deltas) * 2,
        },
        "ranking_audit": {
            "rot_pos_ids": rot_pos_ids,
            "random_ranking": random_ranking,
            "parameter_magnitude_ranking": magnitude_ranking,
            "parameter_magnitude_scores": magnitude_scores,
        },
        "results": rows,
    }
    _atomic_new_json(args.output_json, payload)
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
