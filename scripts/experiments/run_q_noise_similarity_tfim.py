from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pennylane as qml
from scipy.stats import kendalltau, spearmanr

from iprune.experiments.tasks_vqe import tfim_hamiltonian
from iprune.gates import GateOp, build_hea_gates_from_parameters
from iprune.qgeometry import build_Gq_diag_popcount, lambda_from_gamma, q_from_lambda
from scripts.experiments.run_q_matched_k_tfim_seed import _identity_candidate, _pools_by_position
from scripts.experiments.run_q_sensitivity_selection_diagnostic import _compute_prefix_states_fast, _weighted_angles


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = (
    PROJECT_DIR
    / "results"
    / "q_sensitivity_upgrade_20260810"
    / "tfim_matched_k"
    / "rows"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "q_noise_validation_20260811" / "tfim"
VARIANTS = ("q1", "q_lambda", "fixed_q")
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


def _correlation(x: Sequence[float], y: Sequence[float], kind: str) -> float:
    value = spearmanr(x, y).statistic if kind == "spearman" else kendalltau(x, y).statistic
    return float(value) if np.isfinite(value) else 0.0


def _mixed_trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    difference = np.asarray(rho, dtype=np.complex128) - np.asarray(sigma, dtype=np.complex128)
    difference = 0.5 * (difference + difference.conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def _density_energy(rho: np.ndarray, hamiltonian: np.ndarray) -> float:
    return float(np.real(np.trace(np.asarray(rho) @ hamiltonian)))


def _apply_noise(channel: str, strength: float, wires: Sequence[int]) -> None:
    for wire in wires:
        if channel == "depolarizing":
            qml.DepolarizingChannel(strength, wires=int(wire))
        elif channel == "amplitude_damping":
            qml.AmplitudeDamping(strength, wires=int(wire))
        else:
            raise ValueError(channel)


def _noisy_density_runner(
    circuit: Sequence[GateOp], n_qubits: int, channel: str, strength: float
):
    device = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(device, interface=None)
    def run(deleted_position: int):
        for gate in circuit:
            if int(gate.pos_id) == int(deleted_position):
                continue
            gate.apply()
            _apply_noise(channel, strength, gate.wires)
        return qml.density_matrix(wires=range(n_qubits))

    return run


def _variant_q_values(strength: float, alpha: float, fixed_q: float) -> Dict[str, float]:
    lam = lambda_from_gamma(strength, alpha)
    return {
        "q1": 1.0,
        "q_lambda": q_from_lambda(lam, 1.0),
        "fixed_q": float(fixed_q),
    }


def _ideal_q_scores(
    *,
    all_gates: Sequence[GateOp],
    baseline_circuit: Sequence[GateOp],
    n_qubits: int,
    q_values: Mapping[str, float],
) -> Dict[str, Dict[int, float]]:
    initial_state = np.zeros((1, 2**n_qubits), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    prefixes = _compute_prefix_states_fast(baseline_circuit, initial_state, n_qubits)
    pools = _pools_by_position(all_gates)
    weights = {
        variant: build_Gq_diag_popcount(q, n_qubits)[0]
        for variant, q in q_values.items()
    }
    scores: Dict[str, Dict[int, float]] = {variant: {} for variant in q_values}
    for gate in baseline_circuit:
        if gate.gate_type != "Rot":
            continue
        position = int(gate.pos_id)
        identity = _identity_candidate(pools[position])
        for variant in q_values:
            angles = _weighted_angles(
                gate, identity, prefixes[position], weights[variant], n_qubits
            )
            scores[variant][position] = float(np.max(angles))
    return scores


def _evaluate_seed(
    *,
    seed: int,
    source_path: Path,
    noise_strengths: Sequence[float],
    channels: Sequence[str],
    n_qubits: int,
    alpha: float,
    fixed_q: float,
    J: float,
    h: float,
    top_ks: Sequence[int],
) -> Dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    parameters = np.asarray(source["optimized_parameters"], dtype=np.float64)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        parameters, include_cnot=True, include_identity_rot=True
    )
    positions = sorted(int(gate.pos_id) for gate in baseline_circuit if gate.gate_type == "Rot")
    hamiltonian = np.asarray(
        qml.matrix(tfim_hamiltonian(n_qubits, J=J, h=h), wire_order=range(n_qubits)),
        dtype=np.complex128,
    )
    position_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for channel in channels:
        for strength in noise_strengths:
            q_values = _variant_q_values(float(strength), alpha, fixed_q)
            scores = _ideal_q_scores(
                all_gates=all_gates,
                baseline_circuit=baseline_circuit,
                n_qubits=n_qubits,
                q_values=q_values,
            )
            run_density = _noisy_density_runner(
                baseline_circuit, n_qubits, channel, float(strength)
            )
            baseline_density = np.asarray(run_density(-1), dtype=np.complex128)
            baseline_energy = _density_energy(baseline_density, hamiltonian)
            truth: Dict[int, Dict[str, float]] = {}
            for position in positions:
                deleted_density = np.asarray(run_density(position), dtype=np.complex128)
                deleted_energy = _density_energy(deleted_density, hamiltonian)
                truth[position] = {
                    "abs_energy_drift": abs(deleted_energy - baseline_energy),
                    "trace_distance": _mixed_trace_distance(baseline_density, deleted_density),
                }

            energy = [truth[position]["abs_energy_drift"] for position in positions]
            trace = [truth[position]["trace_distance"] for position in positions]
            for variant in VARIANTS:
                score_vector = [scores[variant][position] for position in positions]
                order = sorted(positions, key=lambda position: (scores[variant][position], position))
                row: Dict[str, Any] = {
                    "seed": seed,
                    "channel": channel,
                    "noise_strength": float(strength),
                    "variant": variant,
                    "q": q_values[variant],
                    "positions": len(positions),
                    "baseline_noisy_energy": baseline_energy,
                    "spearman_energy": _correlation(score_vector, energy, "spearman"),
                    "kendall_energy": _correlation(score_vector, energy, "kendall"),
                    "spearman_trace": _correlation(score_vector, trace, "spearman"),
                    "kendall_trace": _correlation(score_vector, trace, "kendall"),
                }
                for top_k in top_ks:
                    selected = order[: min(int(top_k), len(order))]
                    selected_energy = [truth[position]["abs_energy_drift"] for position in selected]
                    oracle = sorted(energy)[: len(selected)]
                    row[f"top{top_k}_positions"] = ",".join(str(value) for value in selected)
                    row[f"top{top_k}_mean_energy_drift"] = float(np.mean(selected_energy))
                    row[f"top{top_k}_regret"] = float(np.mean(selected_energy) - np.mean(oracle))
                summary_rows.append(row)
                for position in positions:
                    position_rows.append(
                        {
                            "seed": seed,
                            "channel": channel,
                            "noise_strength": float(strength),
                            "variant": variant,
                            "q": q_values[variant],
                            "position": position,
                            "score": scores[variant][position],
                            "abs_energy_drift": truth[position]["abs_energy_drift"],
                            "trace_distance": truth[position]["trace_distance"],
                        }
                    )
            print(
                f"[OK] seed={seed} channel={channel} p={strength:g} "
                f"q_lambda={q_values['q_lambda']:.6f}",
                flush=True,
            )
    return {
        "meta": {
            "seed": seed,
            "source": str(source_path.resolve()),
            "channels": list(channels),
            "noise_strengths": list(noise_strengths),
            "variants": list(VARIANTS),
            "alpha": alpha,
            "fixed_q": fixed_q,
            "q_lambda_mapping": "lambda=clip(1-alpha*p,0,1); q=exp(1-lambda)",
            "noise_placement": "after every retained gate, independently on every touched wire",
            "deletion_semantics": "deleting a gate also removes its associated post-gate noise events",
            "q_score_states": "ideal |0000> baseline prefix states; current q-overlap is pure-state-defined",
            "no_finetuning": True,
        },
        "position_rows": position_rows,
        "summary_rows": summary_rows,
    }


def _bootstrap(values: Sequence[float], seed: int, draws: int = 20000) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }


def _summarize(rows: Sequence[Mapping[str, Any]], top_ks: Sequence[int]) -> List[Dict[str, Any]]:
    q1 = {
        (int(row["seed"]), str(row["channel"]), float(row["noise_strength"])): row
        for row in rows
        if row["variant"] == "q1"
    }
    metrics = ["spearman_energy", "kendall_energy", "spearman_trace", "kendall_trace"]
    metrics.extend(f"top{top_k}_mean_energy_drift" for top_k in top_ks)
    metrics.extend(f"top{top_k}_regret" for top_k in top_ks)
    output: List[Dict[str, Any]] = []
    group_keys = sorted(
        {(str(row["channel"]), float(row["noise_strength"]), str(row["variant"])) for row in rows}
    )
    for group_index, (channel, strength, variant) in enumerate(group_keys):
        group = [
            row
            for row in rows
            if row["channel"] == channel
            and float(row["noise_strength"]) == strength
            and row["variant"] == variant
        ]
        item: Dict[str, Any] = {
            "channel": channel,
            "noise_strength": strength,
            "variant": variant,
            "q": float(group[0]["q"]),
            "seeds": len(group),
        }
        for metric_index, metric in enumerate(metrics):
            values = [float(row[metric]) for row in group]
            estimate = _bootstrap(values, 20260811 + 100 * group_index + metric_index)
            item[f"{metric}_mean"] = estimate["mean"]
            item[f"{metric}_ci_low"] = estimate["ci_low"]
            item[f"{metric}_ci_high"] = estimate["ci_high"]
            differences = []
            for row in group:
                base = q1[(int(row["seed"]), channel, strength)]
                differences.append(float(row[metric]) - float(base[metric]))
            paired = _bootstrap(differences, 20261811 + 100 * group_index + metric_index)
            item[f"{metric}_paired_vs_q1"] = paired["mean"]
            item[f"{metric}_paired_vs_q1_ci_low"] = paired["ci_low"]
            item[f"{metric}_paired_vs_q1_ci_high"] = paired["ci_high"]
        output.append(item)
    return output


def _rebuild(output_dir: Path, seeds: Sequence[int], top_ks: Sequence[int]) -> None:
    payloads = [
        json.loads((output_dir / "rows" / f"tfim_seed{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    positions = [row for payload in payloads for row in payload["position_rows"]]
    seed_rows = [row for payload in payloads for row in payload["summary_rows"]]
    summary = _summarize(seed_rows, top_ks)
    _atomic_csv(output_dir / "position_level.csv", positions)
    _atomic_csv(output_dir / "seed_level.csv", seed_rows)
    _atomic_csv(output_dir / "summary_by_noise.csv", summary)
    _atomic_json(
        output_dir / "checkpoint.json",
        {
            "meta": payloads[0]["meta"] | {"seeds": list(seeds)},
            "summary_by_noise": summary,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TFIM q noise-similarity validation.")
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--channels", nargs="+", choices=CHANNELS, default=list(CHANNELS))
    parser.add_argument(
        "--noise_strengths", nargs="+", type=float, default=[0.0, 0.001, 0.003, 0.01, 0.03, 0.05]
    )
    parser.add_argument("--top_ks", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--fixed_q", type=float, default=1.35)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    args = parser.parse_args()
    if any(value < 0.0 or value > 1.0 for value in args.noise_strengths):
        raise ValueError("Noise strengths must be in [0,1]")
    if args.fixed_q <= 0.0:
        raise ValueError("fixed_q must be positive")

    output_dir = args.output_dir.resolve()
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        destination = rows_dir / f"tfim_seed{seed}.json"
        if destination.exists():
            print(f"[SKIP] seed={seed} existing={destination}", flush=True)
            continue
        payload = _evaluate_seed(
            seed=seed,
            source_path=args.source_dir.resolve() / f"tfim_seed{seed}.json",
            noise_strengths=sorted(set(float(value) for value in args.noise_strengths)),
            channels=args.channels,
            n_qubits=args.n_qubits,
            alpha=args.alpha,
            fixed_q=args.fixed_q,
            J=args.J,
            h=args.h,
            top_ks=args.top_ks,
        )
        _atomic_new_json(destination, payload)
    _rebuild(output_dir, args.seeds, args.top_ks)
    print(f"Saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
