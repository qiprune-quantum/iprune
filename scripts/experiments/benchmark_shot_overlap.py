"""Finite-shot compute-uncompute overlap audit of frozen iPrune paths.

The deletion path is selected with exact state vectors. This experiment tests
measurement of its accepted local overlaps, not shot-based gate selection or
execution on hardware. A one-sided Clopper-Pearson interval with Bonferroni
allocation gives a simultaneous upper bound on the realized path sum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from scripts.experiments.benchmark_selection_cost import checkpoint_path
from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry
from iprune.gates import build_hea_gates_from_parameters
from iprune.pruning_true import q_iPrune_iterative_identity_audited


def one_sided_lower_fidelity(successes: np.ndarray, shots: int, tail: float) -> np.ndarray:
    lower = beta.ppf(tail, successes, shots - successes + 1)
    return np.where(successes == 0, 0.0, lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[3001, 3002, 3003])
    parser.add_argument("--shots", type=int, nargs="+", default=[100, 1000, 10000, 100000])
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--delta", type=float, default=1.2)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if not (0 < args.alpha < 1) or args.trials < 1 or any(s < 1 for s in args.shots):
        raise ValueError("alpha must be in (0,1), with positive trials and shots")
    rows = []
    for n in args.sizes:
        for seed in args.seeds:
            checkpoint = np.load(checkpoint_path(args.root, n, seed))
            params = np.asarray(checkpoint["params"], dtype=np.float64)
            all_gates, baseline = build_hea_gates_from_parameters(
                params, include_cnot=True, include_identity_rot=True
            )
            initial_state = np.zeros((1, 2**n), dtype=np.complex128)
            initial_state[0, 0] = 1.0
            config = PaperProtocolConfig(
                device="default.qubit", n_qubits=n, depth=4, M=1, seed=seed,
                gamma=0.05, alpha=0.6, qbeta=0.0, delta=args.delta,
                rot_pool_k=1, rot_pool_sigma=0.0, include_cnot=True,
            )
            _, _, geometry, _, epsilon = make_q_geometry(config)
            result = q_iPrune_iterative_identity_audited(
                all_gates=all_gates, baseline_circuit=baseline,
                n_qubits=n, device_name="default.qubit", task_states=initial_state,
                Gq_diag=geometry, q_epsilon=epsilon, physical_epsilon=epsilon,
            )
            positions = list(result.accepted_positions)
            angles = np.asarray([result.physical_angle_max_by_position[p] for p in positions])
            L = len(angles)
            exact_path = min(1.0, float(np.sum(np.sin(angles))))
            actual = float(result.actual_trace_distance_max)
            assert actual <= exact_path + 1e-10
            print(f"PATH N={n} seed={seed} L={L} exact={exact_path:.4f} actual={actual:.4f}", flush=True)
            if L == 0:
                continue
            fidelities = np.cos(angles) ** 2
            for shots in args.shots:
                rng = np.random.default_rng(n * 10**9 + seed * 10**5 + shots)
                successes = rng.binomial(shots, fidelities, size=(args.trials, L))
                lower = one_sided_lower_fidelity(successes, shots, args.alpha / L)
                upper_angles = np.sqrt(np.clip(1.0 - lower, 0.0, 1.0))
                bounds = np.minimum(1.0, np.sum(upper_angles, axis=1))
                row = {
                    "n_qubits": n, "seed": seed, "delta": args.delta,
                    "shots_per_overlap": shots, "measurement_trials": args.trials,
                    "familywise_alpha": args.alpha, "accepted_gates": L,
                    "overlap_measurements_per_trial": L,
                    "shots_per_trial": shots * L,
                    "exact_path_bound": exact_path,
                    "actual_trace_distance": actual,
                    "median_shot_bound": float(np.median(bounds)),
                    "bound_p05": float(np.quantile(bounds, 0.05)),
                    "bound_p95": float(np.quantile(bounds, 0.95)),
                    "nontrivial_fraction": float(np.mean(bounds < 1.0)),
                    "coverage_of_exact_path": float(np.mean(bounds + 1e-12 >= exact_path)),
                    "coverage_of_actual_trace": float(np.mean(bounds + 1e-12 >= actual)),
                }
                rows.append(row)
                print(f"  S={shots} median={row['median_shot_bound']:.4f} useful={row['nontrivial_fraction']:.3f}", flush=True)
    output = args.root / f"shot_overlap_audit_delta_{str(args.delta).replace('.', 'p')}.json"
    output.write_text(json.dumps({"design": "exact-path selection, sampled compute-uncompute overlap audit", "rows": rows}, indent=2), encoding="utf-8")
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
