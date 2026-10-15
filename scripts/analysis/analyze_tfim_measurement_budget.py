"""Auditable shot-accounting comparison for TFIM overlap and energy selectors.

This is a protocol-level measurement-budget calculation, not a hardware
benchmark.  It counts circuits and measurement bases needed to evaluate every
rotation once.  The TFIM ring Hamiltonian has N mutually commuting ZZ terms and
N mutually commuting X terms, hence two standard measurement bases per energy
evaluation when terms are grouped by basis.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots < 1:
        raise ValueError("shots must be positive")

    payload = json.loads(args.audit.read_text(encoding="utf-8"))
    selected = [row for row in payload["rows"] if row["shots_per_overlap"] == args.shots]
    by_n = defaultdict(list)
    for row in selected:
        by_n[int(row["n_qubits"])].append(row)

    rows = []
    for n, observations in sorted(by_n.items()):
        seeds = {int(row["seed"]) for row in observations}
        if len(seeds) != len(observations):
            raise ValueError(f"Duplicate or incomplete seed records for N={n}.")
        rotations = 4 * n
        candidates_including_baseline = rotations + 1
        mean_l = float(np.mean([row["accepted_gates"] for row in observations]))
        overlap_selection = rotations * args.shots
        overlap_audit = mean_l * args.shots
        energy_grouped = 2 * candidates_including_baseline * args.shots
        energy_termwise = (2 * n) * candidates_including_baseline * args.shots
        rows.append({
            "n_qubits": n,
            "independent_seeds": len(seeds),
            "rotations_inspected": rotations,
            "mean_accepted_deletions": mean_l,
            "shots_per_setting": args.shots,
            "iprune_overlap_settings": rotations,
            "iprune_selection_shots": overlap_selection,
            "iprune_fixed_path_audit_shots_mean": overlap_audit,
            "energy_probe_grouped_settings": 2 * candidates_including_baseline,
            "energy_probe_grouped_shots": energy_grouped,
            "energy_probe_termwise_settings": (2 * n) * candidates_including_baseline,
            "energy_probe_termwise_shots": energy_termwise,
            "grouped_energy_to_overlap_shot_ratio": energy_grouped / overlap_selection,
        })

    result = {
        "scope": "TFIM, depth 4, delta=0.5; exact-state path selection; conventional compute-uncompute overlap and energy measurement accounting",
        "assumptions": [
            "One overlap circuit and one measurement setting per iPrune candidate rotation.",
            "One shared baseline circuit plus one deleted circuit per direct energy-probe candidate.",
            "TFIM energy uses two commuting measurement bases: all Z_i Z_{i+1} terms in Z and all X_i terms in X.",
            "No state-preparation, transpilation, mitigation, adaptive allocation, or confidence-control overhead is included.",
            "The fixed-path audit count is reported separately and must not be added to selection when selection outcomes are retained for certification.",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"N={row['n_qubits']} P={row['rotations_inspected']} "
            f"Lbar={row['mean_accepted_deletions']:.3f} "
            f"overlap={row['iprune_selection_shots']} "
            f"energy_grouped={row['energy_probe_grouped_shots']} "
            f"ratio={row['grouped_energy_to_overlap_shot_ratio']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
