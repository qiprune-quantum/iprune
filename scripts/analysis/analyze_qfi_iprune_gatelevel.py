"""Seed-level statistics for the full-QFI versus iPrune gate experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 20000):
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def paired_summary(values: list[float], rng: np.random.Generator):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "seed_bootstrap_95_ci_for_mean": bootstrap_mean_ci(array, rng),
        "per_seed": [float(x) for x in array],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("runs/qfi_iprune_gatelevel_n4_20260916/gate_level_results.csv"),
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--selection-angle", choices=("physical_fs_angle", "iprune_q_angle"),
        default="physical_fs_angle",
        help="Main-text gate ranking uses the physical Fubini-Study angle.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    rng = np.random.default_rng(20260916)
    measures = {
        "qfi_redundancy_vs_energy_damage": "qfi_null_participation_mean",
        "iprune_q_angle_vs_energy_damage": "iprune_q_angle",
        "physical_fs_angle_vs_energy_damage": "physical_fs_angle",
        "iprune_q_angle_vs_trace_distance": "iprune_q_angle",
    }
    targets = {
        "qfi_redundancy_vs_energy_damage": "single_delete_energy_abs",
        "iprune_q_angle_vs_energy_damage": "single_delete_energy_abs",
        "physical_fs_angle_vs_energy_damage": "single_delete_energy_abs",
        "iprune_q_angle_vs_trace_distance": "single_delete_trace_distance",
    }

    correlations: dict[str, list[float]] = {name: [] for name in measures}
    qfi_top_damage: list[float] = []
    iprune_top_damage: list[float] = []
    overlap_counts: list[int] = []

    for _, group in frame.groupby("seed", sort=True):
        for name, predictor in measures.items():
            correlations[name].append(
                float(spearmanr(group[predictor], group[targets[name]]).statistic)
            )
        qfi_selected = group.nlargest(args.top_k, "qfi_null_participation_mean")
        iprune_selected = group.nsmallest(args.top_k, args.selection_angle)
        qfi_top_damage.append(float(qfi_selected["single_delete_energy_abs"].mean()))
        iprune_top_damage.append(float(iprune_selected["single_delete_energy_abs"].mean()))
        overlap_counts.append(len(set(qfi_selected.position) & set(iprune_selected.position)))

    qfi_array = np.asarray(qfi_top_damage)
    iprune_array = np.asarray(iprune_top_damage)
    delta = qfi_array - iprune_array
    result = {
        "analysis_unit": "within-seed; 12 independently trained circuits",
        "gate_rows_are_not_treated_as_independent_replicates": True,
        "seed_count": int(frame.seed.nunique()),
        "gates_per_seed": int(frame.groupby("seed").size().iloc[0]),
        "selection_angle": args.selection_angle,
        "within_seed_spearman": {
            name: paired_summary(values, rng) for name, values in correlations.items()
        },
        "lowest_damage_selection": {
            "k_gates_per_seed": args.top_k,
            "qfi_highest_redundancy_mean_damage": paired_summary(qfi_top_damage, rng),
            "iprune_lowest_angle_mean_damage": paired_summary(iprune_top_damage, rng),
            "qfi_minus_iprune_damage": paired_summary(delta.tolist(), rng),
            "paired_wilcoxon_two_sided_p": float(wilcoxon(qfi_array, iprune_array).pvalue),
            "selection_overlap_mean_of_k": float(np.mean(overlap_counts)),
            "selection_overlap_per_seed": overlap_counts,
        },
    }
    suffix = "physical" if args.selection_angle == "physical_fs_angle" else "q_weighted"
    output = args.input.parent / f"statistical_summary_{suffix}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
