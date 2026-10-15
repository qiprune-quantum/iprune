from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt


CLASSIFICATION_METRICS = [
    ("actual_gate_prune_ratio", "Actual gate prune ratio"),
    ("L", "Actual deleted positions (L)"),
    ("trace_dist_max", "Maximum trace distance"),
    ("rhs_raw", "Trace-distance RHS"),
    ("dq_max_over_replaced", "Maximum q-angle over deleted gates"),
    ("acc_drop", "Accuracy drop (baseline - pruned)"),
]

VQE_METRICS = [
    ("actual_gate_prune_ratio", "Actual gate prune ratio"),
    ("L", "Actual deleted positions (L)"),
    ("trace_dist_max", "Maximum trace distance"),
    ("rhs_raw", "Trace-distance RHS"),
    ("dq_max_over_replaced", "Maximum q-angle over deleted gates"),
    ("dE_max", "Maximum absolute energy deviation"),
]


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("meta", {}).get("schema_version") != 2:
        raise ValueError("plot_from_json.py expects corrected schema_version=2 results.")
    return list(payload["results"])


def save_plot(
    *,
    x: List[float],
    series: List[tuple[str, List[float], List[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure()
    for label, y, yerr in series:
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(series) > 1:
        plt.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_classification(rows: List[Dict[str, Any]], output_dir: str) -> None:
    for dataset in ["mnist49", "fashion_sb", "bas"]:
        dataset_rows = [
            row
            for row in rows
            if row.get("task") == "classification" and row.get("dataset") == dataset
        ]
        if not dataset_rows:
            continue
        sigmas = sorted({float(row["rot_pool_sigma"]) for row in dataset_rows})
        deltas = sorted({float(row["delta"]) for row in dataset_rows})
        n_qubits = int(dataset_rows[0]["n_qubits"])

        for metric, ylabel in CLASSIFICATION_METRICS:
            series = []
            for delta in deltas:
                by_sigma = {
                    float(row["rot_pool_sigma"]): row
                    for row in dataset_rows
                    if abs(float(row["delta"]) - delta) < 1e-12
                }
                y = [float(by_sigma[sigma][metric]) for sigma in sigmas]
                yerr = [float(by_sigma[sigma].get(f"{metric}_std") or 0.0) for sigma in sigmas]
                series.append((f"delta={delta:g}", y, yerr))

            save_plot(
                x=sigmas,
                series=series,
                xlabel="Rotation-pool perturbation sigma",
                ylabel=ylabel,
                title=f"{dataset}: corrected true pruning ({n_qubits} qubits)",
                output_path=os.path.join(output_dir, f"{dataset}_{metric}.png"),
            )


def plot_vqe(rows: List[Dict[str, Any]], output_dir: str) -> None:
    vqe_rows = [
        row for row in rows if row.get("task") == "vqe" and row.get("dataset") == "tfim"
    ]
    if not vqe_rows:
        return
    by_delta = {float(row["delta"]): row for row in vqe_rows}
    deltas = sorted(by_delta)
    n_qubits = int(vqe_rows[0]["n_qubits"])

    for metric, ylabel in VQE_METRICS:
        y = [float(by_delta[delta][metric]) for delta in deltas]
        yerr = [float(by_delta[delta].get(f"{metric}_std") or 0.0) for delta in deltas]
        save_plot(
            x=deltas,
            series=[("optimized TFIM ansatz", y, yerr)],
            xlabel="Per-replacement tolerance delta",
            ylabel=ylabel,
            title=f"TFIM VQE: corrected true pruning ({n_qubits} qubits)",
            output_path=os.path.join(output_dir, f"vqe_{metric}.png"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot corrected q-iPrune V2 results.")
    parser.add_argument(
        "--input",
        default="./results/qiprune_sweep_v2_fixed.json",
        help="Corrected schema-version-2 result file.",
    )
    parser.add_argument("--output-dir", default="./results/plots_v2_fixed")
    args = parser.parse_args()

    rows = load_results(args.input)
    plot_classification(rows, args.output_dir)
    plot_vqe(rows, args.output_dir)
    print(f"Saved corrected V2 plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
