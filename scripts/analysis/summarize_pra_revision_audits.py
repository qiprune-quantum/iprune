"""Summarize archived raw records used in the PRA revision audit tables.

The script is deliberately read-only: it does not rerun optimization and makes
the aggregation rules used in the manuscript explicit.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"


def load_records(directory: Path):
    records = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        # Manifests and aggregate audit files are metadata, not result rows.
        if "results" not in payload:
            continue
        records.extend(payload["results"])
    return records


def pm(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}" if len(values) > 1 else f"{values.mean():.4f}"


def phase_tightness():
    records = load_records(RUNS / "tfim_n4_physical_q1_20260924")
    grouped = defaultdict(list)
    for row in records:
        bound = min(1.0, row["realized_path_trace_sum_max"])
        actual = row["actual_trace_distance_max"]
        grouped[row["delta"]].append((actual, bound, bound - actual, actual / bound if bound else 0.0))
    print("[certificate tightness: 12 seed paths per threshold]")
    for delta in sorted(grouped):
        values = np.asarray(grouped[delta], dtype=float)
        print(
            f"delta={delta:.1f}; D={pm(values[:,0])}; B={pm(values[:,1])}; "
            f"B-D={pm(values[:,2])}; D/B={pm(values[:,3])}; B<1: {(values[:,1] < 1.0).sum()}/{len(values)}"
        )


def violation_events():
    order = load_records(RUNS / "tfim_order_greedy_ablation_20260916")
    # v2 corrects the state-conditioning implementation while retaining the
    # same seed/threshold grid and mismatched-state records.
    state = load_records(RUNS / "tfim_state_conditioning_ablation_20260916_v2")
    print("\n[mechanism audit: event totals and conditions with >=1 event]")
    for label, rows, key in [("prefix", order, "method"), ("state", state, "state_condition")]:
        buckets = defaultdict(list)
        for row in rows:
            buckets[row[key]].append(row["physical_threshold_violations"])
        for name, events in sorted(buckets.items()):
            print(f"{label:6s} {name:18s} events={sum(events):2d}; conditions={sum(v>0 for v in events):2d}/{len(events)}")


def multiscale():
    dirs = [RUNS / "tfim_scale_matched_baselines_20260923", RUNS / "tfim_n10_matched_baselines_20260923"]
    rows = []
    for directory in dirs:
        rows.extend(load_records(directory))
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["n_qubits"], row["delta"])
        grouped[key][row["method"]].append(row)
    names = [
        "qiprune_iterative_hybrid_path",
        "single_gate_energy_probe_matched_l",
        "diagonal_qfi_quadratic_saliency_matched_l",
        "parameter_magnitude_matched_l",
        "gradient_integrated_energy_matched_k",
        "random_matched_l",
    ]
    print("\n[historical dual-rule q=1.030454533953517 multiscale matched-count final gaps across 3 independently optimized checkpoints]")
    for key in sorted(grouped):
        first = grouped[key]["qiprune_iterative_hybrid_path"]
        counts = [row["matched_delete_count"] for row in first]
        fractions = [row["actual_gate_prune_ratio"] for row in first]
        cells = "; ".join(
            f"{name.replace('_matched_l','').replace('_matched_k','').replace('qiprune_iterative_hybrid_path','iPrune')}="
            f"{pm([row['energy_gap_finetuned_vs_control'] for row in grouped[key][name]])}"
            for name in names
        )
        print(f"N={key[0]}, delta={key[1]}; L={pm(counts)}; pruned={pm(fractions)}: {cells}")


if __name__ == "__main__":
    phase_tightness()
    violation_events()
    multiscale()
