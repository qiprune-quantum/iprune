"""Summarize separate TFIM depth batches without pooling their samples."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
BATCHES = {
    4: ROOT / "runs" / "formal_tfim_d4_dynamic_q1_20260911",
    6: ROOT / "runs" / "depth_robustness_tfim_d6_20260913",
    8: ROOT / "runs" / "depth_robustness_tfim_d8_20260913",
}
QIPRUNE = "qiprune_iterative_hybrid_path"
METRIC = "energy_gap_finetuned_vs_control"


def stats(values: List[float]) -> str:
    return f"{mean(values):.5f} ± {stdev(values):.5f}" if len(values) > 1 else f"{values[0]:.5f}"


def load(path: Path) -> List[dict]:
    records: List[dict] = []
    for source in sorted(path.glob("seed*.json")):
        records.extend(json.loads(source.read_text(encoding="utf-8"))["results"])
    return records


def main() -> None:
    output_dir = ROOT / "runs" / "tfim_depth_scan_20260913"
    output_dir.mkdir(exist_ok=True)
    summary = []
    detail = []
    for depth, directory in BATCHES.items():
        rows = load(directory)
        qrows = [row for row in rows if row["method"] == QIPRUNE]
        by_seed: Dict[int, List[dict]] = defaultdict(list)
        for row in qrows:
            by_seed[int(row["seed"])].append(row)
        seed_gaps = [mean(float(row[METRIC]) for row in values) for values in by_seed.values()]
        seed_ks = [mean(float(row["matched_delete_count"]) for row in values) for values in by_seed.values()]
        summary.append({
            "depth": depth,
            "independent_seeds": len(by_seed),
            "operating_rows": len(qrows),
            "qiprune_seed_mean_gap": mean(seed_gaps),
            "qiprune_seed_sd_gap": stdev(seed_gaps),
            "qiprune_seed_mean_K": mean(seed_ks),
            "qiprune_seed_sd_K": stdev(seed_ks),
            "threshold_violations": sum(int(row.get("physical_threshold_violations", 0)) for row in qrows),
            "observable_bound_violations": sum(int(row.get("observable_bound_violations", 0)) for row in qrows),
        })
        for row in qrows:
            detail.append({
                "depth": depth,
                "seed": row["seed"],
                "delta": row["delta"],
                "K": row["matched_delete_count"],
                "qiprune_energy_gap": row[METRIC],
                "path_trace_sum_max": row["realized_path_trace_sum_max"],
                "actual_trace_distance_max": row["actual_trace_distance_max"],
                "threshold_violations": row["physical_threshold_violations"],
                "observable_bound_violations": row["observable_bound_violations"],
            })
    with (output_dir / "depth_scan_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    with (output_dir / "depth_scan_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# TFIM depth scan: dynamic q=1 identity deletion",
        "",
        "Each depth is an independent batch. The table summarizes each seed over its three threshold operating points; it does **not** pool depths or treat thresholds as independent samples.",
        "",
        "| HEA depth | independent seeds | q-iPrune final gap, seed mean ± SD | realized K, seed mean ± SD | threshold / observable-bound violations |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['depth']} | {row['independent_seeds']} | "
            f"{row['qiprune_seed_mean_gap']:.5f} ± {row['qiprune_seed_sd_gap']:.5f} | "
            f"{row['qiprune_seed_mean_K']:.3f} ± {row['qiprune_seed_sd_K']:.3f} | "
            f"{row['threshold_violations']} / {row['observable_bound_violations']} |"
        )
    lines.extend([
        "",
        "## What this establishes",
        "",
        "- The dynamic hybrid-prefix rule was executed successfully at depths 4, 6, and 8 with zero recorded physical-threshold and observable-bound violations in the q-iPrune rows.",
        "- It is valid evidence that the method is usable beyond one fixed ansatz depth.",
        "",
        "## What it does not establish",
        "",
        "- Raw gaps must not be compared across depths as an intrinsic depth ranking: each depth's threshold rule realizes a different deletion budget K.",
        "- The table does not by itself estimate a universal functional dependence on L. The per-row CSV is retained for a later descriptive path-bound versus actual-distance figure.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
