"""Summarize frozen TFIM and MNIST49 dynamic q=1 batches without pseudo-replication."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TFIM = ROOT / "runs" / "formal_tfim_d4_dynamic_q1_20260911"
DEFAULT_MNIST = ROOT / "runs" / "formal_mnist49_dynamic_q1_20260911"


def load_rows(directory: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("seed*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")).get("results", []))
    if not rows:
        raise FileNotFoundError(f"No seed JSON files found in {directory}")
    return rows


def format_stats(values: Iterable[float]) -> str:
    values = list(values)
    if len(values) == 1:
        return f"{values[0]:.5f}"
    return f"{mean(values):.5f} ± {stdev(values):.5f}"


def seed_level_summary(rows: List[Dict[str, Any]], value_key: str) -> List[str]:
    by_method_seed: Dict[tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        by_method_seed[(str(row["method"]), int(row["seed"]))].append(float(row[value_key]))
    by_method: Dict[str, List[float]] = defaultdict(list)
    for (method, _), values in by_method_seed.items():
        by_method[method].append(mean(values))
    return [
        f"| {method} | {len(values)} | {format_stats(values)} |"
        for method, values in sorted(by_method.items())
    ]


def method_pair_wlt(rows: List[Dict[str, Any]], reference: str, metric: str) -> List[str]:
    indexed: Dict[tuple[int, float], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        indexed[(int(row["seed"]), float(row["delta"]))][str(row["method"])] = float(row[metric])
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    for methods in indexed.values():
        if reference not in methods:
            continue
        for method, value in methods.items():
            if method == reference:
                continue
            delta = methods[reference] - value
            if abs(delta) < 1e-12:
                counts[method][1] += 1
            elif delta < 0.0:
                counts[method][0] += 1
            else:
                counts[method][2] += 1
    return [f"| {method} | {win}/{tie}/{loss} |" for method, (win, tie, loss) in sorted(counts.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfim_dir", type=Path, default=DEFAULT_TFIM)
    parser.add_argument("--mnist_dir", type=Path, default=DEFAULT_MNIST)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "formal_dynamic_q1_summary_20260913.md")
    args = parser.parse_args()
    tfim_rows = load_rows(args.tfim_dir)
    mnist_rows = load_rows(args.mnist_dir)

    tfim_q = [r for r in tfim_rows if r["method"] == "qiprune_iterative_hybrid_path"]
    mnist_q = [r for r in mnist_rows if r["method"] == "qiprune_iterative_hybrid_path"]
    lines = [
        "# Formal dynamic q=1 batch summary",
        "",
        "This summary averages threshold rows within each seed before reporting mean ± sample SD. Thresholds are operating points, not independent replicates.",
        "",
        "## TFIM: final energy gap to equally fine-tuned full circuit",
        "",
        "| method | independent seeds | seed-level mean ± SD |",
        "|---|---:|---:|",
        *seed_level_summary(tfim_rows, "energy_gap_finetuned_vs_control"),
        "",
        "### TFIM paired record for q-iPrune (win/tie/loss; lower gap wins)",
        "",
        "| comparator | q-iPrune W/T/L |",
        "|---|---:|",
        *method_pair_wlt(tfim_rows, "qiprune_iterative_hybrid_path", "energy_gap_finetuned_vs_control"),
        "",
        "## MNIST49: final accuracy drop to equally fine-tuned full circuit",
        "",
        "| method | independent seeds | seed-level mean ± SD |",
        "|---|---:|---:|",
        *seed_level_summary(mnist_rows, "accuracy_drop_finetuned_vs_control"),
        "",
        "### MNIST49 paired record for q-iPrune (win/tie/loss; lower drop wins)",
        "",
        "| comparator | q-iPrune W/T/L |",
        "|---|---:|",
        *method_pair_wlt(mnist_rows, "qiprune_iterative_hybrid_path", "accuracy_drop_finetuned_vs_control"),
        "",
        "## Deletion-count and certificate audit",
        "",
        f"- TFIM q-iPrune: {len(tfim_q)} rows, K range {min(r['matched_delete_count'] for r in tfim_q)}–{max(r['matched_delete_count'] for r in tfim_q)}; physical-threshold violations = {sum(int(r.get('physical_threshold_violations', 0)) for r in tfim_q)}; observable-bound violations = {sum(int(r.get('observable_bound_violations', 0)) for r in tfim_q)}.",
        f"- MNIST49 q-iPrune: {len(mnist_q)} rows, K range {min(r['matched_delete_count'] for r in mnist_q)}–{max(r['matched_delete_count'] for r in mnist_q)}; held-out path-bound violations = {sum(int(r.get('heldout_path_bound_violations', 0)) for r in mnist_q)}; held-out physical-gate violations = {sum(int(r.get('heldout_physical_gate_violations', 0)) for r in mnist_q)}.",
        "",
        "## Interpretation boundary",
        "",
        "The task metrics are not numerically comparable: TFIM uses an energy gap, whereas MNIST49 uses classification-accuracy loss. The unequal realized K ranges must not be used to claim that a Hamiltonian alone explains any task difference. A forced-K cross-task diagnostic is needed for that question.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
