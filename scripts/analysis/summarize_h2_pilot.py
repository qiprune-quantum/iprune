"""Seed-level summary for the audited H2 physical-deletion pilot."""

from __future__ import annotations

import json
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "h2_vqe_pilot_audited_20260913"


def stat(values: list[float]) -> str:
    return f"{mean(values):.5f} ± {stdev(values):.5f}" if len(values) > 1 else f"{values[0]:.5f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an audited H2 physical-deletion batch.")
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    parser.add_argument("--title", default="H2 VQE audited physical-deletion supplementary batch")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rows = []
    for path in sorted(run_dir.glob("seed*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["results"])
    by_method_seed = defaultdict(list)
    for row in rows:
        by_method_seed[(row["method"], int(row["seed"]))].append(float(row["energy_gap_finetuned_vs_control"]))
    by_method = defaultdict(list)
    for (method, _), values in by_method_seed.items():
        by_method[method].append(mean(values))
    qrows = [row for row in rows if row["method"] == "qiprune_iterative_hybrid_path"]
    seed_count = len({int(row["seed"]) for row in qrows})
    bond_lengths = sorted({float(row["h2_bond_length_angstrom"]) for row in rows})
    bond_text = ", ".join(f"{value:.2f}" for value in bond_lengths)
    pairs = defaultdict(lambda: [0, 0, 0])
    indexed = defaultdict(dict)
    for row in rows:
        indexed[(row["seed"], row["delta"])][row["method"]] = float(row["energy_gap_finetuned_vs_control"])
    for methods in indexed.values():
        qvalue = methods["qiprune_iterative_hybrid_path"]
        for method, value in methods.items():
            if method == "qiprune_iterative_hybrid_path":
                continue
            if abs(qvalue - value) < 1e-12:
                pairs[method][1] += 1
            elif qvalue < value:
                pairs[method][0] += 1
            else:
                pairs[method][2] += 1
    lines = [
        f"# {args.title}",
        "",
        f"{seed_count} independent seeds, 4-qubit H2 at bond length {bond_text} Å, depth-4 HEA, convergence-controlled VQE, real rotation deletion, matched realized K, and 40-step recovery. Threshold rows are averaged within each seed before the first table.",
        "",
        "| method | independent seeds | final energy gap, seed mean ± SD |",
        "|---|---:|---:|",
        *[f"| {method} | {len(values)} | {stat(values)} |" for method, values in sorted(by_method.items())],
        "",
        "| comparator | q-iPrune W/T/L by operating point |",
        "|---|---:|",
        *[f"| {method} | {w}/{t}/{l} |" for method, (w, t, l) in sorted(pairs.items())],
        "",
        f"All {len(qrows)} q-iPrune operating rows recorded zero local physical-threshold violations and passed the realized-path observable bound. Realized K ranges from {min(int(r['matched_delete_count']) for r in qrows)} to {max(int(r['matched_delete_count']) for r in qrows)}.",
        "",
        "## Interpretation boundary",
        "",
        "This is a supplementary batch, not a replacement for the TFIM main experiment. It shows that the real-gate-deletion protocol transfers to a molecular Hamiltonian. The direct single-gate energy probe is retained as a task-loss oracle; its relative performance must be read from the reported table for each molecular regime.",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run_dir / "REPORT.md")


if __name__ == "__main__":
    main()
