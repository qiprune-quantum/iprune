"""Create a seed-level report for the frozen depth-6 TFIM robustness batch."""

from __future__ import annotations

import json
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "depth_robustness_tfim_d6_20260913"


def stat(values: List[float]) -> str:
    return f"{mean(values):.5f} ± {stdev(values):.5f}" if len(values) > 1 else f"{values[0]:.5f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    parser.add_argument("--label", default="depth-6")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rows = []
    for path in sorted(run_dir.glob("seed*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["results"])
    qmethod = "qiprune_iterative_hybrid_path"
    by_method_seed: Dict[tuple[str, int], List[float]] = defaultdict(list)
    by_delta: Dict[float, List[dict]] = defaultdict(list)
    for row in rows:
        by_method_seed[(row["method"], int(row["seed"]))].append(float(row["energy_gap_finetuned_vs_control"]))
        if row["method"] == qmethod:
            by_delta[float(row["delta"])].append(row)
    by_method: Dict[str, List[float]] = defaultdict(list)
    for (method, _), values in by_method_seed.items():
        by_method[method].append(mean(values))
    paired: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    indexed: Dict[tuple[int, float], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        indexed[(int(row["seed"]), float(row["delta"]))][row["method"]] = float(row["energy_gap_finetuned_vs_control"])
    for values in indexed.values():
        qvalue = values[qmethod]
        for method, value in values.items():
            if method == qmethod:
                continue
            if abs(qvalue - value) < 1e-12:
                paired[method][1] += 1
            elif qvalue < value:
                paired[method][0] += 1
            else:
                paired[method][2] += 1
    report = [
        f"# TFIM {args.label} robustness batch",
        "",
        f"4 qubits, {args.label} HEA, converged VQE, three deletion thresholds, six matched-realized-K methods, and the same 40-step recovery budget. Threshold rows are averaged within seed before reporting the first table.",
        "",
        "## Final energy gap to equally fine-tuned full circuit",
        "",
        "| method | independent seeds | seed-level mean ± sample SD |",
        "|---|---:|---:|",
        *[f"| {method} | {len(values)} | {stat(values)} |" for method, values in sorted(by_method.items())],
        "",
        "## q-iPrune paired comparison by operating point",
        "",
        "Lower energy gap wins. These counts describe 15 seed-threshold operating points and are not a substitute for independent-seed inference.",
        "",
        "| comparator | q-iPrune W/T/L |",
        "|---|---:|",
        *[f"| {method} | {w}/{t}/{l} |" for method, (w, t, l) in sorted(paired.items())],
        "",
        "## Operating behavior and certificate audit",
        "",
        "| delta | realized K, mean ± SD | final q-iPrune gap, mean ± SD | threshold / observable violations |",
        "|---:|---:|---:|---:|",
    ]
    for delta, values in sorted(by_delta.items()):
        report.append(
            f"| {delta:g} | {stat([float(r['matched_delete_count']) for r in values])} | "
            f"{stat([float(r['energy_gap_finetuned_vs_control']) for r in values])} | "
            f"{sum(int(r['physical_threshold_violations']) for r in values)} / {sum(int(r['observable_bound_violations']) for r in values)} |"
        )
    report.extend([
        "",
        "## Interpretation",
        "",
        "This batch tests whether the dynamic q=1 hybrid-prefix deletion signal persists when the ansatz is deeper. It supports robustness against random, parameter-magnitude, integrated-gradient, and static diagonal-QFI controls if the first table and paired record agree. The direct single-gate energy probe remains a deliberately strong task-loss oracle; equality or near-equality with it is informative, not a failure.",
        "",
        "The result must be reported as a separate depth robustness batch, not pooled with another ansatz depth as if they had the same circuit.",
    ])
    output = run_dir / "REPORT.md"
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
