"""Summarize paired selection-cost measurements without pseudoreplication."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.root / "raw").glob("*.json"))]
    groups = defaultdict(list)
    for record in records:
        groups[(record["n_qubits"], record["seed"], record["method"])].append(record)
    summary = []
    for (n, seed, method), observations in sorted(groups.items()):
        assert len({item["repeat"] for item in observations}) == len(observations)
        assert len({tuple(item["selected_positions"]) for item in observations}) == 1
        row = {
            "n_qubits": n,
            "seed": seed,
            "method": method,
            "technical_repeats": len(observations),
            "median_elapsed_ms": float(np.median([item["elapsed_ms"] for item in observations])),
            "median_peak_rss_mib": float(np.median([item["peak_rss_mib"] for item in observations])),
            "median_peak_increment_mib": float(np.median([item["peak_increment_mib"] for item in observations])),
            "deletion_count": observations[0]["deletion_count"],
            "rotation_count": observations[0]["rotation_count"],
            "training_steps": observations[0]["training_steps"],
        }
        summary.append(row)
    for n in sorted({row["n_qubits"] for row in summary}):
        selected = [row for row in summary if row["n_qubits"] == n]
        paired = defaultdict(dict)
        for row in selected:
            paired[row["seed"]][row["method"]] = row
        complete = [item for item in paired.values() if set(item) == {"iprune", "probe"}]
        if not complete:
            continue
        ratios = [item["iprune"]["median_elapsed_ms"] / item["probe"]["median_elapsed_ms"] for item in complete]
        print(f"N={n}, independent seeds={len(complete)}")
        for method in ("iprune", "probe"):
            times = np.array([item[method]["median_elapsed_ms"] for item in complete])
            rss = np.array([item[method]["median_peak_rss_mib"] for item in complete])
            print(f"  {method}: time {np.mean(times):.1f} +/- {np.std(times, ddof=1) if len(times)>1 else float('nan'):.1f} ms; peak RSS {np.mean(rss):.1f} +/- {np.std(rss, ddof=1) if len(rss)>1 else float('nan'):.1f} MiB")
        print(f"  paired time ratio: {np.mean(ratios):.2f}x; L: {[item['iprune']['deletion_count'] for item in complete]}")
    output = args.root / "seed_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
