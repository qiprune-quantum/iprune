"""Summarize matched selection timings, using seed medians as replicates."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(root: Path) -> list[dict]:
    records = [json.loads(p.read_text()) for p in sorted((root / "raw").glob("*.json"))]
    seed_rows = []
    for n in (4, 6, 8):
        for seed in (3001, 3002, 3003):
            counts = set()
            for method in ("iprune", "probe", "dynamic"):
                rows = [r for r in records if (r["n_qubits"], r["seed"], r["method"]) == (n, seed, method)]
                assert len(rows) == 3, (n, seed, method, len(rows))
                assert {r["repeat"] for r in rows} == {1, 2, 3}
                assert all(r["selected_positions"] == rows[0]["selected_positions"] for r in rows)
                counts.update(r["deletion_count"] for r in rows)
                assert len(counts) == 1, (n, seed, counts)
                count = rows[0]["deletion_count"]
                assert len(rows[0]["selected_positions"]) == count
                rotations = rows[0]["rotation_count"]
                seed_rows.append(dict(
                    n_qubits=n, seed=seed, method=method, deletion_count=count,
                    rotation_count=rotations,
                    median_seconds=statistics.median(r["elapsed_ms"] / 1000 for r in rows),
                    median_peak_rss_mib=statistics.median(r["peak_rss_mib"] for r in rows),
                    candidate_evaluations=(rotations * count - count * (count - 1) // 2
                                           if method == "dynamic" else rotations),
                ))
    result = []
    for n in (4, 6, 8):
        for method in ("iprune", "probe", "dynamic"):
            rows = [r for r in seed_rows if r["n_qubits"] == n and r["method"] == method]
            times = [r["median_seconds"] for r in rows]
            result.append(dict(n_qubits=n, method=method, seeds=3, repeats_per_seed=3,
                               time_mean_seconds=statistics.mean(times),
                               time_sd_seconds=statistics.stdev(times),
                               peak_rss_mean_mib=statistics.mean(r["median_peak_rss_mib"] for r in rows),
                               deletion_count_mean=statistics.mean(r["deletion_count"] for r in rows)))
    payload = dict(protocol="q=1, delta=1.2; fresh worker per method and repetition; warm call excluded; matched L setup excluded",
                   seed_medians=seed_rows, summary=result)
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    for row in summarize(parser.parse_args().root):
        print(row)
