#!/usr/bin/env python3
"""Aggregate the formal TFIM Route-B batch (seeds 3001-3012)."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path


RUN = Path(__file__).resolve().parents[2] / "runs" / "formal_tfim_d4_dynamic_q1_20260911"
MAIN = "qiprune_iterative_hybrid_path"
METRIC = "energy_gap_finetuned_vs_control"


def main() -> None:
    per = {}          # (delta, method) -> list of (seed, metric)
    viol_total = 0
    n_seed = 0
    for p in sorted(RUN.glob("seed*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        n_seed += 1
        for row in payload["results"]:
            key = (float(row["delta"]), row["method"])
            per.setdefault(key, []).append((row.get("seed"), row[METRIC]))
            viol_total += int(row.get("physical_threshold_violations", 0))
            viol_total += int(row.get("observable_bound_violations", 0))

    print(f"seeds={n_seed}  total_violations={viol_total}")
    print("\n=== mean +/- sample SD of fine-tuned energy gap vs control (lower better) ===")
    for d in sorted({d for (d, _) in per}):
        print(f"\n-- delta={d} --")
        for m in sorted({mm for (dd, mm) in per if dd == d}):
            vals = [v for (_, v) in per[(d, m)]]
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{m:44s} n={len(vals):2d} mean={st.mean(vals):+.4f} sd={sd:.4f}")

    print("\n=== q-iPrune win/tie/loss vs each baseline (paired by seed, same delta) ===")
    for d in sorted({d for (d, _) in per}):
        base = dict(per[(d, MAIN)])
        print(f"\n-- delta={d} --")
        for m in sorted({mm for (dd, mm) in per if dd == d and mm != MAIN}):
            other = dict(per[(d, m)])
            w = t = l = 0
            for s in base:
                if s not in other:
                    continue
                diff = other[s] - base[s]   # positive => baseline gap larger => q-iPrune better
                if diff > 1e-9:
                    w += 1
                elif diff < -1e-9:
                    l += 1
                else:
                    t += 1
            print(f"{m:44s} win={w} tie={t} loss={l}")


if __name__ == "__main__":
    main()
