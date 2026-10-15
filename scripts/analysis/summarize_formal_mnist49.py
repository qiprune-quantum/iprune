#!/usr/bin/env python3
"""Aggregate the formal MNIST49 Route-B batch (seeds 2011-2022)."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path


RUN = Path(__file__).resolve().parents[2] / "runs" / "formal_mnist49_dynamic_q1_20260911"
MAIN = "qiprune_iterative_hybrid_path"


def main() -> None:
    per = {}  # (delta, method) -> list of (seed, drop, L, viol)
    for p in sorted(RUN.glob("seed*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        seed = payload["meta"].get("seed")
        for row in payload["results"]:
            key = (float(row["delta"]), row["method"])
            viol = (
                row.get("heldout_physical_gate_violations", 0)
                + row.get("heldout_physical_state_position_violations", 0)
                + row.get("heldout_path_bound_violations", 0)
                + row.get("calibration_physical_threshold_violations", 0)
            )
            per.setdefault(key, []).append(
                (seed, row["accuracy_drop_finetuned_vs_control"], row["matched_delete_count"], viol)
            )

    deltas = sorted({d for (d, _) in per})
    print("=== mean +/- sample SD of fine-tuned accuracy drop vs control ===")
    for d in deltas:
        print(f"\n-- delta={d} --")
        methods = sorted({m for (dd, m) in per if dd == d})
        for m in methods:
            vals = [v for (_, v, _, _) in per[(d, m)]]
            Ls = [l for (_, _, l, _) in per[(d, m)]]
            viol = sum(x for (_, _, _, x) in per[(d, m)])
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{m:42s} n={len(vals):2d} mean={st.mean(vals):+.4f} sd={sd:.4f} "
                  f"L_mean={st.mean(Ls):.1f} viol={viol}")

    print("\n=== q-iPrune win/tie/loss vs each baseline (paired by delta) ===")
    for d in deltas:
        base = {s: v for (s, v, _, _) in per[(d, MAIN)]}
        print(f"\n-- delta={d} --")
        for m in sorted({mm for (dd, mm) in per if dd == d and mm != MAIN}):
            other = {s: v for (s, v, _, _) in per[(d, m)]}
            w = t = l = 0
            for s in base:
                if s not in other:
                    continue
                diff = other[s] - base[s]  # positive => baseline drop larger => q-iPrune better
                if diff > 1e-9:
                    w += 1
                elif diff < -1e-9:
                    l += 1
                else:
                    t += 1
            print(f"{m:42s} win={w} tie={t} loss={l}")


if __name__ == "__main__":
    main()
