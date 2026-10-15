from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.run_vqe_qiprune_vs_gradient_seed import _atomic_new_json


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_same_baseline(base: dict[str, Any], extra: dict[str, Any], path: Path) -> None:
    base_meta, extra_meta = base["meta"], extra["meta"]
    exact_meta = ["seed", "vqe_steps_run", "vqe_converged"]
    for key in exact_meta:
        if base_meta[key] != extra_meta[key]:
            raise ValueError(f"Baseline meta mismatch for {key} in {path}.")

    base_row = base["results"][0]
    extra_row = extra["results"][0]
    exact_rows = [
        "baseline_circuit_signature",
        "gate_count_before",
        "rotation_count_before",
        "n_qubits",
        "vqe_depth",
    ]
    for key in exact_rows:
        if base_row[key] != extra_row[key]:
            raise ValueError(f"Baseline row mismatch for {key} in {path}.")
    numeric_rows = [
        "energy_initial",
        "energy_original_optimized",
        "energy_full_control_after_extra_steps",
        "vqe_final_window_range",
    ]
    for key in numeric_rows:
        if not np.isclose(float(base_row[key]), float(extra_row[key]), atol=1e-12, rtol=0.0):
            raise ValueError(f"Baseline numeric mismatch for {key} in {path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the 15-point and six midpoint TFIM scans.")
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("runs/tfim_pruning_transition_formal_20260915"),
    )
    parser.add_argument(
        "--extra_dir",
        type=Path,
        default=Path("runs/tfim_pruning_transition_midpoints_20260915"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/tfim_pruning_transition_formal_dense_20260915"),
    )
    parser.add_argument("--expected_seeds", type=int, default=12)
    args = parser.parse_args()

    base_files = sorted(args.base_dir.glob("seed*.json"))
    extra_files = sorted(args.extra_dir.glob("seed*.json"))
    if len(base_files) != args.expected_seeds or len(extra_files) != args.expected_seeds:
        raise ValueError("Both source directories must contain the expected seed count.")
    if [path.name for path in base_files] != [path.name for path in extra_files]:
        raise ValueError("Base and midpoint seed files do not match one-to-one.")

    for base_path, extra_path in zip(base_files, extra_files):
        base, extra = load(base_path), load(extra_path)
        assert_same_baseline(base, extra, extra_path)
        base_deltas = {float(row["delta"]) for row in base["results"]}
        extra_deltas = {float(row["delta"]) for row in extra["results"]}
        overlap = base_deltas & extra_deltas
        if overlap:
            raise ValueError(f"Overlapping delta values for {base_path.name}: {sorted(overlap)}")
        rows = sorted(base["results"] + extra["results"], key=lambda row: float(row["delta"]))
        deltas = [float(row["delta"]) for row in rows]
        if len(rows) != 21 or np.any(np.diff(deltas) <= 0.0):
            raise ValueError(f"Merged grid is incomplete for {base_path.name}.")
        meta = dict(base["meta"])
        meta.update(
            {
                "status": "formal",
                "protocol": "converged_tfim_dynamic_prefix_real_deletion_dense_21_threshold_scan",
                "deltas": deltas,
                "expected_rows": len(rows),
                "base_scan": str(args.base_dir),
                "midpoint_scan": str(args.extra_dir),
                "baseline_identity_checks": "passed",
            }
        )
        _atomic_new_json(args.output_dir / base_path.name, {"meta": meta, "results": rows})
        print(f"[MERGED] seed={meta['seed']} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
