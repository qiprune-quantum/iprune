"""Audit historical dual-threshold scans against physical-only q=1 reruns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[int, dict[float, dict]]:
    grouped: dict[int, dict[float, dict]] = {}
    for file in sorted(path.glob("seed*.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        seed = int(payload["meta"]["seed"])
        if len(payload["results"]) != len(payload["meta"]["deltas"]):
            raise ValueError(f"Incomplete threshold scan: {file}")
        if "physical_q1" in path.name and (
            payload["meta"].get("selection_geometry") != "physical_fubini_study"
            or "qbeta" not in payload["meta"]
            or abs(float(payload["meta"]["qbeta"])) > 1e-12
        ):
            raise ValueError(f"Physical-only metadata mismatch: {file}")
        grouped[seed] = {float(row["delta"]): row for row in payload["results"]}
    if not grouped:
        raise ValueError(f"No seed records found in {path}")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    legacy = load(args.legacy_dir)
    physical = load(args.physical_dir)
    if legacy.keys() != physical.keys():
        raise ValueError("Seed sets differ")
    comparisons = []
    for seed in sorted(legacy):
        if legacy[seed].keys() != physical[seed].keys():
            raise ValueError(f"Threshold grid differs for seed {seed}")
        for delta in sorted(legacy[seed]):
            before, after = legacy[seed][delta], physical[seed][delta]
            if abs(float(before["q"]) - 1.0) < 1e-12:
                raise ValueError("Legacy scan is already q=1")
            if abs(float(after["q"]) - 1.0) > 1e-12:
                raise ValueError("Physical rerun has q != 1")
            for key in ("energy_original_optimized", "energy_full_control_after_extra_steps"):
                if abs(float(before[key]) - float(after[key])) > 1e-9:
                    raise ValueError(f"Trained/control checkpoint differs: seed={seed} delta={delta} {key}")
            changed = before["deleted_pos_ids"] != after["deleted_pos_ids"]
            comparisons.append({
                "seed": seed,
                "delta": delta,
                "changed_path": changed,
                "old_positions": before["deleted_pos_ids"],
                "physical_positions": after["deleted_pos_ids"],
                "old_final_gap": float(before["energy_gap_finetuned_vs_control"]),
                "physical_final_gap": float(after["energy_gap_finetuned_vs_control"]),
            })
    changed_rows = [row for row in comparisons if row["changed_path"]]
    result = {
        "legacy_dir": str(args.legacy_dir),
        "physical_dir": str(args.physical_dir),
        "seeds": len(legacy),
        "seed_threshold_rows": len(comparisons),
        "changed_paths": len(changed_rows),
        "changed_seed_count": len({row["seed"] for row in changed_rows}),
        "changed_rows": changed_rows,
        "legacy_mean_final_gap": sum(row["old_final_gap"] for row in comparisons) / len(comparisons),
        "physical_mean_final_gap": sum(row["physical_final_gap"] for row in comparisons) / len(comparisons),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Checked {len(comparisons)} rows; changed paths: {len(changed_rows)} across {result['changed_seed_count']} seeds")
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
