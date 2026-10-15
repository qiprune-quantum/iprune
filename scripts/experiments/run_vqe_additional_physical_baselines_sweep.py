from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[2]
SEED_RUNNER = PROJECT_DIR / "run_vqe_additional_physical_baselines_seed.py"
SOURCE_METHODS = {"qiprune_true_identity", "gradient_integrated_energy_matched_k"}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_additional(rows_dir: Path) -> tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    completed_seeds = 0
    for path in sorted(rows_dir.glob("seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed_rows = payload.get("results", [])
        if seed_rows:
            completed_seeds += 1
            rows.extend(seed_rows)
    return rows, completed_seeds


def _load_source(path: Path, completed_seed_ids: set[int]) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        row
        for row in payload.get("results", [])
        if int(row["seed"]) in completed_seed_ids and row["method"] in SOURCE_METHODS
    ]


def _checkpoint(
    output_dir: Path,
    meta: Dict[str, Any],
    source_checkpoint: Path,
    started: float,
) -> None:
    additional_rows, completed_seeds = _load_additional(output_dir / "rows")
    completed_seed_ids = {int(row["seed"]) for row in additional_rows}
    source_rows = _load_source(source_checkpoint, completed_seed_ids)
    combined = source_rows + additional_rows
    payload = {
        "meta": meta,
        "results": combined,
        "source_method_rows": len(source_rows),
        "additional_method_rows": len(additional_rows),
        "completed_seeds": completed_seeds,
        "completed_rows": len(combined),
        "elapsed_sec": time.time() - started,
    }
    _atomic_json(output_dir / "physical_gate_baselines_checkpoint.json", payload)
    _atomic_csv(output_dir / "physical_gate_baselines_checkpoint.csv", combined)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable matched-K random/parameter VQE physical-deletion sweep."
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 0.9, 1.0, 1.2, 1.5, 2.0])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqe_depth", type=int, default=6)
    parser.add_argument("--vqe_steps", type=int, default=40)
    parser.add_argument("--vqe_lr", type=float, default=0.1)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--finetune_lr", type=float, default=0.05)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--random_seed_offset", type=int, default=20260807)
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment_family": "supplementary_upgrade",
        "protocol": "same_tfim_circuit_matched_K_four_physical_gate_deletion_methods",
        "runner": str(Path(__file__).resolve()),
        "seed_runner": str(SEED_RUNNER),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "seeds": args.seeds,
        "deltas": args.deltas,
        "methods": [
            "qiprune_true_identity",
            "gradient_integrated_energy_matched_k",
            "random_rotation_matched_k",
            "parameter_circular_magnitude_matched_k",
        ],
        "expected_seeds": len(args.seeds),
        "expected_additional_rows": len(args.seeds) * len(args.deltas) * 2,
        "expected_combined_rows": len(args.seeds) * len(args.deltas) * 4,
        "vqe_steps": args.vqe_steps,
        "finetune_steps": args.finetune_steps,
    }
    _atomic_json(output_dir / "manifest.json", meta)
    started = time.time()

    for index, seed in enumerate(args.seeds, start=1):
        row_path = rows_dir / f"seed{seed}.json"
        if row_path.exists():
            print(f"[SKIP {index}/{len(args.seeds)}] seed={seed}", flush=True)
            continue
        command = [
            sys.executable,
            "-u",
            str(SEED_RUNNER),
            "--output_json",
            str(row_path),
            "--source_checkpoint",
            str(args.source_checkpoint),
            "--seed",
            str(seed),
            "--deltas",
            *[str(value) for value in args.deltas],
            "--device",
            args.device,
            "--n_qubits",
            str(args.n_qubits),
            "--vqe_depth",
            str(args.vqe_depth),
            "--vqe_steps",
            str(args.vqe_steps),
            "--vqe_lr",
            str(args.vqe_lr),
            "--finetune_steps",
            str(args.finetune_steps),
            "--finetune_lr",
            str(args.finetune_lr),
            "--J",
            str(args.J),
            "--h",
            str(args.h),
            "--random_seed_offset",
            str(args.random_seed_offset),
        ]
        print(f"[RUN  {index}/{len(args.seeds)}] seed={seed}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
        if completed.returncode != 0:
            print(
                f"[FAIL {index}/{len(args.seeds)}] seed={seed} exit={completed.returncode}",
                flush=True,
            )
            _checkpoint(output_dir, meta, args.source_checkpoint, started)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
            continue
        _checkpoint(output_dir, meta, args.source_checkpoint, started)
        print(f"[DONE {index}/{len(args.seeds)}] seed={seed}", flush=True)

    _checkpoint(output_dir, meta, args.source_checkpoint, started)
    rows, completed_seeds = _load_additional(rows_dir)
    print(
        f"Completed {completed_seeds}/{len(args.seeds)} seeds and "
        f"{len(rows)}/{meta['expected_additional_rows']} additional rows. "
        f"Checkpoint: {output_dir / 'physical_gate_baselines_checkpoint.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
