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
SEED_RUNNER = PROJECT_DIR / "run_vqe_qiprune_vs_gradient_seed.py"


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


def _load(rows_dir: Path) -> tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    completed_seeds = 0
    for path in sorted(rows_dir.glob("seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed_rows = payload.get("results", [])
        if seed_rows:
            completed_seeds += 1
            rows.extend(seed_rows)
    return rows, completed_seeds


def _checkpoint(output_dir: Path, meta: Dict[str, Any], started: float) -> None:
    rows, completed_seeds = _load(output_dir / "rows")
    payload = {
        "meta": meta,
        "results": rows,
        "completed_seeds": completed_seeds,
        "completed_rows": len(rows),
        "elapsed_sec": time.time() - started,
    }
    _atomic_json(output_dir / "qiprune_vs_gradient_checkpoint.json", payload)
    _atomic_csv(output_dir / "qiprune_vs_gradient_checkpoint.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable matched-K q-iPrune/gradient VQE sweep.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.8, 0.9, 1.0, 1.2, 1.5, 2.0])
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqe_depth", type=int, default=6)
    parser.add_argument("--vqe_steps", type=int, default=40)
    parser.add_argument("--vqe_lr", type=float, default=0.1)
    parser.add_argument("--finetune_steps", type=int, default=40)
    parser.add_argument("--finetune_lr", type=float, default=0.05)
    parser.add_argument("--gradient_points", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--qbeta", type=float, default=0.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment_family": "supplementary_upgrade",
        "protocol": "same_tfim_circuit_matched_K_qiprune_vs_real_integrated_gradient",
        "runner": str(Path(__file__).resolve()),
        "seed_runner": str(SEED_RUNNER),
        "seeds": args.seeds,
        "deltas": args.deltas,
        "expected_seeds": len(args.seeds),
        "expected_rows": len(args.seeds) * len(args.deltas) * 2,
        "vqe_steps": args.vqe_steps,
        "finetune_steps": args.finetune_steps,
        "gradient_points": args.gradient_points,
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
            "--gradient_points",
            str(args.gradient_points),
            "--gamma",
            str(args.gamma),
            "--alpha",
            str(args.alpha),
            "--qbeta",
            str(args.qbeta),
            "--J",
            str(args.J),
            "--h",
            str(args.h),
        ]
        print(f"[RUN  {index}/{len(args.seeds)}] seed={seed}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
        if completed.returncode != 0:
            print(f"[FAIL {index}/{len(args.seeds)}] seed={seed} exit={completed.returncode}", flush=True)
            _checkpoint(output_dir, meta, started)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
            continue
        _checkpoint(output_dir, meta, started)
        print(f"[DONE {index}/{len(args.seeds)}] seed={seed}", flush=True)

    _checkpoint(output_dir, meta, started)
    rows, completed_seeds = _load(rows_dir)
    print(
        f"Completed {completed_seeds}/{len(args.seeds)} seeds and "
        f"{len(rows)}/{meta['expected_rows']} rows. Checkpoint: "
        f"{output_dir / 'qiprune_vs_gradient_checkpoint.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
