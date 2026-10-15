from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scripts.experiments.run_q_matched_k_mnist49_seed import DEFAULT_SOURCE_CHECKPOINT


PROJECT_DIR = Path(__file__).resolve().parents[2]
SEED_RUNNER = PROJECT_DIR / "run_q_matched_k_mnist49_seed.py"
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "q_sensitivity_upgrade_20260810"
    / "mnist49_matched_k"
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row if not isinstance(row[key], (list, dict))})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_rows(rows_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(rows_dir.glob("mnist49_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("results", []))
    return rows


def _checkpoint(output_dir: Path, meta: Mapping[str, Any], started: float) -> None:
    rows = _load_rows(output_dir / "rows")
    payload = {
        "meta": dict(meta),
        "completed_seeds": len({int(row["seed"]) for row in rows}),
        "completed_rows": len(rows),
        "elapsed_sec": time.time() - started,
        "results": rows,
    }
    _atomic_json(output_dir / "checkpoint.json", payload)
    _atomic_csv(output_dir / "checkpoint.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable MNIST49 matched-K q pilot sweep.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source_checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--snapshot_dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.3, 0.5])
    parser.add_argument("--qbetas", nargs="+", type=float, default=[0, 1, 3, 6, 10, 20])
    parser.add_argument("--selection_only", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment_family": "q_sensitivity_upgrade_20260810",
        "protocol": "mnist49_q_geometry_matched_k_pilot",
        "runner": str(Path(__file__).resolve()),
        "seed_runner": str(SEED_RUNNER),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "snapshot_dir": str(args.snapshot_dir.resolve()) if args.snapshot_dir else None,
        "seeds": args.seeds,
        "deltas": args.deltas,
        "qbetas": args.qbetas,
        "selection_only": bool(args.selection_only),
        "expected_rows": len(args.seeds) * len(args.deltas) * len(args.qbetas),
    }
    if not (output_dir / "manifest.json").exists():
        _atomic_json(output_dir / "manifest.json", meta)
    started = time.time()

    for index, seed in enumerate(args.seeds, start=1):
        output_path = rows_dir / f"mnist49_seed{seed}.json"
        if output_path.exists():
            print(f"[SKIP {index}/{len(args.seeds)}] seed={seed}", flush=True)
            continue
        command = [
            sys.executable,
            "-u", "-m", "scripts.experiments.run_q_matched_k_mnist49_seed",
            "--output_json",
            str(output_path),
            "--source_checkpoint",
            str(args.source_checkpoint.resolve()),
            "--seed",
            str(seed),
            "--deltas",
            *[str(value) for value in args.deltas],
            "--qbetas",
            *[str(value) for value in args.qbetas],
        ]
        if args.selection_only:
            command.append("--selection_only")
        if args.snapshot_dir:
            snapshot = args.snapshot_dir.resolve() / f"mnist49_seed{seed}.json"
            if not snapshot.exists():
                raise FileNotFoundError(snapshot)
            command.extend(["--snapshot_json", str(snapshot)])
        print(f"[RUN  {index}/{len(args.seeds)}] seed={seed}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
        _checkpoint(output_dir, meta, started)
        if completed.returncode != 0:
            print(f"[FAIL {index}/{len(args.seeds)}] seed={seed}", flush=True)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
        else:
            print(f"[DONE {index}/{len(args.seeds)}] seed={seed}", flush=True)

    _checkpoint(output_dir, meta, started)
    rows = _load_rows(rows_dir)
    print(
        f"Completed {len({int(row['seed']) for row in rows})}/{len(args.seeds)} seeds "
        f"and {len(rows)} rows. Checkpoint: {output_dir / 'checkpoint.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
