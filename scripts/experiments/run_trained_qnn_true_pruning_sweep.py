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
CONFIG_RUNNER = PROJECT_DIR / "run_trained_qnn_true_pruning_seed.py"


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
    completed_configs = 0
    for path in sorted(rows_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config_rows = payload.get("results", [])
        if config_rows:
            completed_configs += 1
            rows.extend(config_rows)
    return rows, completed_configs


def _checkpoint(output_dir: Path, meta: Dict[str, Any], started: float) -> None:
    rows, completed_configs = _load(output_dir / "rows")
    payload = {
        "meta": meta,
        "results": rows,
        "completed_configs": completed_configs,
        "completed_rows": len(rows),
        "elapsed_sec": time.time() - started,
    }
    _atomic_json(output_dir / "trained_qnn_pruning_checkpoint.json", payload)
    _atomic_csv(output_dir / "trained_qnn_pruning_checkpoint.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable trained-QNN true-pruning and post-fine-tuning sweep."
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["mnist49", "fashion_sb", "bas"], default=["mnist49", "fashion_sb", "bas"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
    parser.add_argument("--qnn_depth", type=int, default=3)
    parser.add_argument("--train_steps", type=int, default=40)
    parser.add_argument("--finetune_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_train", type=int, default=512)
    parser.add_argument("--n_calibration", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    configs = [(dataset, seed) for seed in args.seeds for dataset in args.datasets]
    meta = {
        "experiment_family": "supplementary_upgrade",
        "protocol": "trained_qnn_train_calibration_test_true_pruning_post_finetune",
        "runner": str(Path(__file__).resolve()),
        "config_runner": str(CONFIG_RUNNER),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "deltas": args.deltas,
        "expected_configs": len(configs),
        "expected_rows": len(configs) * len(args.deltas),
        "qnn_depth": args.qnn_depth,
        "train_steps": args.train_steps,
        "finetune_steps": args.finetune_steps,
        "split": {
            "train": args.n_train,
            "calibration": args.n_calibration,
            "test": args.n_test,
            "pruning_ensemble_M": args.M,
        },
    }
    _atomic_json(output_dir / "manifest.json", meta)
    started = time.time()

    for index, (dataset, seed) in enumerate(configs, start=1):
        row_path = rows_dir / f"{dataset}_seed{seed}.json"
        if row_path.exists():
            print(f"[SKIP {index}/{len(configs)}] {dataset} seed={seed}", flush=True)
            continue
        command = [
            sys.executable,
            "-u", "-m", "scripts.experiments.run_trained_qnn_true_pruning_seed",
            "--dataset",
            dataset,
            "--output_json",
            str(row_path),
            "--seed",
            str(seed),
            "--deltas",
            *[str(value) for value in args.deltas],
            "--qnn_depth",
            str(args.qnn_depth),
            "--train_steps",
            str(args.train_steps),
            "--finetune_steps",
            str(args.finetune_steps),
            "--lr",
            str(args.lr),
            "--batch_size",
            str(args.batch_size),
            "--n_train",
            str(args.n_train),
            "--n_calibration",
            str(args.n_calibration),
            "--n_test",
            str(args.n_test),
            "--M",
            str(args.M),
            "--data_dir",
            str(args.data_dir),
        ]
        if not args.allow_download:
            command.append("--no_download")
        print(f"[RUN  {index}/{len(configs)}] {dataset} seed={seed}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
        if completed.returncode != 0:
            print(f"[FAIL {index}/{len(configs)}] {dataset} seed={seed} exit={completed.returncode}", flush=True)
            _checkpoint(output_dir, meta, started)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
            continue
        _checkpoint(output_dir, meta, started)
        print(f"[DONE {index}/{len(configs)}] {dataset} seed={seed}", flush=True)

    _checkpoint(output_dir, meta, started)
    rows, completed_configs = _load(rows_dir)
    print(
        f"Completed {completed_configs}/{len(configs)} configs and {len(rows)}/{meta['expected_rows']} rows. "
        f"Checkpoint: {output_dir / 'trained_qnn_pruning_checkpoint.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
