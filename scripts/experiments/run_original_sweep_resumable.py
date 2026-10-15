from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_DIR = Path(__file__).resolve().parents[2]
SINGLE_ROW_RUNNER = PROJECT_DIR / "run_paper_protocol.py"


def _token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _row_path(
    rows_dir: Path,
    *,
    task: str,
    dataset: str,
    seed: int,
    delta: float,
    sigma: float,
) -> Path:
    return rows_dir / (
        f"{task}_{dataset}_seed{seed}_delta{_token(delta)}_sigma{_token(sigma)}.json"
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def _load_rows(rows_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(rows_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _checkpoint(output_dir: Path, meta: Dict[str, Any], started: float) -> None:
    rows = _load_rows(output_dir / "rows")
    payload = {
        "meta": meta,
        "results": rows,
        "completed_rows": len(rows),
        "elapsed_sec": time.time() - started,
    }
    _atomic_json(output_dir / "candidate_sweep_checkpoint.json", payload)
    _atomic_csv(output_dir / "candidate_sweep_checkpoint.csv", rows)


def _configs(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    for seed, delta, sigma in itertools.product(
        args.seeds, args.deltas, args.sigmas
    ):
        if "classification" in args.tasks:
            for dataset in args.datasets:
                yield {
                    "task": "classification",
                    "dataset": dataset,
                    "n_qubits": 4 if dataset == "bas" else 8,
                    "seed": seed,
                    "delta": delta,
                    "sigma": sigma,
                }
        if "vqe" in args.tasks:
            yield {
                "task": "vqe",
                "dataset": "tfim",
                "n_qubits": 4,
                "seed": seed,
                "delta": delta,
                "sigma": sigma,
            }


def _command(args: argparse.Namespace, config: Dict[str, Any], row_path: Path) -> List[str]:
    command = [
        sys.executable,
        "-u", "-m", "scripts.experiments.run_paper_protocol",
        "--task",
        str(config["task"]),
        "--device",
        str(args.device),
        "--n_qubits",
        str(config["n_qubits"]),
        "--depth",
        str(args.depth),
        "--M",
        str(args.M),
        "--seed",
        str(config["seed"]),
        "--gamma",
        str(args.gamma),
        "--alpha",
        str(args.alpha),
        "--qbeta",
        str(args.qbeta),
        "--delta",
        str(config["delta"]),
        "--rot_pool_k",
        str(args.rot_pool_k),
        "--rot_pool_sigma",
        str(config["sigma"]),
        "--include_cnot",
        "--output_json",
        str(row_path),
    ]
    if config["task"] == "classification":
        command.extend(
            [
                "--dataset",
                str(config["dataset"]),
                "--data_dir",
                str(args.data_dir),
                "--n_train_pool",
                str(args.n_train_pool),
                "--n_val_pool",
                str(args.n_val_pool),
                "--bas_h",
                str(args.bas_h),
                "--bas_w",
                str(args.bas_w),
            ]
        )
        if not args.allow_download:
            command.append("--no_download")
    else:
        command.extend(
            [
                "--vqe_depth",
                str(args.vqe_depth),
                "--vqe_steps",
                str(args.vqe_steps),
                "--vqe_lr",
                str(args.vqe_lr),
                "--J",
                str(args.J),
                "--h",
                str(args.h),
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable original candidate-pool sweep with one JSON checkpoint per row."
    )
    parser.add_argument("--tasks", nargs="+", choices=["classification", "vqe"], default=["classification", "vqe"])
    parser.add_argument("--datasets", nargs="+", choices=["mnist49", "fashion_sb", "bas"], default=["mnist49", "fashion_sb", "bas"])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.01, 0.02])
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.001, 0.003, 0.006, 0.01])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--qbeta", type=float, default=0.0)
    parser.add_argument("--rot_pool_k", type=int, default=4)
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--n_train_pool", type=int, default=4000)
    parser.add_argument("--n_val_pool", type=int, default=2000)
    parser.add_argument("--bas_h", type=int, default=4)
    parser.add_argument("--bas_w", type=int, default=4)
    parser.add_argument("--vqe_depth", type=int, default=6)
    parser.add_argument("--vqe_steps", type=int, default=40)
    parser.add_argument("--vqe_lr", type=float, default=0.1)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    rows_dir = args.output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    configs = list(_configs(args))
    meta = {
        "protocol": "original_candidate_pool",
        "runner": str(Path(__file__).resolve()),
        "single_row_runner": str(SINGLE_ROW_RUNNER),
        "grid": {
            "tasks": args.tasks,
            "datasets": args.datasets,
            "deltas": args.deltas,
            "sigmas": args.sigmas,
            "seeds": args.seeds,
        },
        "expected_rows": len(configs),
        "checkpoint_semantics": "one immutable row JSON per completed configuration",
    }
    started = time.time()
    _atomic_json(args.output_dir / "manifest.json", meta)

    for index, config in enumerate(configs, start=1):
        row_path = _row_path(
            rows_dir,
            task=config["task"],
            dataset=config["dataset"],
            seed=config["seed"],
            delta=config["delta"],
            sigma=config["sigma"],
        )
        label = row_path.stem
        if row_path.exists():
            print(f"[SKIP {index}/{len(configs)}] {label}", flush=True)
            continue
        print(f"[RUN  {index}/{len(configs)}] {label}", flush=True)
        completed = subprocess.run(
            _command(args, config, row_path), cwd=PROJECT_DIR, check=False
        )
        if completed.returncode != 0:
            print(f"[FAIL {index}/{len(configs)}] {label} exit={completed.returncode}", flush=True)
            _checkpoint(args.output_dir, meta, started)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
            continue
        _checkpoint(args.output_dir, meta, started)
        print(f"[DONE {index}/{len(configs)}] {label}", flush=True)

    _checkpoint(args.output_dir, meta, started)
    rows = _load_rows(rows_dir)
    print(
        f"Completed {len(rows)}/{len(configs)} rows. Checkpoint: "
        f"{args.output_dir / 'candidate_sweep_checkpoint.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
