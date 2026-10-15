from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_DIR = Path(__file__).resolve().parents[2]
FAIR_SWEEP = PROJECT_DIR / "run_fair_baseline_sweep_resumable.py"


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _stamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for the seed-42 fair-baseline batch, then continue seeds 43-46."
    )
    parser.add_argument("--wait_pid", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--status_json", type=Path, required=True)
    args = parser.parse_args()

    status = {
        "experiment_family": "supplementary_upgrade",
        "queue": "fair_baselines_multiseed",
        "wait_pid": args.wait_pid,
        "next_seeds": [43, 44, 45, 46],
        "state": "waiting_for_seed42",
        "updated_at": _stamp(),
    }
    _atomic_json(args.status_json.resolve(), status)
    print(f"[QUEUE] waiting for seed-42 PID {args.wait_pid}", flush=True)
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"Wait-Process -Id {args.wait_pid} -ErrorAction SilentlyContinue",
        ],
        check=False,
    )

    status["state"] = "running_seed43_46"
    status["updated_at"] = _stamp()
    _atomic_json(args.status_json.resolve(), status)
    print("[QUEUE] seed-42 finished; starting seeds 43-46", flush=True)
    command = [
        sys.executable,
        "-u",
        str(FAIR_SWEEP),
        "--output_dir",
        str(args.output_dir.resolve()),
        "--seeds",
        "43",
        "44",
        "45",
        "46",
        "--deltas",
        "0.01",
        "0.02",
        "--sigmas",
        "0.001",
        "0.003",
        "0.006",
        "0.01",
        "--continue_on_error",
    ]
    completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
    status["state"] = "complete" if completed.returncode == 0 else "failed"
    status["exit_code"] = completed.returncode
    status["updated_at"] = _stamp()
    _atomic_json(args.status_json.resolve(), status)
    print(f"[QUEUE] finished with exit={completed.returncode}", flush=True)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
