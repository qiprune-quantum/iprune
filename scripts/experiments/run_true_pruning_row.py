from __future__ import annotations

import argparse
import json
from pathlib import Path

from iprune.experiments.protocol import PaperProtocolConfig
from scripts.legacy.sweep_all_v2 import run_one_classification, run_one_vqe


PROJECT_DIR = Path(__file__).resolve().parents[2]


def _atomic_new_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and save one schema-v2 true identity-gate pruning row."
    )
    parser.add_argument("--task", choices=["classification", "vqe"], required=True)
    parser.add_argument("--dataset", choices=["mnist49", "fashion_sb", "bas"], default="bas")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--qbeta", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--rot_pool_k", type=int, default=4)
    parser.add_argument("--rot_pool_sigma", type=float, default=0.006)
    parser.add_argument("--include_cnot", action="store_true")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--no_download", action="store_true")
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

    cfg = PaperProtocolConfig(
        device=args.device,
        n_qubits=args.n_qubits,
        depth=args.depth,
        M=args.M,
        seed=args.seed,
        gamma=args.gamma,
        alpha=args.alpha,
        qbeta=args.qbeta,
        delta=args.delta,
        rot_pool_k=args.rot_pool_k,
        rot_pool_sigma=args.rot_pool_sigma,
        include_cnot=args.include_cnot,
    )
    if args.task == "classification":
        row = run_one_classification(
            dataset=args.dataset,
            cfg=cfg,
            data_dir=str(args.data_dir),
            download=not args.no_download,
            n_train_pool=args.n_train_pool,
            n_val_pool=args.n_val_pool,
            bas_h=args.bas_h,
            bas_w=args.bas_w,
        )
    else:
        row = run_one_vqe(
            cfg=cfg,
            vqe_depth=args.vqe_depth,
            vqe_steps=args.vqe_steps,
            vqe_lr=args.vqe_lr,
            J=args.J,
            h=args.h,
        )
    _atomic_new_json(args.output_json, row)
    print(
        f"[OK] task={row['task']} dataset={row['dataset']} L_gate={row['L']} "
        f"gates={row['gate_count_before']}->{row['gate_count_after']} "
        f"depth={row['logical_depth_before']}->{row['logical_depth_after']}",
        flush=True,
    )
    print(f"Saved: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
