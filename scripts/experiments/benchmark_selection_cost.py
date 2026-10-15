"""Reproducible TFIM selection-time/RSS benchmark on trained checkpoints.

Training is deliberately excluded from selection measurements.  Each selector
runs in a fresh process after imports and checkpoint loading, so its peak RSS
and elapsed time measure the same stage.  The direct probe is the manuscript's
one-shot single-gate energy ranking, truncated to iPrune's realized L.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from iprune.experiments.protocol import PaperProtocolConfig, make_q_geometry, simulate_outputs
from iprune.experiments.tasks_vqe import VQEConfig, optimize_vqe_parameters_with_diagnostics
from iprune.gates import build_hea_gates_from_parameters
from iprune.pruning_true import q_iPrune_iterative_identity_audited
from scripts.experiments.run_vqe_qiprune_vs_gradient_seed import _single_gate_energy_probe_ranking


def checkpoint_path(root: Path, n: int, seed: int) -> Path:
    return root / "checkpoints" / f"tfim_n{n}_d4_seed{seed}.npz"


def train(root: Path, n: int, seed: int, steps: int) -> None:
    path = checkpoint_path(root, n, seed)
    if path.exists():
        print(f"CHECKPOINT_EXISTS {path}", flush=True)
        return
    result = optimize_vqe_parameters_with_diagnostics(
        VQEConfig(
            n_qubits=n, depth=4, steps=steps, lr=0.1, seed=seed,
            J=1.0, h=1.0, min_steps=200,
            convergence_window=20, convergence_tol=5e-5,
        ),
        device="default.qubit",
    )
    if not result.converged:
        raise RuntimeError(
            f"N={n} seed={seed} did not converge after {result.steps_run} steps"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, params=result.params, n_qubits=n, depth=4, seed=seed,
        steps_run=result.steps_run,
        final_window_range=result.final_window_range,
    )
    print(f"TRAINED {path} steps={result.steps_run}", flush=True)


def monitored_call(callback):
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def poll() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, process.memory_info().rss)
            stop.wait(0.002)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    start = time.perf_counter_ns()
    try:
        value = callback()
    finally:
        elapsed_ns = time.perf_counter_ns() - start
        stop.set()
        watcher.join()
        peak = max(peak, process.memory_info().rss)
    return value, elapsed_ns / 1e6, baseline, peak


def worker(root: Path, n: int, seed: int, delta: float, method: str, repeat: int) -> None:
    checkpoint = np.load(checkpoint_path(root, n, seed))
    params = np.asarray(checkpoint["params"], dtype=np.float64)
    all_gates, baseline_circuit = build_hea_gates_from_parameters(
        params, include_cnot=True, include_identity_rot=True
    )
    initial_state = np.zeros((1, 2**n), dtype=np.complex128)
    initial_state[0, 0] = 1.0
    baseline_state = simulate_outputs(
        baseline_circuit, initial_state, n, "default.qubit"
    )
    config = PaperProtocolConfig(
        device="default.qubit", n_qubits=n, depth=4, M=1, seed=seed,
        gamma=0.05, alpha=0.6, qbeta=0.0, delta=delta,
        rot_pool_k=1, rot_pool_sigma=0.0, include_cnot=True,
    )
    _, _, gq_diag, _, epsilon = make_q_geometry(config)

    def run_iprune():
        return q_iPrune_iterative_identity_audited(
            all_gates=all_gates, baseline_circuit=baseline_circuit,
            n_qubits=n, device_name="default.qubit", task_states=initial_state,
            Gq_diag=gq_diag, q_epsilon=epsilon, physical_epsilon=epsilon,
        )

    def run_probe():
        return _single_gate_energy_probe_ranking(
            baseline_circuit=baseline_circuit, baseline_state=baseline_state,
            n_qubits=n, J=1.0, h=1.0, device="default.qubit",
        )

    # Warm up the chosen code path before measuring Python/PennyLane execution.
    warm = run_iprune() if method == "iprune" else run_probe()
    measured, elapsed_ms, base_rss, peak_rss = monitored_call(
        run_iprune if method == "iprune" else run_probe
    )
    if method == "iprune":
        selected = list(measured.accepted_positions)
        deletion_count = len(selected)
        assert selected == list(warm.accepted_positions)
    else:
        selected = list(measured[0])
        deletion_count = None
        assert selected == list(warm[0])
    payload = {
        "n_qubits": n, "depth": 4, "seed": seed, "delta": delta,
        "method": method, "repeat": repeat,
        "elapsed_ms": elapsed_ms,
        "baseline_rss_mib": base_rss / 2**20,
        "peak_rss_mib": peak_rss / 2**20,
        "peak_increment_mib": (peak_rss - base_rss) / 2**20,
        "deletion_count": deletion_count,
        "selected_positions": selected,
        "rotation_count": sum(g.gate_type == "Rot" for g in baseline_circuit),
        "training_steps": int(checkpoint["steps_run"]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    output = root / "raw" / f"n{n}_seed{seed}_{method}_r{repeat}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"MEASURED {output} {elapsed_ms:.1f} ms {peak_rss / 2**20:.1f} MiB", flush=True)


def run(root: Path, sizes: list[int], seeds: list[int], delta: float, repeats: int, steps: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        for seed in seeds:
            subprocess.run(
                [sys.executable, __file__, "train", "--root", str(root),
                 "--n", str(n), "--seed", str(seed), "--steps", str(steps)],
                check=True,
            )
            for repeat in range(1, repeats + 1):
                for method in ("iprune", "probe"):
                    output = root / "raw" / f"n{n}_seed{seed}_{method}_r{repeat}.json"
                    if output.exists():
                        continue
                    subprocess.run(
                        [sys.executable, __file__, "worker", "--root", str(root),
                         "--n", str(n), "--seed", str(seed), "--delta", str(delta),
                         "--method", method, "--repeat", str(repeat)],
                        check=True,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "worker", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[3001, 3002, 3003])
    parser.add_argument("--delta", type=float, default=1.2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--method", choices=["iprune", "probe"])
    parser.add_argument("--repeat", type=int)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.root, args.n, args.seed, args.steps)
    elif args.mode == "worker":
        worker(args.root, args.n, args.seed, args.delta, args.method, args.repeat)
    else:
        run(args.root, args.sizes, args.seeds, args.delta, args.repeats, args.steps)


if __name__ == "__main__":
    main()
