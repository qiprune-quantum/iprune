from __future__ import annotations
import subprocess
import sys


def run(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    base = [sys.executable, "run_paper_protocol.py",
            "--device", "default.qubit",
            "--n_qubits", "4",
            "--depth", "12",
            "--M", "50",
            "--seed", "42",
            "--include_cnot",
            "--rot_pool_k", "4",
            "--rot_pool_sigma", "0.003",
            "--data_dir", "./data",
            "--n_val_total", "2000"]

    for delta in ["0.01", "0.02"]:
        for ds in ["mnist49", "fashion_sb", "bas"]:
            run(base + ["--task", "classification", "--dataset", ds, "--delta", delta])

        run(base + ["--task", "vqe", "--delta", delta, "--vqe_depth", "6", "--vqe_steps", "40", "--vqe_lr", "0.1"])


if __name__ == "__main__":
    main()
