from __future__ import annotations
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_xy(df: pd.DataFrame, x: str, y: str, title: str, out_path: str):
    df2 = df.sort_values(x)
    plt.figure()
    plt.plot(df2[x].values, df2[y].values, marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser("Plot ablation CSV")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs_ablation/plots")
    args = ap.parse_args()

    _ensure_out_dir(args.out_dir)
    df = pd.read_csv(args.csv)

    # sigma sweep
    df_sigma = df[df["sweep"] == "sigma"].copy()
    if len(df_sigma) > 0:
        plot_xy(
            df_sigma, "rot_pool_sigma", "trace_dist_max",
            "Max trace distance vs sigma (paper-aligned)",
            os.path.join(args.out_dir, "sigma_vs_max_trace_dist.png"),
        )
        plot_xy(
            df_sigma, "rot_pool_sigma", "L",
            "L (replacements) vs sigma",
            os.path.join(args.out_dir, "sigma_vs_L.png"),
        )
        plot_xy(
            df_sigma, "rot_pool_sigma", "dq_max_global",
            "Global dq_max vs sigma",
            os.path.join(args.out_dir, "sigma_vs_dqmax.png"),
        )

    # delta sweep
    df_delta = df[df["sweep"] == "delta"].copy()
    if len(df_delta) > 0:
        plot_xy(
            df_delta, "delta", "trace_dist_max",
            "Max trace distance vs delta (paper-aligned)",
            os.path.join(args.out_dir, "delta_vs_max_trace_dist.png"),
        )
        plot_xy(
            df_delta, "delta", "L",
            "L (replacements) vs delta",
            os.path.join(args.out_dir, "delta_vs_L.png"),
        )

    # k sweep
    df_k = df[df["sweep"] == "k"].copy()
    if len(df_k) > 0:
        plot_xy(
            df_k, "rot_pool_k", "trace_dist_max",
            "Max trace distance vs K (paper-aligned)",
            os.path.join(args.out_dir, "K_vs_max_trace_dist.png"),
        )
        plot_xy(
            df_k, "rot_pool_k", "L",
            "L (replacements) vs K",
            os.path.join(args.out_dir, "K_vs_L.png"),
        )

    print(f"Plots saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
