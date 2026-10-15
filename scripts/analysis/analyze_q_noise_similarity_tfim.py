from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "q_noise_validation_20260811" / "tfim"
CHANNEL_LABELS = {
    "depolarizing": "Depolarizing",
    "amplitude_damping": "Amplitude damping",
}
VARIANT_LABELS = {"q1": "q=1", "q_lambda": "q(lambda)", "fixed_q": "fixed q=1.35"}
COLORS = {"q1": "#3b82c4", "q_lambda": "#f2a93b", "fixed_q": "#777777"}


def _fmt(mean: float, low: float, high: float) -> str:
    return f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TFIM q noise validation.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    summary = pd.read_csv(results_dir / "summary_by_noise.csv")
    seed = pd.read_csv(results_dir / "seed_level.csv")

    figure, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), constrained_layout=True)
    for row_index, channel in enumerate(("depolarizing", "amplitude_damping")):
        channel_rows = summary[summary.channel == channel]
        for variant in ("q1", "q_lambda", "fixed_q"):
            group = channel_rows[channel_rows.variant == variant].sort_values("noise_strength")
            x = group.noise_strength.to_numpy(float)
            for column, prefix in enumerate(("spearman_energy", "spearman_trace")):
                y = group[f"{prefix}_mean"].to_numpy(float)
                low = group[f"{prefix}_ci_low"].to_numpy(float)
                high = group[f"{prefix}_ci_high"].to_numpy(float)
                axes[row_index, column].plot(
                    x, y, marker="o", label=VARIANT_LABELS[variant], color=COLORS[variant]
                )
                axes[row_index, column].fill_between(x, low, high, alpha=0.12, color=COLORS[variant])
            prefix = "top5_mean_energy_drift"
            y = group[f"{prefix}_paired_vs_q1"].to_numpy(float)
            low = group[f"{prefix}_paired_vs_q1_ci_low"].to_numpy(float)
            high = group[f"{prefix}_paired_vs_q1_ci_high"].to_numpy(float)
            axes[row_index, 2].plot(
                x, y, marker="o", label=VARIANT_LABELS[variant], color=COLORS[variant]
            )
            axes[row_index, 2].fill_between(x, low, high, alpha=0.12, color=COLORS[variant])

        axes[row_index, 0].set_ylabel(
            f"{CHANNEL_LABELS[channel]}\nSpearman with |energy drift|"
        )
        axes[row_index, 1].set_ylabel("Spearman with trace distance")
        axes[row_index, 2].set_ylabel("Top-5 paired drift difference vs q=1\n(lower is better)")
        for column in range(3):
            axes[row_index, column].set_xlabel("Noise strength p")
            axes[row_index, column].grid(alpha=0.25)
        axes[row_index, 2].axhline(0.0, color="black", linestyle="--", linewidth=1)

    axes[0, 0].set_title("Task-impact prediction")
    axes[0, 1].set_title("Mixed-state similarity prediction")
    axes[0, 2].set_title("Top-5 selection quality")
    axes[0, 0].legend(loc="best")
    figure.suptitle("TFIM q-geometry under gate-local noise (10 seeds)")
    figure.savefig(results_dir / "fig_q_noise_similarity.png", dpi=220)
    figure.savefig(results_dir / "fig_q_noise_similarity.pdf")
    plt.close(figure)

    q1 = seed[seed.variant == "q1"].set_index(["seed", "channel", "noise_strength"])
    win_rows = []
    for (channel, strength, variant), group in seed.groupby(
        ["channel", "noise_strength", "variant"], sort=True
    ):
        group = group.set_index(["seed", "channel", "noise_strength"]).loc[q1.index.intersection(
            group.set_index(["seed", "channel", "noise_strength"]).index
        )]
        base = q1.loc[group.index]
        corr = group.spearman_energy - base.spearman_energy
        top5 = group.top5_mean_energy_drift - base.top5_mean_energy_drift
        win_rows.append(
            {
                "channel": channel,
                "noise_strength": strength,
                "variant": variant,
                "correlation_better": int((corr > 1e-12).sum()),
                "correlation_worse": int((corr < -1e-12).sum()),
                "correlation_tie": int((corr.abs() <= 1e-12).sum()),
                "top5_better": int((top5 < -1e-12).sum()),
                "top5_worse": int((top5 > 1e-12).sum()),
                "top5_tie": int((top5.abs() <= 1e-12).sum()),
                "top5_selection_changed": int((group.top5_positions != base.top5_positions).sum()),
            }
        )
    wins = pd.DataFrame(win_rows)
    wins.to_csv(results_dir / "paired_win_loss.csv", index=False, encoding="utf-8-sig")

    s = summary.set_index(["channel", "noise_strength", "variant"])
    lines = [
        "# TFIM 真实噪声下 q-相似性严格对照",
        "",
        "更新时间：2026-08-11（Asia/Shanghai）",
        "",
        "## 预注册式实验协议",
        "",
        "- 独立随机种子：42--51，共 10 个；每个种子 24 个可删旋转门。",
        "- 噪声通道：Depolarizing、Amplitude damping。",
        "- 噪声强度：p={0, 0.001, 0.003, 0.01, 0.03, 0.05}，报告完整曲线。",
        "- 比较：q=1、原映射 q(lambda)=exp(0.6p)、预先固定 q=1.35。",
        "- 每个保留门执行后，对其涉及的每根线独立施加相同强度的噪声；删门同时移除该门对应的噪声事件。",
        "- q-score 使用理想 |0000> 基线前缀态，因为当前论文 q-overlap 只定义于纯态；真值由 default.mixed 完整混态电路实际执行得到。",
        "- 主真值：单门删除后的 noisy TFIM 绝对能量漂移；次真值：混态迹距离。",
        "- 不微调，不根据结果挑 q。",
        "",
        "## 最高噪声点 p=0.05",
        "",
        "| 通道 | 方案 | q | 能量相关性 | 相对 q=1 配对差 | Top-5 漂移配对差 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for channel in ("depolarizing", "amplitude_damping"):
        for variant in ("q1", "q_lambda", "fixed_q"):
            row = s.loc[(channel, 0.05, variant)]
            lines.append(
                f"| {CHANNEL_LABELS[channel]} | {VARIANT_LABELS[variant]} | {row.q:.6f} | "
                f"{row.spearman_energy_mean:.4f} | "
                f"{_fmt(row.spearman_energy_paired_vs_q1, row.spearman_energy_paired_vs_q1_ci_low, row.spearman_energy_paired_vs_q1_ci_high)} | "
                f"{_fmt(row.top5_mean_energy_drift_paired_vs_q1, row.top5_mean_energy_drift_paired_vs_q1_ci_low, row.top5_mean_energy_drift_paired_vs_q1_ci_high)} |"
            )
    lines.extend(
        [
            "",
            "## 曲线级结论",
            "",
            "1. 原映射 q(lambda) 在最高噪声下也只有 q=1.030455，因此大部分噪声点的排序和 Top-K 与 q=1 完全相同。",
            "2. q(lambda) 在两种噪声的完整曲线上，相对 q=1 的能量相关性差异都接近 0，所有实质性差异的 95% bootstrap CI 均包含 0。",
            "3. 固定 q=1.35 在两种通道和所有强度下都降低能量漂移相关性；其配对 CI 多数完全低于 0。",
            "4. q(lambda) 的 Top-2 全曲线与 q=1 完全相同；Top-5 只有零星变化，没有稳定曲线级优势。",
            "5. 随噪声增大，q=1 与混态迹距离不再保持无噪声下的精确单调关系，但当前 Hamming-weight q 几何并未改善这种偏离。",
            "6. 因此不触发昂贵 matched-K 微调复核的预定条件：当前实验没有出现稳定的 q 预测信号。",
            "",
            "## 论文边界",
            "",
            "该实验首次真正加入物理噪声通道，但结果不支持当前 q(lambda) 映射具有已验证的噪声自适应收益。论文应把 q 定位为可调几何扩展，并将 task-conditioned、position-aware、physical-certificate pruning 作为实证核心。",
        ]
    )
    (results_dir / "RESULTS_SUMMARY_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved analysis to {results_dir}")


if __name__ == "__main__":
    main()
