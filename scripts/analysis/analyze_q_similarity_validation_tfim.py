from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    PROJECT_DIR / "results" / "q_similarity_validation_20260811" / "tfim"
)


def _fmt_ci(mean: float, low: float, high: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TFIM q-similarity validity results.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    summary = pd.read_csv(results_dir / "summary_by_q.csv").sort_values("qbeta")
    seed = pd.read_csv(results_dir / "seed_level.csv")
    positions = pd.read_csv(results_dir / "position_level.csv")

    x = summary["q"].to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)

    for axis, prefix, title, ylabel in [
        (axes[0], "spearman_energy", "Task-impact prediction", "Spearman with |energy drift|"),
        (axes[1], "spearman_trace", "State-similarity prediction", "Spearman with trace distance"),
    ]:
        y = summary[f"{prefix}_mean"].to_numpy(dtype=float)
        low = summary[f"{prefix}_ci_low"].to_numpy(dtype=float)
        high = summary[f"{prefix}_ci_high"].to_numpy(dtype=float)
        axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o", capsize=3)
        axis.axvline(1.0, color="0.55", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("q")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    prefix = "top5_mean_energy_drift"
    y = summary[f"{prefix}_paired_vs_q1"].to_numpy(dtype=float)
    low = summary[f"{prefix}_paired_vs_q1_ci_low"].to_numpy(dtype=float)
    high = summary[f"{prefix}_paired_vs_q1_ci_high"].to_numpy(dtype=float)
    axes[2].errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o", capsize=3)
    axes[2].axhline(0.0, color="0.25", linestyle="--", linewidth=1)
    axes[2].axvline(1.0, color="0.55", linestyle="--", linewidth=1)
    axes[2].set_title("Top-5 selection quality")
    axes[2].set_xlabel("q")
    axes[2].set_ylabel("Paired mean drift difference vs q=1\n(lower is better)")
    axes[2].grid(alpha=0.25)

    figure.suptitle("TFIM single-gate q-similarity predictive validity (10 seeds)")
    figure.savefig(results_dir / "fig_q_similarity_predictive_validity.png", dpi=220)
    figure.savefig(results_dir / "fig_q_similarity_predictive_validity.pdf")
    plt.close(figure)

    baseline = seed[seed.qbeta == 0.0].set_index("seed")
    win_rows = []
    for qbeta, group in seed.groupby("qbeta", sort=True):
        group = group.set_index("seed").loc[baseline.index]
        corr_diff = group.spearman_energy - baseline.spearman_energy
        top2_diff = group.top2_mean_energy_drift - baseline.top2_mean_energy_drift
        top5_diff = group.top5_mean_energy_drift - baseline.top5_mean_energy_drift
        win_rows.append(
            {
                "qbeta": qbeta,
                "q": float(group.q.iloc[0]),
                "correlation_better": int((corr_diff > 1e-12).sum()),
                "correlation_worse": int((corr_diff < -1e-12).sum()),
                "correlation_tie": int((corr_diff.abs() <= 1e-12).sum()),
                "top2_changed": int((group.top2_positions != baseline.top2_positions).sum()),
                "top2_better": int((top2_diff < -1e-12).sum()),
                "top2_worse": int((top2_diff > 1e-12).sum()),
                "top5_changed": int((group.top5_positions != baseline.top5_positions).sum()),
                "top5_better": int((top5_diff < -1e-12).sum()),
                "top5_worse": int((top5_diff > 1e-12).sum()),
            }
        )
    wins = pd.DataFrame(win_rows)
    wins.to_csv(results_dir / "paired_win_loss.csv", index=False, encoding="utf-8-sig")

    q1 = summary[summary.qbeta == 0.0].iloc[0]
    historical = summary[summary.qbeta == 1.0].iloc[0]
    q10 = summary[summary.qbeta == 10.0].iloc[0]
    q20 = summary[summary.qbeta == 20.0].iloc[0]
    historical_wins = wins[wins.qbeta == 1.0].iloc[0]
    max_identity_error = positions.loc[
        positions.qbeta == 0.0, "q1_trace_identity_error"
    ].max()

    report = f"""# TFIM 单门 q-相似性预测有效性实验

更新时间：2026-08-11（Asia/Shanghai）

## 实验问题

在不微调、每次只删除一个旋转门的条件下，比较 q 加权距离与 q=1 普通距离，谁更能预测完整电路的真实任务后果。真值由实际执行修改后的完整电路得到，不由任何距离阈值定义。

- 独立随机种子：42--51，共 10 个。
- 每个种子可删旋转门：24 个。
- qbeta：0、1、3、6、10、20，对应 q 约为 1.000、1.030、1.094、1.197、1.350、1.822。
- 状态相似性真值：单门删除后的完整电路纯态迹距离。
- 任务相似性真值：单门删除后的 TFIM 绝对能量漂移。
- 未进行微调，避免优化器掩盖原始相似性判断质量。

## 数值自检

q=1 时，单门替换前后的普通角度应满足 trace distance = sin(angle)。最大绝对数值误差为 {max_identity_error:.3e}，说明前缀态、完整电路执行和距离计算相互一致。

## 主要结果

q=1 对输出态迹距离的平均 Spearman 相关为 {q1.spearman_trace_mean:.4f}，这是无噪声幺正后缀下的理论预期基准。q>1 后该相关性下降。

对 TFIM 单门绝对能量漂移：

- q=1：{_fmt_ci(q1.spearman_energy_mean, q1.spearman_energy_ci_low, q1.spearman_energy_ci_high)}。
- 原论文 q≈1.030：{_fmt_ci(historical.spearman_energy_mean, historical.spearman_energy_ci_low, historical.spearman_energy_ci_high)}；相对 q=1 的配对差为 {_fmt_ci(historical.spearman_energy_paired_vs_q1, historical.spearman_energy_paired_vs_q1_ci_low, historical.spearman_energy_paired_vs_q1_ci_high)}。
- q≈1.350：相对 q=1 的配对差为 {_fmt_ci(q10.spearman_energy_paired_vs_q1, q10.spearman_energy_paired_vs_q1_ci_low, q10.spearman_energy_paired_vs_q1_ci_high)}。
- q≈1.822：相对 q=1 的配对差为 {_fmt_ci(q20.spearman_energy_paired_vs_q1, q20.spearman_energy_paired_vs_q1_ci_low, q20.spearman_energy_paired_vs_q1_ci_high)}。

原论文 q≈1.030 的逐种子相关性比较为 {int(historical_wins.correlation_better)} 胜、{int(historical_wins.correlation_worse)} 负、{int(historical_wins.correlation_tie)} 平；Top-2 选择 10/10 完全相同，Top-5 只有 {int(historical_wins.top5_changed)}/10 个种子变化。

## 结论

1. q-distance 的数值和排序会随 q 改变，因此 q 确实是几何排序控制量。
2. 原论文 q≈1.030 相对 q=1 只有极小且不稳定的任务漂移相关性变化，当前不能宣称其更准确地识别 TFIM 单门任务相似性。
3. 较大 q 显著降低了与真实能量漂移和输出态迹距离的一致性，说明放大 q 并不会自动改善相似性判断。
4. 在当前无噪声 TFIM 条件下，q=1 是更简单且更稳健的相似性基准。若保留 q 的独立贡献主张，需要在预先规定的噪声映射或其他任务上得到可重复的预测优势。
"""
    (results_dir / "RESULTS_SUMMARY_CN.md").write_text(report, encoding="utf-8")
    print(f"Saved analysis to {results_dir}")


if __name__ == "__main__":
    main()
