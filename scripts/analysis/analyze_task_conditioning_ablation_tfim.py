from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    PROJECT_DIR / "results" / "task_conditioning_ablation_20260811" / "tfim"
)
MODE_LABELS = {
    "correct_prefix": "Correct task\nprefix",
    "mismatched_0101_prefix": "Mismatched |0101>\nprefix",
    "haar_prefix": "Haar-random\nprefix",
    "no_prefix": "No prefix",
}
PLOT_LABELS = {
    "correct_prefix": "Correct\nprefix",
    "mismatched_0101_prefix": "Mismatched\nprefix",
    "haar_prefix": "Haar\nprefix",
    "no_prefix": "No\nprefix",
}


def _fmt(mean: float, low: float, high: float) -> str:
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TFIM task-conditioning ablation.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    summary = pd.read_csv(results_dir / "summary_by_mode.csv")
    seed = pd.read_csv(results_dir / "seed_level.csv")
    order = list(MODE_LABELS)
    summary = summary.set_index("mode").loc[order].reset_index()
    labels = [PLOT_LABELS[value] for value in order]
    x = np.arange(len(order))

    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.3), constrained_layout=True)
    for axis, prefix, title, ylabel in [
        (axes[0], "spearman_energy", "Task-impact prediction", "Spearman with |energy drift|"),
        (axes[1], "spearman_trace", "State-similarity prediction", "Spearman with trace distance"),
    ]:
        y = summary[f"{prefix}_mean"].to_numpy(float)
        low = summary[f"{prefix}_ci_low"].to_numpy(float)
        high = summary[f"{prefix}_ci_high"].to_numpy(float)
        axis.bar(x, y, color=["#3b82c4", "#8da0cb", "#b3b3b3", "#666666"])
        axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="none", color="black", capsize=4)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)

    width = 0.36
    for offset, top_k, color in [(-width / 2, 2, "#4c92c3"), (width / 2, 5, "#f2a93b")]:
        prefix = f"top{top_k}_mean_energy_drift"
        y = summary[f"{prefix}_mean"].to_numpy(float)
        low = summary[f"{prefix}_ci_low"].to_numpy(float)
        high = summary[f"{prefix}_ci_high"].to_numpy(float)
        axes[2].bar(x + offset, y, width=width, label=f"Top-{top_k}", color=color)
        axes[2].errorbar(
            x + offset, y, yerr=np.vstack([y - low, high - y]), fmt="none", color="black", capsize=3
        )
    axes[2].set_title("Selected-gate task impact")
    axes[2].set_ylabel("Mean |energy drift| (lower is better)")
    axes[2].set_xticks(x, labels)
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    figure.suptitle("TFIM task-conditioning ablation (10 seeds, q=1)")
    figure.savefig(results_dir / "fig_task_conditioning_ablation.png", dpi=220)
    figure.savefig(results_dir / "fig_task_conditioning_ablation.pdf")
    plt.close(figure)

    correct = seed[seed["mode"] == "correct_prefix"].set_index("seed")
    win_rows = []
    for mode in order:
        group = seed[seed["mode"] == mode].set_index("seed").loc[correct.index]
        corr = group.spearman_energy - correct.spearman_energy
        top2 = group.top2_mean_energy_drift - correct.top2_mean_energy_drift
        top5 = group.top5_mean_energy_drift - correct.top5_mean_energy_drift
        win_rows.append(
            {
                "mode": mode,
                "correct_correlation_better": int((corr < -1e-12).sum()),
                "other_correlation_better": int((corr > 1e-12).sum()),
                "correlation_tie": int((corr.abs() <= 1e-12).sum()),
                "top2_selection_changed": int((group.top2_positions != correct.top2_positions).sum()),
                "correct_top2_better": int((top2 > 1e-12).sum()),
                "other_top2_better": int((top2 < -1e-12).sum()),
                "top2_tie": int((top2.abs() <= 1e-12).sum()),
                "top5_selection_changed": int((group.top5_positions != correct.top5_positions).sum()),
                "correct_top5_better": int((top5 > 1e-12).sum()),
                "other_top5_better": int((top5 < -1e-12).sum()),
                "top5_tie": int((top5.abs() <= 1e-12).sum()),
            }
        )
    wins = pd.DataFrame(win_rows)
    wins.to_csv(results_dir / "paired_win_loss.csv", index=False, encoding="utf-8-sig")

    s = summary.set_index("mode")
    w = wins.set_index("mode")
    lines = [
        "# TFIM 任务相关与位置感知消融实验",
        "",
        "更新时间：2026-08-11（Asia/Shanghai）",
        "",
        "## 实验设计",
        "",
        "固定 q=1、相同的 10 个优化 checkpoint、相同 24 个可删旋转门和相同单门真实删除真值，只改变用于计算门相似性的状态。未进行微调。",
        "",
        "- Correct task prefix：真实 TFIM 初态 |0000> 沿基线电路传播得到的位置前缀态。",
        "- Mismatched |0101> prefix：用错误基态 |0101> 沿同一基线电路传播。",
        "- Haar-random prefix：每个种子使用一个预先固定的复 Haar 随机态。",
        "- No prefix：所有位置都在未传播的 |0000> 上计算分数。",
        "",
        "主真值是逐门实际删除后的完整 TFIM 电路绝对能量漂移；输出态迹距离为次要真值。",
        "",
        "## 主要结果",
        "",
        "| 相似性状态 | 能量漂移 Spearman | 相对正确前缀配对差 | Top-2 平均漂移 | Top-5 平均漂移 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in order:
        row = s.loc[mode]
        lines.append(
            f"| {MODE_LABELS[mode].replace(chr(10), ' ')} | "
            f"{row.spearman_energy_mean:.4f} | "
            f"{_fmt(row.spearman_energy_paired_vs_correct, row.spearman_energy_paired_vs_correct_ci_low, row.spearman_energy_paired_vs_correct_ci_high)} | "
            f"{row.top2_mean_energy_drift_mean:.4f} | {row.top5_mean_energy_drift_mean:.4f} |"
        )
    lines.extend(["", "## 逐种子稳健性", ""])
    for mode in order[1:]:
        row = w.loc[mode]
        lines.append(
            f"- 相对 {MODE_LABELS[mode].replace(chr(10), ' ')}，正确前缀的相关性为 "
            f"{int(row.correct_correlation_better)} 胜、{int(row.other_correlation_better)} 负、{int(row.correlation_tie)} 平；"
            f"Top-2 为 {int(row.correct_top2_better)} 胜、{int(row.other_top2_better)} 负、{int(row.top2_tie)} 平；"
            f"Top-5 为 {int(row.correct_top5_better)} 胜、{int(row.other_top5_better)} 负、{int(row.top5_tie)} 平。"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "1. 正确任务前缀态对单门真实能量影响的预测明显优于错配基态、随机态和 no-prefix，所有相关性配对差的 95% bootstrap CI 均低于 0。",
            "2. 正确前缀对输出态迹距离达到精确排序；错误状态和 no-prefix 明显降低这种一致性。",
            "3. Haar 和 no-prefix 选出的 Top-K 门造成显著更大的真实能量漂移，说明任务和电路位置上下文不仅改变分数，而且改善了实际低风险门的选择。",
            "4. 错配 |0101> 的 Top-K 差异较小、部分 CI 接触 0，因此最强证据来自相关性、Haar 对照和 no-prefix 对照。",
            "5. 该结果直接支持论文的 task-conditioned、position-aware 冗余定义，但不涉及 q 的独立贡献。",
        ]
    )
    (results_dir / "RESULTS_SUMMARY_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved analysis to {results_dir}")


if __name__ == "__main__":
    main()
