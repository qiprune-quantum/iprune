from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    PROJECT_DIR / "results" / "task_conditioning_ablation_20260811" / "mnist49"
)
ORDER = [
    "correct_mnist49_prefix",
    "mismatched_fashion_prefix",
    "haar_prefix",
    "no_prefix",
]
LABELS = ["Correct MNIST\nprefix", "Fashion\nprefix", "Haar\nprefix", "No\nprefix"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MNIST49 task-conditioning ablation.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    summary = pd.read_csv(results_dir / "summary_by_mode.csv").set_index("mode").loc[ORDER]
    seed = pd.read_csv(results_dir / "seed_level.csv")
    positions = pd.read_csv(results_dir / "position_level.csv")

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    x = np.arange(len(ORDER))
    for axis, metric, title, ylabel in [
        (axes[0], "spearman_logit", "Task-impact prediction", "Spearman with mean |logit drift|"),
        (axes[1], "spearman_trace", "State-similarity prediction", "Spearman with mean trace distance"),
    ]:
        y = summary[f"{metric}_mean"].to_numpy(float)
        low = summary[f"{metric}_ci_low"].to_numpy(float)
        high = summary[f"{metric}_ci_high"].to_numpy(float)
        axis.bar(x, y, color=["#3b82c4", "#8da0cb", "#b3b3b3", "#666666"])
        axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="none", color="black", capsize=4)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, LABELS)
        axis.grid(axis="y", alpha=0.25)

    width = 0.36
    for offset, top_k, color in [(-width / 2, 2, "#4c92c3"), (width / 2, 5, "#f2a93b")]:
        metric = f"top{top_k}_mean_logit_drift"
        y = summary[f"{metric}_mean"].to_numpy(float)
        low = summary[f"{metric}_ci_low"].to_numpy(float)
        high = summary[f"{metric}_ci_high"].to_numpy(float)
        axes[2].bar(x + offset, y, width=width, label=f"Top-{top_k}", color=color)
        axes[2].errorbar(x + offset, y, yerr=np.vstack([y - low, high - y]), fmt="none", color="black", capsize=3)
    axes[2].set_title("Selected-gate task impact")
    axes[2].set_ylabel("Mean |test-logit drift| (lower is better)")
    axes[2].set_xticks(x, LABELS)
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    figure.suptitle("MNIST49 task-conditioning ablation (5 seeds, worst-case score)")
    figure.savefig(results_dir / "fig_task_conditioning_ablation_mnist49.png", dpi=220)
    figure.savefig(results_dir / "fig_task_conditioning_ablation_mnist49.pdf")
    plt.close(figure)

    correct_seed = seed[seed["mode"] == ORDER[0]].set_index("seed")
    correct_scores = positions[positions["mode"] == ORDER[0]].set_index(["seed", "position"]).score
    audit_rows = []
    for mode in ORDER:
        group = seed[seed["mode"] == mode].set_index("seed").loc[correct_seed.index]
        mode_scores = positions[positions["mode"] == mode].set_index(["seed", "position"]).score.loc[correct_scores.index]
        score_diff = (mode_scores - correct_scores).abs()
        audit_rows.append(
            {
                "mode": mode,
                "max_score_change": float(score_diff.max()),
                "mean_score_change": float(score_diff.mean()),
                "seeds_with_top2_change": int((group.top2_positions != correct_seed.top2_positions).sum()),
                "seeds_with_top5_change": int((group.top5_positions != correct_seed.top5_positions).sum()),
                "seeds_with_correlation_change": int(
                    ((group.spearman_logit - correct_seed.spearman_logit).abs() > 1e-12).sum()
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(results_dir / "selection_invariance_audit.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# MNIST49 任务状态错配消融复核",
        "",
        "更新时间：2026-08-11（Asia/Shanghai）",
        "",
        "## 协议",
        "",
        "复用 seeds 42--46 的 5 个训练快照和本地数据。每个种子使用 50 个正确 MNIST49 校准态、50 个 Fashion-SB 错配态、50 个固定 Haar 态以及 no-prefix；固定 q=1 和论文的 worst-case maximum angle 聚合。主真值是 512 个独立 MNIST49 测试样本上的平均绝对 logit 漂移，不微调。",
        "",
        "## 结果",
        "",
        "| 状态模式 | Logit 漂移 Spearman | 相对正确状态配对差 | Top-2 平均漂移 | Top-5 平均漂移 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in zip(ORDER, LABELS):
        row = summary.loc[mode]
        lines.append(
            f"| {label.replace(chr(10), ' ')} | {row.spearman_logit_mean:.4f} | "
            f"{row.spearman_logit_paired_vs_correct:+.4f} "
            f"[{row.spearman_logit_paired_vs_correct_ci_low:+.4f}, {row.spearman_logit_paired_vs_correct_ci_high:+.4f}] | "
            f"{row.top2_mean_logit_drift_mean:.5f} | {row.top5_mean_logit_drift_mean:.5f} |"
        )
    lines.extend(
        [
            "",
            "## 结论与边界",
            "",
            "1. 四种状态模式的 Top-2 和 Top-5 在全部 5 个种子中完全相同，任务状态变化没有跨过当前安全工作区间的选门边界。",
            "2. 分数数值确实发生变化，但排序只在 seed 46 的较后位置轻微改变，因此主要预测与 Top-K 结果几乎完全一致。",
            "3. TFIM 上观察到的 task-prefix 强优势没有在当前 MNIST49 worst-case-max 规则下复现，不能宣称任务状态选择在所有任务上都稳定改善选门。",
            "4. 这与此前 MNIST q 扫描中 fixed-K 选门不敏感的现象一致，说明分类线路的当前门排序间隔较大，或 worst-case 聚合弱化了数据分布差异。",
            "5. 不事后改用 mean 聚合寻找正结果；该负结果作为任务依赖性边界保留。",
        ]
    )
    (results_dir / "RESULTS_SUMMARY_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved analysis to {results_dir}")


if __name__ == "__main__":
    main()
