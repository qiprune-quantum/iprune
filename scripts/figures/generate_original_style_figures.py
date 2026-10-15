from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

import generate_paper_ready_assets as base


OUTPUT_DIR = base.PAPER_DIR / "generated_results_original_style_20260807"

BLUE = "#5B9BD5"
BLUE_DARK = "#4C78A8"
ORANGE = "#F9B23F"
GREEN = "#70AD47"
GRAY = "#B7B7B7"
GRAY_DARK = "#747474"
BLACK = "#111111"
GRID = "#D8D8D8"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.2,
            "axes.linewidth": 0.95,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.8,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, linestyle="--", linewidth=0.55, color=GRID, alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.055,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        va="bottom",
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def asymmetric_yerr(means: Sequence[float], stds: Sequence[float]) -> np.ndarray:
    mean_values = np.asarray(means, dtype=float)
    std_values = np.asarray(stds, dtype=float)
    return np.vstack([np.minimum(mean_values, std_values), std_values])


def annotate_bars(
    ax: plt.Axes,
    bars,
    formatter,
    *,
    y_pad_fraction: float = 0.018,
    fontsize: float = 7.6,
) -> None:
    y0, y1 = ax.get_ylim()
    pad = (y1 - y0) * y_pad_fraction
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + pad,
            formatter(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def candidate_values(
    rows: Sequence[Dict],
    sigmas: Sequence[float],
    delta: float,
    metric: str,
) -> np.ndarray:
    result = []
    for sigma in sigmas:
        row = next(
            item
            for item in rows
            if np.isclose(float(item["delta"]), delta)
            and np.isclose(float(item["rot_pool_sigma"]), sigma)
        )
        result.append(float(row[metric]))
    return np.asarray(result, dtype=float)


def grouped_candidate_panel(
    ax: plt.Axes,
    rows: Sequence[Dict],
    sigmas: Sequence[float],
    deltas: Sequence[float],
    metric: str,
    *,
    formatter,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
) -> None:
    x = np.arange(len(sigmas))
    width = 0.34
    all_bars = []
    for index, (delta, color) in enumerate(zip(deltas, [BLUE, ORANGE])):
        values = candidate_values(rows, sigmas, delta, metric)
        offset = (index - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=color,
            edgecolor=BLACK,
            linewidth=0.75,
            label=fr"$\delta={delta:g}$",
        )
        all_bars.append(bars)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for index, bars in enumerate(all_bars):
        annotate_bars(
            ax,
            bars,
            formatter,
            y_pad_fraction=0.014 + 0.045 * index,
        )
    ax.set_xticks(x, [f"{sigma:g}" for sigma in sigmas])
    if legend:
        ax.legend(loc="upper right", frameon=True, edgecolor="#777777")
    style_axis(ax)


def plot_candidate_classification(rows: Sequence[Dict]) -> None:
    datasets = ["bas", "fashion_sb", "mnist49"]
    titles = ["BAS", "FASHION-SB", "MNIST49"]
    sigmas = sorted({float(row["rot_pool_sigma"]) for row in rows})
    deltas = sorted({float(row["delta"]) for row in rows})

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.05), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.81, wspace=0.25, hspace=0.36)

    for column, (dataset, title) in enumerate(zip(datasets, titles)):
        subset = [row for row in rows if row["dataset"] == dataset]
        grouped_candidate_panel(
            axes[0, column],
            subset,
            sigmas,
            deltas,
            "replace_ratio_over_candidates",
            formatter=lambda value: f"{value:.3f}",
            ylim=(0, 0.69),
        )
        grouped_candidate_panel(
            axes[1, column],
            subset,
            sigmas,
            deltas,
            "L",
            formatter=lambda value: f"{int(round(value))}",
        )
        axes[0, column].set_title(title, pad=7)
        axes[1, column].set_xlabel(r"$\sigma$ (Candidate Pool)")

    axes[0, 0].set_ylabel("Replace Ratio")
    axes[1, 0].set_ylabel("Redundant Candidates (L)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), frameon=True, edgecolor="#777777", ncol=2)
    save(fig, "fig_candidate_pool_classification")


def plot_candidate_vqe(rows: Sequence[Dict]) -> None:
    sigmas = sorted({float(row["rot_pool_sigma"]) for row in rows})
    deltas = sorted({float(row["delta"]) for row in rows})
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 5.2), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.92, hspace=0.30)

    grouped_candidate_panel(
        axes[0], rows, sigmas, deltas, "replace_ratio_over_candidates",
        formatter=lambda value: f"{value:.3f}", ylim=(0, 0.69), legend=True,
    )
    grouped_candidate_panel(
        axes[1], rows, sigmas, deltas, "L",
        formatter=lambda value: f"{int(round(value))}",
    )
    grouped_candidate_panel(
        axes[2], rows, sigmas, deltas, "dE_mean",
        formatter=lambda value: f"{value:.4f}",
    )
    axes[0].set_title("TFIM", pad=8)
    axes[0].set_ylabel("Replace Ratio")
    axes[1].set_ylabel("Redundant Candidates (L)")
    axes[2].set_ylabel(r"Mean $|\Delta E|$")
    axes[2].set_xlabel(r"$\sigma$ (Candidate Pool)")
    save(fig, "fig_candidate_pool_vqe")


def plot_true_pruning(rows: Sequence[Dict]) -> None:
    deltas = sorted({float(row["delta"]) for row in rows})
    groups = [[row for row in rows if float(row["delta"]) == delta] for delta in deltas]
    x = np.arange(len(deltas))
    labels = [f"{delta:g}" for delta in deltas]

    pruning = [base.mean_std(group, "actual_gate_prune_ratio") for group in groups]
    one = [base.mean_std(group, "abs_dE_one_shot_vs_original") for group in groups]
    tuned = [base.mean_std(group, "abs_dE_finetuned_vs_original") for group in groups]
    recovery = [base.mean_std(group, "recovery_fraction_vs_control") for group in groups]

    fig, axes = plt.subplots(3, 1, figsize=(7.16, 5.3), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.92, hspace=0.32)

    prune_mean, prune_std = map(np.asarray, zip(*pruning))
    bars = axes[0].bar(x, prune_mean, color=BLUE, edgecolor=BLACK, linewidth=0.75)
    axes[0].errorbar(x, prune_mean, yerr=asymmetric_yerr(prune_mean, prune_std), fmt="none", ecolor=BLACK, capsize=2.5, linewidth=0.8)
    axes[0].set_ylim(0, 0.60)
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    annotate_bars(axes[0], bars, lambda value: f"{value:.1%}", y_pad_fraction=0.022)
    axes[0].set_ylabel("Physical Pruning Ratio")
    axes[0].set_title("TRUE GATE DELETION ON TFIM VQE", pad=8)
    panel_label(axes[0], "(a)")
    style_axis(axes[0])

    one_mean, one_std = map(np.asarray, zip(*one))
    tuned_mean, tuned_std = map(np.asarray, zip(*tuned))
    width = 0.34
    one_bars = axes[1].bar(x - width / 2, one_mean, width, color=ORANGE, edgecolor=BLACK, linewidth=0.75, label="One-shot")
    tuned_bars = axes[1].bar(x + width / 2, tuned_mean, width, color=BLUE, edgecolor=BLACK, linewidth=0.75, label="Fine-tuned")
    axes[1].errorbar(x - width / 2, one_mean, yerr=asymmetric_yerr(one_mean, one_std), fmt="none", ecolor=BLACK, capsize=2.2, linewidth=0.75)
    axes[1].errorbar(x + width / 2, tuned_mean, yerr=asymmetric_yerr(tuned_mean, tuned_std), fmt="none", ecolor=BLACK, capsize=2.2, linewidth=0.75)
    annotate_bars(axes[1], one_bars, lambda value: f"{value:.2f}", fontsize=7.0)
    annotate_bars(axes[1], tuned_bars, lambda value: f"{value:.2f}", fontsize=7.0)
    axes[1].set_ylabel(r"Absolute $|\Delta E|$")
    axes[1].legend(loc="upper left", frameon=True, edgecolor="#777777", ncol=2)
    panel_label(axes[1], "(b)")
    style_axis(axes[1])

    recovery_mean, recovery_std = map(np.asarray, zip(*recovery))
    axes[2].plot(x, recovery_mean, color=GREEN, marker="o", markersize=4.3, linewidth=2.0)
    axes[2].fill_between(x, np.maximum(0, recovery_mean - recovery_std), np.minimum(1.05, recovery_mean + recovery_std), color=GREEN, alpha=0.17, linewidth=0)
    for index, value in enumerate(recovery_mean):
        axes[2].text(index, value + (0.035 if value < 0.98 else -0.065), f"{value:.1%}", ha="center", fontsize=7.7)
    axes[2].set_ylim(0, 1.08)
    axes[2].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[2].set_ylabel("Recovery Fraction")
    axes[2].set_xlabel(r"$\delta$ (Pruning Threshold Scale)")
    axes[2].set_xticks(x, labels)
    panel_label(axes[2], "(c)")
    style_axis(axes[2])
    save(fig, "fig_true_pruning_finetune")


def plot_fair_baselines(rows: Sequence[Dict]) -> None:
    summary = base.fair_summary(rows)
    labels = [item["label"] for item in summary]
    labels = ["q-iPrune", r"$q=1$", "No Prefix", "Unitary", "Parameter", "Random"]
    colors = [BLUE, "#8FB9DD", GRAY, GRAY_DARK, GRAY, ORANGE]
    metrics = [
        ("accuracy", r"Mean $|\Delta\mathrm{Acc}|$", "{:.4f}"),
        ("energy", r"Mean $|\Delta E|$", "{:.4f}"),
        ("trace", r"Mean max $D_{\mathrm{tr}}$", "{:.4f}"),
    ]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 5.2), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.15, top=0.92, hspace=0.28)

    for index, (metric, ylabel, formatter) in enumerate(metrics):
        means = np.asarray([item[metric][0] for item in summary])
        stds = np.asarray([item[metric][1] for item in summary])
        bars = axes[index].bar(x, means, color=colors, edgecolor=BLACK, linewidth=0.75)
        axes[index].errorbar(x, means, yerr=asymmetric_yerr(means, stds), fmt="none", ecolor=BLACK, capsize=2.3, linewidth=0.75)
        axes[index].set_ylim(bottom=0)
        annotate_bars(axes[index], bars, lambda value, fmt=formatter: fmt.format(value), fontsize=7.1)
        axes[index].set_ylabel(ylabel)
        panel_label(axes[index], f"({chr(ord('a') + index)})")
        style_axis(axes[index])
    axes[0].set_title("MATCHED-BUDGET BASELINE COMPARISON", pad=8)
    axes[-1].set_xticks(x, labels, rotation=18, ha="right")
    save(fig, "fig_fair_baselines")


def plot_gradient(rows: Sequence[Dict]) -> None:
    paired = base.pair_gradient(rows)
    summary = base.gradient_summary(rows)
    deltas = [item["delta"] for item in summary]
    x = np.arange(len(deltas))
    q_mean = np.asarray([item["q"][0] for item in summary])
    q_std = np.asarray([item["q"][1] for item in summary])
    g_mean = np.asarray([item["g"][0] for item in summary])
    g_std = np.asarray([item["g"][1] for item in summary])
    q_wins = np.asarray([item["q_win"] for item in summary])
    g_wins = np.asarray([item["g_win"] for item in summary])
    ties = np.asarray([item["tie"] for item in summary])

    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.8), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.11, top=0.91, hspace=0.32)
    axes[0].errorbar(x, q_mean, yerr=asymmetric_yerr(q_mean, q_std), color=BLUE, marker="o", markersize=4.5, linewidth=2.0, capsize=2.5, label="q-iPrune")
    axes[0].errorbar(x, g_mean, yerr=asymmetric_yerr(g_mean, g_std), color=ORANGE, marker="s", markersize=4.5, linewidth=2.0, capsize=2.5, label="Real Gradient")
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Absolute Gap to Full Control")
    axes[0].set_title("MATCHED-K TFIM VQE COMPARISON", pad=8)
    axes[0].legend(loc="upper left", frameon=True, edgecolor="#777777")
    panel_label(axes[0], "(a)")
    style_axis(axes[0])

    width = 0.24
    qbars = axes[1].bar(x - width, q_wins, width, color=BLUE, edgecolor=BLACK, linewidth=0.75, label="q-iPrune wins")
    gbars = axes[1].bar(x, g_wins, width, color=ORANGE, edgecolor=BLACK, linewidth=0.75, label="Gradient wins")
    tbars = axes[1].bar(x + width, ties, width, color=GRAY, edgecolor=BLACK, linewidth=0.75, label="Ties")
    for bars in [qbars, gbars, tbars]:
        for bar in bars:
            value = int(round(bar.get_height()))
            if value:
                axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.08, str(value), ha="center", fontsize=7.6)
    axes[1].set_ylim(0, 5.8)
    axes[1].set_yticks(range(0, 6))
    axes[1].set_ylabel("Number of Seeds")
    axes[1].set_xlabel(r"$\delta$ (Pruning Threshold Scale)")
    axes[1].set_xticks(x, [f"{delta:g}" for delta in deltas])
    axes[1].legend(loc="upper left", frameon=True, edgecolor="#777777", ncol=3)
    panel_label(axes[1], "(b)")
    style_axis(axes[1])
    save(fig, "fig_qiprune_vs_gradient")


def copy_tables_and_text() -> None:
    names = [
        "results_text.tex",
        "tab_true_pruning_finetune.tex",
        "tab_fair_baselines.tex",
        "tab_qiprune_vs_gradient.tex",
        "tab_candidate_pool_summary.tex",
        "manifest.json",
    ]
    for name in names:
        shutil.copy2(base.OUTPUT_DIR / name, OUTPUT_DIR / name)


def write_include_file() -> None:
    folder = OUTPUT_DIR.name
    text = rf"""% Original-paper-style figures generated from formal checkpoints.
% Insert with: \input{{{folder}/paper_ready_experiments}}
\input{{{folder}/results_text}}

\input{{{folder}/tab_true_pruning_finetune}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=0.94\textwidth]{{{folder}/fig_true_pruning_finetune.pdf}}
  \caption{{Physical gate deletion and post-pruning fine-tuning on TFIM VQE over five seeds. Error bars and shaded bands denote one standard deviation.}}
  \label{{fig:true_pruning_finetune}}
\end{{figure*}}

\input{{{folder}/tab_fair_baselines}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=0.94\textwidth]{{{folder}/fig_fair_baselines.pdf}}
  \caption{{Matched-budget comparison with five baselines over seeds 42--46. Lower values are better; error bars denote one standard deviation.}}
  \label{{fig:fair_baselines}}
\end{{figure*}}

\input{{{folder}/tab_qiprune_vs_gradient}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=0.94\textwidth]{{{folder}/fig_qiprune_vs_gradient.pdf}}
  \caption{{Matched-$k$ comparison between q-iPrune and integrated real-gradient deletion on TFIM VQE over seeds 42--46. Wins are determined by the lower final VQE energy.}}
  \label{{fig:qiprune_gradient}}
\end{{figure*}}

\input{{{folder}/tab_candidate_pool_summary}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=0.94\textwidth]{{{folder}/fig_candidate_pool_classification.pdf}}
  \caption{{Original candidate-pool sweep on BAS, Fashion-SB, and MNIST 4 vs 9 (seed 42). $L$ denotes redundant candidate entries rather than physically deleted circuit gates.}}
  \label{{fig:candidate_pool_classification}}
\end{{figure*}}

\begin{{figure*}}[t]
  \centering
  \includegraphics[width=0.94\textwidth]{{{folder}/fig_candidate_pool_vqe.pdf}}
  \caption{{Original candidate-pool sensitivity sweep on TFIM VQE (seed 42).}}
  \label{{fig:candidate_pool_vqe}}
\end{{figure*}}
"""
    (OUTPUT_DIR / "paper_ready_experiments.tex").write_text(text, encoding="utf-8")


def write_readme() -> None:
    text = f"""# 原论文风格美化版

本目录保留 2026-08-06 版，另生成更接近原论文的排版：蓝橙配色、宽幅分层面板、统一数值标注和更充足的留白。

在 `main.tex` 中插入：

```tex
\\input{{{OUTPUT_DIR.name}/paper_ready_experiments}}
```

正文建议使用真实删门、公平基线和真实梯度三张图；候选池两张图建议放在附录。
"""
    (OUTPUT_DIR / "README_CN.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    candidate_rows = base.load_rows(base.CANDIDATE_PATH)
    true_rows = base.load_rows(base.TRUE_PATH)
    gradient_rows = base.load_rows(base.GRADIENT_PATH)
    fair_rows: List[Dict] = []
    for path in base.FAIR_PATHS:
        fair_rows.extend(base.load_rows(path))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_candidate_classification([row for row in candidate_rows if row["task"] == "classification"])
    plot_candidate_vqe([row for row in candidate_rows if row["task"] == "vqe"])
    plot_true_pruning(true_rows)
    plot_fair_baselines(fair_rows)
    plot_gradient(gradient_rows)
    copy_tables_and_text()
    write_include_file()
    write_readme()
    print(f"Original-style paper assets saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
