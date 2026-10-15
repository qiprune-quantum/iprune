from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
PAPER_DIR = PROJECT_DIR.parent / "Shao_Paper_4_v2"
OUTPUT_DIR = PAPER_DIR / "generated_results_20260806"

CANDIDATE_PATH = (
    PROJECT_DIR
    / "results"
    / "original_candidate_20260805"
    / "candidate_sweep_checkpoint.json"
)
TRUE_PATH = (
    PROJECT_DIR
    / "results"
    / "supplementary_upgrade_20260806"
    / "true_pruning_vqe_finetune_seed42_46"
    / "true_pruning_finetune_checkpoint.json"
)
GRADIENT_PATH = (
    PROJECT_DIR
    / "results"
    / "supplementary_upgrade_20260806"
    / "vqe_qiprune_vs_real_gradient_seed42_46"
    / "qiprune_vs_gradient_checkpoint.json"
)
FAIR_PATHS = [
    PROJECT_DIR
    / "results"
    / "supplementary_upgrade_20260805"
    / "fair_baselines_seed42"
    / "fair_baseline_checkpoint.json",
    PROJECT_DIR
    / "results"
    / "supplementary_upgrade_20260805"
    / "fair_baselines_seed43_46"
    / "fair_baseline_checkpoint.json",
]

BLUE = "#4C78A8"
BLUE_LIGHT = "#8FB9DD"
ORANGE = "#F2A541"
GREEN = "#59A14F"
GRAY = "#B7B7B7"
GRAY_DARK = "#6F6F6F"
BLACK = "#202020"
GRID = "#D9D9D9"

METHOD_ORDER = [
    "qiprune_positionaware",
    "q1_positionaware_matched_selected_L",
    "qiprune_no_prefix_matched_selected_L",
    "unitary_overlap_matched_selected_L",
    "parameter_euclidean_matched_selected_L",
    "random_reference_matched_selected_L",
]
METHOD_LABELS = ["q-iPrune", "$q=1$", "No prefix", "Unitary", "Parameter", "Random"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["results"])


def numeric(rows: Sequence[Dict], key: str, absolute: bool = False) -> np.ndarray:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) is not None],
        dtype=np.float64,
    )
    return np.abs(values) if absolute else values


def mean_std(rows: Sequence[Dict], key: str, absolute: bool = False) -> tuple[float, float]:
    sample = numeric(rows, key, absolute=absolute)
    if len(sample) == 0:
        return np.nan, np.nan
    std = float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0
    return float(np.mean(sample)), std


def paper_axes(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linestyle="--", linewidth=0.55, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="bottom",
    )


def clipped_yerr(means: Sequence[float], stds: Sequence[float]) -> np.ndarray:
    means_array = np.asarray(means, dtype=float)
    std_array = np.asarray(stds, dtype=float)
    return np.vstack([np.minimum(means_array, std_array), std_array])


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_true_pruning(rows: Sequence[Dict]) -> None:
    deltas = sorted({float(row["delta"]) for row in rows})
    groups = [[row for row in rows if float(row["delta"]) == delta] for delta in deltas]
    x = np.arange(len(deltas))

    prune = [mean_std(group, "actual_gate_prune_ratio") for group in groups]
    one_shot = [mean_std(group, "abs_dE_one_shot_vs_original") for group in groups]
    fine_tuned = [mean_std(group, "abs_dE_finetuned_vs_original") for group in groups]
    recovery = [mean_std(group, "recovery_fraction_vs_control") for group in groups]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), constrained_layout=True)

    prune_mean, prune_std = map(np.asarray, zip(*prune))
    axes[0].bar(x, prune_mean, color=BLUE, edgecolor=BLACK, linewidth=0.65)
    axes[0].errorbar(x, prune_mean, yerr=clipped_yerr(prune_mean, prune_std), fmt="none", ecolor=BLACK, capsize=2, linewidth=0.7)
    for index, value in enumerate(prune_mean):
        axes[0].text(index, value + 0.025, f"{value:.1%}", ha="center", fontsize=7.1)
    axes[0].set_ylabel("Physical gate-pruning ratio")
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0].set_ylim(0, 0.60)
    panel_label(axes[0], "(a)")
    paper_axes(axes[0])

    one_mean, one_std = map(np.asarray, zip(*one_shot))
    tuned_mean, tuned_std = map(np.asarray, zip(*fine_tuned))
    width = 0.34
    axes[1].bar(x - width / 2, one_mean, width, color=ORANGE, edgecolor=BLACK, linewidth=0.65, label="One-shot")
    axes[1].bar(x + width / 2, tuned_mean, width, color=BLUE, edgecolor=BLACK, linewidth=0.65, label="Fine-tuned")
    axes[1].errorbar(x - width / 2, one_mean, yerr=clipped_yerr(one_mean, one_std), fmt="none", ecolor=BLACK, capsize=1.8, linewidth=0.65)
    axes[1].errorbar(x + width / 2, tuned_mean, yerr=clipped_yerr(tuned_mean, tuned_std), fmt="none", ecolor=BLACK, capsize=1.8, linewidth=0.65)
    axes[1].set_ylabel(r"Absolute energy deviation $|\Delta E|$")
    axes[1].legend(loc="upper left", frameon=False)
    panel_label(axes[1], "(b)")
    paper_axes(axes[1])

    recovery_mean, recovery_std = map(np.asarray, zip(*recovery))
    axes[2].plot(x, recovery_mean, color=GREEN, marker="o", markersize=3.7, linewidth=1.7)
    axes[2].fill_between(x, np.maximum(0, recovery_mean - recovery_std), np.minimum(1.05, recovery_mean + recovery_std), color=GREEN, alpha=0.16, linewidth=0)
    axes[2].set_ylabel("Recovery fraction")
    axes[2].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[2].set_ylim(0, 1.08)
    panel_label(axes[2], "(c)")
    paper_axes(axes[2])

    for ax in axes:
        ax.set_xticks(x, [f"{delta:g}" for delta in deltas])
        ax.set_xlabel(r"Threshold scale $\delta$")

    save_figure(fig, "fig_true_pruning_finetune")


def plot_fair_baselines(rows: Sequence[Dict]) -> None:
    classification = [row for row in rows if row["task"] == "classification"]
    vqe = [row for row in rows if row["task"] == "vqe"]
    metrics = []
    for method in METHOD_ORDER:
        cls = [row for row in classification if row["baseline_type"] == method]
        energy = [row for row in vqe if row["baseline_type"] == method]
        metrics.append(
            (
                mean_std(cls, "acc_drop", absolute=True),
                mean_std(energy, "dE_max"),
                mean_std(cls, "trace_dist_max"),
            )
        )

    colors = [BLUE, BLUE_LIGHT, GRAY, GRAY_DARK, GRAY, ORANGE]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), constrained_layout=True)
    panels = [
        (0, "Mean absolute accuracy change"),
        (1, r"Mean energy deviation $|\Delta E|$"),
        (2, "Mean maximum trace distance"),
    ]
    x = np.arange(len(METHOD_ORDER))
    for panel_index, (metric_index, ylabel) in enumerate(panels):
        means = np.asarray([row[metric_index][0] for row in metrics])
        stds = np.asarray([row[metric_index][1] for row in metrics])
        axes[panel_index].bar(x, means, color=colors, edgecolor=BLACK, linewidth=0.6)
        axes[panel_index].errorbar(x, means, yerr=clipped_yerr(means, stds), fmt="none", ecolor=BLACK, capsize=1.7, linewidth=0.65)
        axes[panel_index].set_xticks(x, METHOD_LABELS, rotation=34, ha="right")
        axes[panel_index].set_ylabel(ylabel)
        axes[panel_index].set_ylim(bottom=0)
        panel_label(axes[panel_index], f"({chr(ord('a') + panel_index)})")
        paper_axes(axes[panel_index])

    save_figure(fig, "fig_fair_baselines")


def pair_gradient(rows: Sequence[Dict]) -> Dict[float, List[tuple[Dict, Dict]]]:
    paired: Dict[float, List[tuple[Dict, Dict]]] = defaultdict(list)
    for delta in sorted({float(row["delta"]) for row in rows}):
        seeds = sorted({int(row["seed"]) for row in rows if float(row["delta"]) == delta})
        for seed in seeds:
            subset = [row for row in rows if float(row["delta"]) == delta and int(row["seed"]) == seed]
            qrow = next(row for row in subset if row["method"] == "qiprune_true_identity")
            grow = next(row for row in subset if row["method"] == "gradient_integrated_energy_matched_k")
            paired[delta].append((qrow, grow))
    return paired


def plot_gradient(rows: Sequence[Dict]) -> None:
    paired = pair_gradient(rows)
    deltas = sorted(paired)
    x = np.arange(len(deltas))
    q_mean, q_std, g_mean, g_std = [], [], [], []
    q_wins, g_wins, ties = [], [], []
    for delta in deltas:
        q_values = np.asarray([abs(float(q["energy_gap_finetuned_vs_control"])) for q, _ in paired[delta]])
        g_values = np.asarray([abs(float(g["energy_gap_finetuned_vs_control"])) for _, g in paired[delta]])
        q_mean.append(float(np.mean(q_values)))
        g_mean.append(float(np.mean(g_values)))
        q_std.append(float(np.std(q_values, ddof=1)))
        g_std.append(float(np.std(g_values, ddof=1)))
        q_energy = np.asarray([float(q["energy_pruned_finetuned"]) for q, _ in paired[delta]])
        g_energy = np.asarray([float(g["energy_pruned_finetuned"]) for _, g in paired[delta]])
        ties_at_delta = np.abs(q_energy - g_energy) < 1e-10
        q_win = int(np.sum((q_energy < g_energy) & ~ties_at_delta))
        g_win = int(np.sum((g_energy < q_energy) & ~ties_at_delta))
        q_wins.append(q_win)
        g_wins.append(g_win)
        ties.append(int(np.sum(ties_at_delta)))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), constrained_layout=True)
    axes[0].errorbar(x, q_mean, yerr=clipped_yerr(q_mean, q_std), color=BLUE, marker="o", markersize=3.4, linewidth=1.6, capsize=2, label="q-iPrune")
    axes[0].errorbar(x, g_mean, yerr=clipped_yerr(g_mean, g_std), color=ORANGE, marker="s", markersize=3.4, linewidth=1.6, capsize=2, label="Real gradient")
    axes[0].set_ylabel("Absolute energy gap to full control")
    axes[0].set_xlabel(r"Threshold scale $\delta$")
    axes[0].set_xticks(x, [f"{delta:g}" for delta in deltas])
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left", frameon=False)
    panel_label(axes[0], "(a)")
    paper_axes(axes[0])

    width = 0.24
    axes[1].bar(x - width, q_wins, width, color=BLUE, edgecolor=BLACK, linewidth=0.6, label="q-iPrune")
    axes[1].bar(x, g_wins, width, color=ORANGE, edgecolor=BLACK, linewidth=0.6, label="Gradient")
    axes[1].bar(x + width, ties, width, color=GRAY, edgecolor=BLACK, linewidth=0.6, label="Tie")
    axes[1].set_ylabel("Number of seeds")
    axes[1].set_xlabel(r"Threshold scale $\delta$")
    axes[1].set_xticks(x, [f"{delta:g}" for delta in deltas])
    axes[1].set_ylim(0, 5.6)
    axes[1].set_yticks(range(0, 6))
    axes[1].legend(loc="upper left", frameon=False, ncol=3)
    axes[1].text(
        0.98,
        0.78,
        f"Total: {sum(q_wins)} / {sum(g_wins)} / {sum(ties)}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
    )
    panel_label(axes[1], "(b)")
    paper_axes(axes[1])

    save_figure(fig, "fig_qiprune_vs_gradient")


def grouped_candidate_bars(
    ax: plt.Axes,
    rows: Sequence[Dict],
    sigmas: Sequence[float],
    deltas: Sequence[float],
    metric: str,
    absolute: bool = False,
) -> None:
    x = np.arange(len(sigmas))
    width = 0.34
    colors = [BLUE, ORANGE]
    for index, delta in enumerate(deltas):
        heights = []
        for sigma in sigmas:
            row = next(
                item
                for item in rows
                if np.isclose(float(item["delta"]), delta)
                and np.isclose(float(item["rot_pool_sigma"]), sigma)
            )
            value = float(row[metric])
            heights.append(abs(value) if absolute else value)
        offset = (index - 0.5) * width
        ax.bar(x + offset, heights, width, color=colors[index], edgecolor=BLACK, linewidth=0.55, label=fr"$\delta={delta:g}$")
    ax.set_xticks(x, [f"{sigma:g}" for sigma in sigmas])
    paper_axes(ax)


def plot_candidate_classification(rows: Sequence[Dict]) -> None:
    datasets = ["bas", "fashion_sb", "mnist49"]
    titles = ["BAS", "Fashion-SB", "MNIST 4 vs 9"]
    sigmas = sorted({float(row["rot_pool_sigma"]) for row in rows})
    deltas = sorted({float(row["delta"]) for row in rows})
    metrics = [
        ("replace_ratio_over_candidates", "Candidate replacement ratio", False),
        ("trace_dist_max", "Maximum trace distance", False),
        ("acc_drop", "Absolute accuracy change", True),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(7.16, 5.15), sharex=True, constrained_layout=True)
    for column, (dataset, title) in enumerate(zip(datasets, titles)):
        subset = [row for row in rows if row["dataset"] == dataset]
        for row_index, (metric, ylabel, absolute) in enumerate(metrics):
            ax = axes[row_index, column]
            grouped_candidate_bars(ax, subset, sigmas, deltas, metric, absolute=absolute)
            if row_index == 0:
                ax.set_title(title)
                ax.set_ylim(0, 0.68)
                ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
            elif row_index == 1:
                ax.set_ylim(0, 0.058)
            else:
                ax.set_ylim(0, 0.0155)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == 2:
                ax.set_xlabel(r"Pool spread $\sigma$")

    axes[0, 0].text(-0.17, 0.615, r"$\delta=.01$", ha="center", va="bottom", fontsize=6.8)
    axes[0, 0].text(0.17, 0.650, r"$\delta=.02$", ha="center", va="bottom", fontsize=6.8)
    save_figure(fig, "fig_candidate_pool_classification")


def plot_candidate_vqe(rows: Sequence[Dict]) -> None:
    sigmas = sorted({float(row["rot_pool_sigma"]) for row in rows})
    deltas = sorted({float(row["delta"]) for row in rows})
    metrics = [
        ("replace_ratio_over_candidates", "Candidate replacement ratio"),
        ("L", "Redundant candidates $L$"),
        ("dE_mean", r"Mean energy deviation $|\Delta E|$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), constrained_layout=True)
    for index, (metric, ylabel) in enumerate(metrics):
        grouped_candidate_bars(axes[index], rows, sigmas, deltas, metric)
        axes[index].set_ylabel(ylabel)
        axes[index].set_xlabel(r"Pool spread $\sigma$")
        panel_label(axes[index], f"({chr(ord('a') + index)})")
        if metric == "replace_ratio_over_candidates":
            axes[index].set_ylim(0, 0.68)
            axes[index].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0].legend(loc="upper right", frameon=False)
    save_figure(fig, "fig_candidate_pool_vqe")


def tex_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_true_table(rows: Sequence[Dict]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Physical gate deletion and post-pruning fine-tuning on TFIM VQE over five seeds. Values are mean $\pm$ one standard deviation.}",
        r"\label{tab:true_pruning_finetune}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$\delta$ & Physical pruning (\%) & One-shot $|\Delta E|$ & Fine-tuned $|\Delta E|$ & Recovery (\%) \\",
        r"\midrule",
    ]
    for delta in sorted({float(row["delta"]) for row in rows}):
        group = [row for row in rows if float(row["delta"]) == delta]
        pruning = mean_std(group, "actual_gate_prune_ratio")
        one = mean_std(group, "abs_dE_one_shot_vs_original")
        tuned = mean_std(group, "abs_dE_finetuned_vs_original")
        recovery = mean_std(group, "recovery_fraction_vs_control")
        lines.append(
            f"{delta:g} & {100 * pruning[0]:.1f} $\\pm$ {100 * pruning[1]:.1f} & "
            f"{tex_pm(*one)} & {tex_pm(*tuned)} & "
            f"{100 * recovery[0]:.1f} $\\pm$ {100 * recovery[1]:.1f} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (OUTPUT_DIR / "tab_true_pruning_finetune.tex").write_text("\n".join(lines), encoding="utf-8")


def fair_summary(rows: Sequence[Dict]) -> List[Dict]:
    classification = [row for row in rows if row["task"] == "classification"]
    vqe = [row for row in rows if row["task"] == "vqe"]
    result = []
    for method, label in zip(METHOD_ORDER, METHOD_LABELS):
        cls = [row for row in classification if row["baseline_type"] == method]
        energy = [row for row in vqe if row["baseline_type"] == method]
        result.append(
            {
                "method": method,
                "label": label,
                "accuracy": mean_std(cls, "acc_drop", absolute=True),
                "energy": mean_std(energy, "dE_max"),
                "trace": mean_std(cls, "trace_dist_max"),
            }
        )
    return result


def write_fair_table(rows: Sequence[Dict]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched-budget comparison over five seeds. Lower is better; values are mean $\pm$ one standard deviation.}",
        r"\label{tab:fair_baselines}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Classification $|\Delta\mathrm{Acc}|$ & TFIM VQE $|\Delta E|$ & Maximum trace distance \\",
        r"\midrule",
    ]
    for item in fair_summary(rows):
        label = item["label"]
        if item["method"] == "qiprune_positionaware":
            label = r"\textbf{q-iPrune}"
        lines.append(
            f"{label} & {tex_pm(*item['accuracy'], digits=5)} & "
            f"{tex_pm(*item['energy'], digits=5)} & {tex_pm(*item['trace'], digits=5)} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (OUTPUT_DIR / "tab_fair_baselines.tex").write_text("\n".join(lines), encoding="utf-8")


def gradient_summary(rows: Sequence[Dict]) -> List[Dict]:
    paired = pair_gradient(rows)
    result = []
    for delta in sorted(paired):
        q_values = np.asarray([abs(float(q["energy_gap_finetuned_vs_control"])) for q, _ in paired[delta]])
        g_values = np.asarray([abs(float(g["energy_gap_finetuned_vs_control"])) for _, g in paired[delta]])
        q_energy = np.asarray([float(q["energy_pruned_finetuned"]) for q, _ in paired[delta]])
        g_energy = np.asarray([float(g["energy_pruned_finetuned"]) for _, g in paired[delta]])
        ties_at_delta = np.abs(q_energy - g_energy) < 1e-10
        q_win = int(np.sum((q_energy < g_energy) & ~ties_at_delta))
        g_win = int(np.sum((g_energy < q_energy) & ~ties_at_delta))
        result.append(
            {
                "delta": delta,
                "q": (float(np.mean(q_values)), float(np.std(q_values, ddof=1))),
                "g": (float(np.mean(g_values)), float(np.std(g_values, ddof=1))),
                "q_win": q_win,
                "g_win": g_win,
                "tie": int(np.sum(ties_at_delta)),
            }
        )
    return result


def write_gradient_table(rows: Sequence[Dict]) -> None:
    summary = gradient_summary(rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched-$k$ comparison with real-gradient deletion over five seeds. Values are mean $\pm$ one standard deviation; W/L/T is determined by the lower final VQE energy.}",
        r"\label{tab:qiprune_gradient}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$\delta$ & q-iPrune & Gradient & W/L/T \\",
        r"\midrule",
    ]
    for item in summary:
        lines.append(
            f"{item['delta']:g} & {tex_pm(*item['q'])} & {tex_pm(*item['g'])} & "
            f"{item['q_win']}/{item['g_win']}/{item['tie']} \\\\" 
        )
    total_q = sum(item["q_win"] for item in summary)
    total_g = sum(item["g_win"] for item in summary)
    total_tie = sum(item["tie"] for item in summary)
    lines.extend(
        [
            r"\midrule",
            f"Total & -- & -- & {total_q}/{total_g}/{total_tie} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (OUTPUT_DIR / "tab_qiprune_vs_gradient.tex").write_text("\n".join(lines), encoding="utf-8")


def write_candidate_table(rows: Sequence[Dict]) -> None:
    order = ["mnist49", "fashion_sb", "bas"]
    labels = {"bas": "BAS", "fashion_sb": "Fashion-SB", "mnist49": "MNIST49"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Candidate-pool results on three classification datasets (seed 42). Acc$_\mathrm{drop}=$Acc$_\mathrm{base}-$Acc$_\mathrm{pruned}$; negative values indicate improved accuracy after candidate replacement. Candidate replacement measures proposal-pool compression rather than physical gate deletion.}",
        r"\label{tab:candidate_pool_summary}",
        r"\begin{tabular}{l|l|lcccccc}",
        r"\hline",
        r"Dataset & $\delta$ & $\sigma$ & Acc$_\mathrm{base}$ (\%) & Acc$_\mathrm{pruned}$ (\%) & Acc$_\mathrm{drop}$ (p.p.) & Candidate Replace (\%) & RHS (raw$\rightarrow$clipped) & $d_{q,\max}^{(\mathrm{repl.})}$ \\",
        r"\hline",
    ]
    for dataset in order:
        group = sorted(
            [row for row in rows if row["task"] == "classification" and row["dataset"] == dataset],
            key=lambda row: (float(row["delta"]), float(row["rot_pool_sigma"])),
        )
        deltas = sorted({float(row["delta"]) for row in group})
        dataset_rows = len(group)
        first_dataset_row = True
        for delta_index, delta in enumerate(deltas):
            delta_rows = [row for row in group if float(row["delta"]) == delta]
            first_delta_row = True
            for row in delta_rows:
                dataset_cell = (
                    rf"\multirow{{{dataset_rows}}}{{*}}{{\rotatebox{{90}}{{{labels[dataset]}}}}}"
                    if first_dataset_row
                    else ""
                )
                delta_cell = (
                    rf"\multirow{{{len(delta_rows)}}}{{*}}{{\centering\rotatebox{{90}}{{$\delta={delta:.2f}$}}}}"
                    if first_delta_row
                    else ""
                )
                sigma = float(row["rot_pool_sigma"])
                lines.append(
                    f"{dataset_cell} & {delta_cell} & {sigma:g} & "
                    f"{100 * float(row['acc_base']):.2f} & {100 * float(row['acc_prun']):.2f} & "
                    f"{100 * float(row['acc_drop']):.2f} & "
                    f"{100 * float(row['replace_ratio_over_candidates']):.2f} & "
                    f"{float(row['rhs_raw']):.2f}{{$\\rightarrow$}}{float(row['rhs_clipped']):.2f} & "
                    f"{float(row['dq_max_over_replaced']):.4f} \\\\" 
                )
                first_dataset_row = False
                first_delta_row = False
            if delta_index < len(deltas) - 1:
                lines.append(r"\cline{2-9}")
        lines.append(r"\hline")
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    (OUTPUT_DIR / "tab_candidate_pool_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def write_result_text(true_rows: Sequence[Dict], fair_rows: Sequence[Dict], gradient_rows: Sequence[Dict]) -> None:
    true_groups = {
        delta: [row for row in true_rows if float(row["delta"]) == delta]
        for delta in sorted({float(row["delta"]) for row in true_rows})
    }
    recovery_effective = [mean_std(true_groups[delta], "recovery_fraction_vs_control")[0] for delta in sorted(true_groups) if delta <= 1.5]
    pruning_effective = [mean_std(true_groups[delta], "actual_gate_prune_ratio")[0] for delta in sorted(true_groups) if delta <= 1.5]

    fair = {item["method"]: item for item in fair_summary(fair_rows)}
    q = fair["qiprune_positionaware"]
    random = fair["random_reference_matched_selected_L"]
    cls_improvement = 1 - q["accuracy"][0] / random["accuracy"][0]
    vqe_improvement = 1 - q["energy"][0] / random["energy"][0]
    trace_improvement = 1 - q["trace"][0] / random["trace"][0]
    gradient = gradient_summary(gradient_rows)
    q_wins = sum(item["q_win"] for item in gradient)
    g_wins = sum(item["g_win"] for item in gradient)
    ties = sum(item["tie"] for item in gradient)

    text = rf"""\subsection{{Strengthened empirical evaluation}}
\label{{sec:strengthened_evaluation}}

\paragraph{{Physical gate deletion and recovery.}}
Figure~\ref{{fig:true_pruning_finetune}} and Table~\ref{{tab:true_pruning_finetune}} report physical deletion results on TFIM VQE over five seeds. In the effective range $\delta\in[0.8,1.5]$, q-iPrune removes {100 * min(pruning_effective):.1f}\%--{100 * max(pruning_effective):.1f}\% of the circuit gates, while post-pruning fine-tuning recovers {100 * min(recovery_effective):.1f}\%--{100 * max(recovery_effective):.1f}\% of the control improvement. At $\delta=2.0$, all trainable rotation gates are removed, yielding a deliberate failure-boundary case rather than a useful operating point.

\paragraph{{Matched-budget baselines.}}
Figure~\ref{{fig:fair_baselines}} and Table~\ref{{tab:fair_baselines}} compare six selection rules under the same forward replacement budget. Relative to random selection, q-iPrune reduces the mean absolute classification accuracy change by {100 * cls_improvement:.1f}\%, the TFIM VQE energy deviation by {100 * vqe_improvement:.1f}\%, and the maximum output-state trace distance by {100 * trace_improvement:.1f}\%. Performance is comparable to the strongest heuristic baselines, so the evidence supports consistent improvement over random pruning rather than universal dominance over every heuristic.

\paragraph{{Comparison with real-gradient deletion.}}
Figure~\ref{{fig:qiprune_gradient}} and Table~\ref{{tab:qiprune_gradient}} compare q-iPrune with integrated real-gradient deletion at matched $k$. Using the lower final VQE energy to define a win, q-iPrune records {q_wins} wins, the gradient baseline {g_wins} wins, and {ties} ties across 30 paired configurations. The five ties at $\delta=2.0$ arise because both methods reach the same all-rotation-deleted boundary.

\paragraph{{Candidate-pool protocol check.}}
Figures~\ref{{fig:candidate_pool_classification}} and~\ref{{fig:candidate_pool_vqe}}, together with Table~\ref{{tab:candidate_pool_summary}}, retain the original 32-configuration candidate-pool sweep as a protocol check. Increasing $\sigma$ generally reduces candidate redundancy, whereas the wider $\delta=0.02$ threshold preserves a larger candidate replacement ratio. All replaced candidates satisfy the prescribed $q$-distance threshold. Here $L$ counts redundant candidate entries and is not interpreted as physical circuit compression.
"""
    (OUTPUT_DIR / "results_text.tex").write_text(text, encoding="utf-8")


def write_include_file() -> None:
    text = r"""% Auto-generated from completed formal checkpoints.
% Insert with: \input{generated_results_20260806/paper_ready_experiments}
\input{generated_results_20260806/results_text}

\input{generated_results_20260806/tab_true_pruning_finetune}
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{generated_results_20260806/fig_true_pruning_finetune.pdf}
  \caption{Physical gate deletion and post-pruning fine-tuning on TFIM VQE over five seeds. (a) Realized physical gate-pruning ratio. (b) Absolute energy deviation before and after fine-tuning. (c) Recovery fraction relative to the full-control optimization. Error bars and bands denote one standard deviation.}
  \label{fig:true_pruning_finetune}
\end{figure*}

\input{generated_results_20260806/tab_fair_baselines}
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{generated_results_20260806/fig_fair_baselines.pdf}
  \caption{Matched-budget comparison with five baselines over five seeds. Lower values are better. q-iPrune consistently improves over random selection in classification accuracy change, TFIM VQE energy deviation, and output-state drift, while remaining comparable to the strongest heuristic baselines. Error bars denote one standard deviation.}
  \label{fig:fair_baselines}
\end{figure*}

\input{generated_results_20260806/tab_qiprune_vs_gradient}
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{generated_results_20260806/fig_qiprune_vs_gradient.pdf}
  \caption{Matched-$k$ comparison between q-iPrune and integrated real-gradient gate deletion on TFIM VQE over five seeds. (a) Mean absolute energy gap to the full-control optimization. (b) Per-threshold win, loss, and tie counts, where a win is defined by the lower final VQE energy. Error bars denote one standard deviation.}
  \label{fig:qiprune_gradient}
\end{figure*}

\input{generated_results_20260806/tab_candidate_pool_summary}
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{generated_results_20260806/fig_candidate_pool_classification.pdf}
  \caption{Original candidate-pool sweep on BAS, Fashion-SB, and MNIST 4 vs 9 (seed 42). The panels report candidate replacement ratio, maximum trace distance, and absolute accuracy change.}
  \label{fig:candidate_pool_classification}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{generated_results_20260806/fig_candidate_pool_vqe.pdf}
  \caption{Original candidate-pool sensitivity sweep on TFIM VQE (seed 42). Here $L$ denotes redundant candidates in the sampled pool and not physically deleted circuit gates.}
  \label{fig:candidate_pool_vqe}
\end{figure*}
"""
    (OUTPUT_DIR / "paper_ready_experiments.tex").write_text(text, encoding="utf-8")


def write_preview() -> None:
    text = r"""\documentclass[twocolumn,letterpaper]{article}
\usepackage[margin=0.72in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{microtype}
\usepackage{caption}
\title{q-iPrune: Paper-ready Experimental Results Preview}
\author{}
\date{}
\begin{document}
\maketitle
\input{generated_results_20260806/paper_ready_experiments}
\end{document}
"""
    (PAPER_DIR / "paper_ready_results_preview.tex").write_text(text, encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "source_rows": {
            "candidate_pool": 32,
            "fair_baseline_method_rows": 960,
            "true_pruning": 30,
            "gradient_method_rows": 60,
        },
        "sample_sizes": {
            "candidate_pool": "seed 42",
            "fair_baselines": "seeds 42-46",
            "true_pruning": "seeds 42-46",
            "gradient_comparison": "seeds 42-46",
        },
        "uncertainty": "one standard deviation",
        "figures": [
            "fig_true_pruning_finetune",
            "fig_fair_baselines",
            "fig_qiprune_vs_gradient",
            "fig_candidate_pool_classification",
            "fig_candidate_pool_vqe",
        ],
        "tables": [
            "tab_true_pruning_finetune",
            "tab_fair_baselines",
            "tab_qiprune_vs_gradient",
            "tab_candidate_pool_summary",
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    configure_style()
    candidate_rows = load_rows(CANDIDATE_PATH)
    true_rows = load_rows(TRUE_PATH)
    gradient_rows = load_rows(GRADIENT_PATH)
    fair_rows: List[Dict] = []
    for path in FAIR_PATHS:
        fair_rows.extend(load_rows(path))

    if (len(candidate_rows), len(true_rows), len(gradient_rows), len(fair_rows)) != (32, 30, 60, 960):
        raise RuntimeError(
            "Unexpected formal row counts: "
            f"candidate={len(candidate_rows)}, true={len(true_rows)}, "
            f"gradient={len(gradient_rows)}, fair={len(fair_rows)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_true_pruning(true_rows)
    plot_fair_baselines(fair_rows)
    plot_gradient(gradient_rows)
    plot_candidate_classification([row for row in candidate_rows if row["task"] == "classification"])
    plot_candidate_vqe([row for row in candidate_rows if row["task"] == "vqe"])
    write_true_table(true_rows)
    write_fair_table(fair_rows)
    write_gradient_table(gradient_rows)
    write_candidate_table(candidate_rows)
    write_result_text(true_rows, fair_rows, gradient_rows)
    write_include_file()
    write_preview()
    write_manifest()
    print(f"Paper-ready assets saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
