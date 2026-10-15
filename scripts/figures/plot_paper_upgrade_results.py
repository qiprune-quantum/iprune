from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_DIR.parent
    / "outputs"
    / "019fcf7c-3f68-7763-89e2-cc708014fdb8"
    / "paper_figures_20260806"
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
CANDIDATE_PATH = (
    PROJECT_DIR
    / "results"
    / "original_candidate_20260805"
    / "candidate_sweep_checkpoint.json"
)


BLUE = "#5B9BD5"
BLUE_DARK = "#2F5597"
BLUE_LIGHT = "#9DC3E6"
ORANGE = "#F9B23F"
GRAY = "#A5A5A5"
GRAY_DARK = "#666666"
GREEN = "#70AD47"
BLACK = "#111111"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["results"])


def values(rows: Sequence[Dict], key: str) -> np.ndarray:
    result = [float(row[key]) for row in rows if row.get(key) is not None]
    return np.asarray(result, dtype=np.float64)


def mean_std(rows: Sequence[Dict], key: str) -> tuple[float, float]:
    sample = values(rows, key)
    if len(sample) == 0:
        return np.nan, np.nan
    return float(np.mean(sample)), float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def paper_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, linestyle="--", linewidth=0.7, color="#D9D9D9", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_true_pruning(rows: Sequence[Dict]) -> None:
    deltas = sorted({float(row["delta"]) for row in rows})
    groups = [[row for row in rows if float(row["delta"]) == delta] for delta in deltas]
    x = np.arange(len(deltas))

    prune_mean, prune_std = zip(*(mean_std(group, "actual_gate_prune_ratio") for group in groups))
    one_mean, one_std = zip(*(mean_std(group, "abs_dE_one_shot_vs_original") for group in groups))
    tuned_mean, tuned_std = zip(*(mean_std(group, "abs_dE_finetuned_vs_original") for group in groups))
    recovery_mean, recovery_std = zip(*(mean_std(group, "recovery_fraction_vs_control") for group in groups))

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), constrained_layout=True)

    axes[0].bar(x, prune_mean, width=0.62, color=BLUE, edgecolor=BLACK, linewidth=0.9)
    axes[0].errorbar(x, prune_mean, yerr=prune_std, fmt="none", ecolor=BLACK, capsize=3, linewidth=1)
    axes[0].set_ylabel("Physical Gate\nPruning Ratio")
    axes[0].set_ylim(0, 0.58)
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0].set_title("TRUE GATE DELETION ON TFIM VQE")
    for index, value in enumerate(prune_mean):
        axes[0].text(index, value + 0.022, f"{value:.1%}", ha="center", fontsize=9)
    paper_grid(axes[0])
    add_panel_label(axes[0], "(a)")

    width = 0.34
    axes[1].bar(x - width / 2, one_mean, width=width, color=ORANGE, edgecolor=BLACK, linewidth=0.9, label="One-shot")
    axes[1].bar(x + width / 2, tuned_mean, width=width, color=BLUE, edgecolor=BLACK, linewidth=0.9, label="Post-fine-tuned")
    axes[1].errorbar(x - width / 2, one_mean, yerr=one_std, fmt="none", ecolor=BLACK, capsize=2.5, linewidth=0.9)
    axes[1].errorbar(x + width / 2, tuned_mean, yerr=tuned_std, fmt="none", ecolor=BLACK, capsize=2.5, linewidth=0.9)
    axes[1].set_ylabel("Absolute Energy\nDeviation |ΔE|")
    axes[1].legend(loc="upper left", frameon=True, edgecolor="#777777")
    paper_grid(axes[1])
    add_panel_label(axes[1], "(b)")

    axes[2].plot(x, recovery_mean, marker="o", markersize=6, linewidth=2.2, color=GREEN)
    axes[2].fill_between(
        x,
        np.maximum(0.0, np.asarray(recovery_mean) - np.nan_to_num(recovery_std)),
        np.minimum(1.05, np.asarray(recovery_mean) + np.nan_to_num(recovery_std)),
        color=GREEN,
        alpha=0.18,
        linewidth=0,
    )
    axes[2].set_ylabel("Recovery Fraction")
    axes[2].set_xlabel("δ (Pruning Threshold Scale)")
    axes[2].set_ylim(0, 1.08)
    axes[2].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    paper_grid(axes[2])
    add_panel_label(axes[2], "(c)")

    for ax in axes:
        ax.set_xticks(x, [f"{delta:g}" for delta in deltas])

    save_figure(fig, "fig_true_pruning_vqe_finetune")


def plot_fair_baselines(rows: Sequence[Dict]) -> None:
    order = [
        "qiprune_positionaware",
        "q1_positionaware_matched_selected_L",
        "qiprune_no_prefix_matched_selected_L",
        "unitary_overlap_matched_selected_L",
        "parameter_euclidean_matched_selected_L",
        "random_reference_matched_selected_L",
    ]
    labels = ["q-iPrune", "q=1", "No prefix", "Unitary", "Parameter", "Random"]
    palette = [BLUE_DARK, BLUE, BLUE_LIGHT, GRAY_DARK, GRAY, ORANGE]
    classification = [row for row in rows if row["task"] == "classification"]
    vqe = [row for row in rows if row["task"] == "vqe"]

    cls_mean, cls_std, vqe_mean, vqe_std, trace_mean = [], [], [], [], []
    for method in order:
        cls_group = [row for row in classification if row["baseline_type"] == method]
        vqe_group = [row for row in vqe if row["baseline_type"] == method]
        cls_abs = np.abs(values(cls_group, "acc_drop"))
        cls_mean.append(float(np.mean(cls_abs)))
        cls_std.append(float(np.std(cls_abs, ddof=1)))
        m, s = mean_std(vqe_group, "dE_max")
        vqe_mean.append(m)
        vqe_std.append(s)
        trace_mean.append(float(np.mean(values(cls_group, "trace_dist_max"))))

    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), constrained_layout=True)

    panels = [
        (axes[0], cls_mean, cls_std, "Mean |Accuracy Change|", "CLASSIFICATION"),
        (axes[1], vqe_mean, vqe_std, "Mean Energy Deviation |ΔE|", "TFIM VQE"),
        (axes[2], trace_mean, None, "Mean Maximum Trace Distance", "OUTPUT DRIFT"),
    ]
    for panel_index, (ax, means, stds, ylabel, title) in enumerate(panels):
        ax.bar(x, means, color=palette, edgecolor=BLACK, linewidth=0.8)
        if stds is not None:
            ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=BLACK, capsize=2.5, linewidth=0.9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        paper_grid(ax)
        add_panel_label(ax, f"({chr(ord('a') + panel_index)})")

    save_figure(fig, "fig_fair_baseline_comparison")


def plot_gradient_comparison(rows: Sequence[Dict]) -> None:
    deltas = sorted({float(row["delta"]) for row in rows})
    paired: Dict[float, List[tuple[Dict, Dict]]] = defaultdict(list)
    for delta in deltas:
        seeds = sorted({int(row["seed"]) for row in rows if float(row["delta"]) == delta})
        for seed in seeds:
            sample = [row for row in rows if float(row["delta"]) == delta and int(row["seed"]) == seed]
            qrow = next(row for row in sample if row["method"] == "qiprune_true_identity")
            grow = next(row for row in sample if row["method"] == "gradient_integrated_energy_matched_k")
            paired[delta].append((qrow, grow))

    q_mean, q_std, g_mean, g_std = [], [], [], []
    q_wins, g_wins, ties, overlap_mean = [], [], [], []
    for delta in deltas:
        pairs = paired[delta]
        q_values = np.asarray([abs(float(q["energy_gap_finetuned_vs_control"])) for q, _ in pairs])
        g_values = np.asarray([abs(float(g["energy_gap_finetuned_vs_control"])) for _, g in pairs])
        q_mean.append(float(np.mean(q_values)))
        g_mean.append(float(np.mean(g_values)))
        q_std.append(float(np.std(q_values, ddof=1)))
        g_std.append(float(np.std(g_values, ddof=1)))
        q_win = g_win = tie = 0
        overlaps = []
        for qrow, grow in pairs:
            q_energy = float(qrow["energy_pruned_finetuned"])
            g_energy = float(grow["energy_pruned_finetuned"])
            if abs(q_energy - g_energy) < 1e-10:
                tie += 1
            elif q_energy < g_energy:
                q_win += 1
            else:
                g_win += 1
            q_ids = set(qrow.get("deleted_pos_ids", []))
            g_ids = set(grow.get("deleted_pos_ids", []))
            overlaps.append(len(q_ids & g_ids) / max(1, len(q_ids)))
        q_wins.append(q_win)
        g_wins.append(g_win)
        ties.append(tie)
        overlap_mean.append(float(np.mean(overlaps)))

    x = np.arange(len(deltas))
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.5), constrained_layout=True)

    axes[0].errorbar(x, q_mean, yerr=q_std, marker="o", markersize=5, linewidth=2.1, capsize=3, color=BLUE, label="q-iPrune")
    axes[0].errorbar(x, g_mean, yerr=g_std, marker="s", markersize=5, linewidth=2.1, capsize=3, color=ORANGE, label="Real gradient")
    axes[0].set_title("MATCHED-K TFIM VQE COMPARISON")
    axes[0].set_ylabel("Absolute Energy Gap\nto Full Control")
    axes[0].legend(loc="upper left", frameon=True, edgecolor="#777777")
    paper_grid(axes[0])
    add_panel_label(axes[0], "(a)")

    width = 0.24
    axes[1].bar(x - width, q_wins, width=width, color=BLUE, edgecolor=BLACK, linewidth=0.8, label="q-iPrune wins")
    axes[1].bar(x, g_wins, width=width, color=ORANGE, edgecolor=BLACK, linewidth=0.8, label="Gradient wins")
    axes[1].bar(x + width, ties, width=width, color=GRAY, edgecolor=BLACK, linewidth=0.8, label="Ties")
    axes[1].set_ylabel("Number of Seeds")
    axes[1].set_xlabel("δ (Pruning Threshold Scale)")
    axes[1].set_ylim(0, 5.7)
    axes[1].set_yticks(range(0, 6))
    axes[1].legend(loc="upper left", ncol=3, frameon=True, edgecolor="#777777")
    paper_grid(axes[1])
    add_panel_label(axes[1], "(b)")

    for ax in axes:
        ax.set_xticks(x, [f"{delta:g}" for delta in deltas])

    save_figure(fig, "fig_qiprune_vs_real_gradient")


def candidate_metric(row: Dict, key: str) -> float:
    value = float(row[key])
    return abs(value) if key == "acc_drop" else value


def grouped_candidate_bars(
    ax: plt.Axes,
    rows: Sequence[Dict],
    sigmas: Sequence[float],
    deltas: Sequence[float],
    metric: str,
    *,
    percent: bool = False,
) -> None:
    x = np.arange(len(sigmas))
    width = 0.34
    colors = [BLUE, ORANGE]
    for delta_index, delta in enumerate(deltas):
        heights = []
        for sigma in sigmas:
            row = next(
                item
                for item in rows
                if np.isclose(float(item["delta"]), delta)
                and np.isclose(float(item["rot_pool_sigma"]), sigma)
            )
            heights.append(candidate_metric(row, metric))
        offset = (delta_index - (len(deltas) - 1) / 2) * width
        ax.bar(
            x + offset,
            heights,
            width=width,
            color=colors[delta_index],
            edgecolor=BLACK,
            linewidth=0.8,
            label=fr"$\delta={delta:g}$",
        )
    ax.set_xticks(x, [f"{sigma:g}" for sigma in sigmas])
    if percent:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    paper_grid(ax)


def plot_candidate_classification(rows: Sequence[Dict]) -> None:
    datasets = ["bas", "fashion_sb", "mnist49"]
    titles = ["BAS", "FASHION-SB", "MNIST 4 VS 9"]
    sigmas = sorted({float(row["rot_pool_sigma"]) for row in rows})
    deltas = sorted({float(row["delta"]) for row in rows})
    metrics = [
        ("replace_ratio_over_candidates", "Candidate Replacement Ratio", True),
        ("trace_dist_max", "Maximum Trace Distance", False),
        ("acc_drop", "Absolute Accuracy Change", False),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12.8, 8.2), sharex=True, constrained_layout=True)
    for column, (dataset, title) in enumerate(zip(datasets, titles)):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for row_index, (metric, ylabel, percent) in enumerate(metrics):
            ax = axes[row_index, column]
            grouped_candidate_bars(
                ax,
                dataset_rows,
                sigmas,
                deltas,
                metric,
                percent=percent,
            )
            if row_index == 0:
                ax.set_title(title)
                ax.set_ylim(0, 0.68)
            elif row_index == 1:
                ax.set_ylim(0, 0.058)
            else:
                ax.set_ylim(0, 0.0155)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == len(metrics) - 1:
                ax.set_xlabel(r"$\sigma$ (Candidate-Pool Spread)")

    axes[0, 0].text(-0.17, 0.615, r"$\delta=.01$", ha="center", va="bottom", fontsize=8)
    axes[0, 0].text(0.17, 0.650, r"$\delta=.02$", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "ORIGINAL CANDIDATE-POOL EXPERIMENT: CLASSIFICATION",
        y=1.04,
        fontsize=17,
        fontweight="bold",
    )
    save_figure(fig, "fig_candidate_pool_classification")


def plot_candidate_vqe(rows: Sequence[Dict]) -> None:
    sigmas = sorted({float(row["rot_pool_sigma"]) for row in rows})
    deltas = sorted({float(row["delta"]) for row in rows})
    metrics = [
        ("replace_ratio_over_candidates", "Candidate Replacement Ratio", True),
        ("L", "Redundant Candidates L", False),
        ("dE_mean", "Mean Energy Deviation |ΔE|", False),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.2), sharex=True, constrained_layout=True)
    for index, (metric, ylabel, percent) in enumerate(metrics):
        grouped_candidate_bars(
            axes[index],
            rows,
            sigmas,
            deltas,
            metric,
            percent=percent,
        )
        axes[index].set_ylabel(ylabel)
        add_panel_label(axes[index], f"({chr(ord('a') + index)})")
        if metric == "replace_ratio_over_candidates":
            axes[index].set_ylim(0, 0.68)
    axes[0].set_title("ORIGINAL CANDIDATE-POOL EXPERIMENT: TFIM VQE")
    axes[0].legend(loc="upper right", frameon=True, edgecolor="#777777")
    axes[-1].set_xlabel(r"$\sigma$ (Candidate-Pool Spread)")
    save_figure(fig, "fig_candidate_pool_vqe")


def write_summary(
    true_rows: Sequence[Dict],
    gradient_rows: Sequence[Dict],
    fair_rows: Sequence[Dict],
    candidate_rows: Sequence[Dict],
) -> None:
    summary = {
        "formal_experiment_suites": 4,
        "independent_configurations": 252,
        "method_result_rows": 1082,
        "true_pruning_rows": len(true_rows),
        "fair_baseline_rows": len(fair_rows),
        "gradient_method_rows": len(gradient_rows),
        "candidate_pool_rows": len(candidate_rows),
        "figures": [
            "fig_true_pruning_vqe_finetune",
            "fig_fair_baseline_comparison",
            "fig_qiprune_vs_real_gradient",
            "fig_candidate_pool_classification",
            "fig_candidate_pool_vqe",
        ],
    }
    (OUTPUT_DIR / "figure_data_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    configure_style()
    true_rows = load_rows(TRUE_PATH)
    gradient_rows = load_rows(GRADIENT_PATH)
    candidate_rows = load_rows(CANDIDATE_PATH)
    fair_rows: List[Dict] = []
    for path in FAIR_PATHS:
        fair_rows.extend(load_rows(path))

    if (
        len(true_rows) != 30
        or len(gradient_rows) != 60
        or len(fair_rows) != 960
        or len(candidate_rows) != 32
    ):
        raise RuntimeError(
            f"Unexpected formal row counts: true={len(true_rows)}, "
            f"gradient={len(gradient_rows)}, fair={len(fair_rows)}, "
            f"candidate={len(candidate_rows)}"
        )

    plot_true_pruning(true_rows)
    plot_fair_baselines(fair_rows)
    plot_gradient_comparison(gradient_rows)
    plot_candidate_classification(
        [row for row in candidate_rows if row["task"] == "classification"]
    )
    plot_candidate_vqe([row for row in candidate_rows if row["task"] == "vqe"])
    write_summary(true_rows, gradient_rows, fair_rows, candidate_rows)
    print(f"Saved paper figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
