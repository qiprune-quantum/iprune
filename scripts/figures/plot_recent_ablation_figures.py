from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from iprune.figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment


PROJECT_DIR = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT_DIR
BLUE = "#2C6DB2"
RED = "#D45B52"
GREEN = "#4D9A57"
ORANGE = "#E49B45"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("seed*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["results"])
    return rows


def mean_ci95(values: list[float]) -> tuple[float, float]:
    sample = np.asarray(values, dtype=np.float64)
    n = len(sample)
    mean = float(np.mean(sample))
    if n <= 1:
        return mean, 0.0
    sem = float(np.std(sample, ddof=1) / np.sqrt(n))
    return mean, float(student_t.ppf(0.975, df=n - 1) * sem)


def seed_mean_gaps(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], bool]) -> list[float]:
    values = []
    for seed in range(3001, 3013):
        group = [r for r in rows if r["seed"] == seed and key(r)]
        values.append(float(np.mean([r["energy_gap_finetuned_vs_control"] for r in group])))
    return values


def seed_mean_metric(
    rows: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], bool],
    metric: str,
) -> list[float]:
    values = []
    for seed in range(3001, 3013):
        group = [r for r in rows if r["seed"] == seed and key(r)]
        values.append(float(np.mean([r[metric] for r in group])))
    return values


def _band_plot(
    ax: plt.Axes,
    x: list[float],
    mean: list[float],
    ci: list[float],
    color: str,
    *,
    label: str | None = None,
    linestyle: str = "-",
) -> None:
    mean_array = np.asarray(mean, dtype=float)
    ci_array = np.asarray(ci, dtype=float)
    ax.fill_between(
        x,
        mean_array - ci_array,
        mean_array + ci_array,
        color=color,
        alpha=0.15,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        x,
        mean_array,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=0.7,
        linewidth=0.9,
        linestyle=linestyle,
        color=color,
        label=label,
        zorder=3,
    )


def haug_style_figure(
    ring: list[dict[str, Any]],
    linear: list[dict[str, Any]],
    all_to_all: list[dict[str, Any]],
    budget: list[dict[str, Any]],
    size: list[dict[str, Any]],
    output_dir: Path,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.07, top=0.95, wspace=0.34, hspace=0.55)
    deltas = [0.8, 1.0, 1.2]
    architectures = [("ring", ring, BLUE), ("linear", linear, ORANGE), ("all-to-all", all_to_all, GREEN)]

    # a: architecture energy gap
    ax = axes[0, 0]
    for label, rows, color in architectures:
        means, cis = [], []
        for delta in deltas:
            m, c = mean_ci95(seed_mean_gaps(rows, lambda r, d=delta: abs(float(r["delta"]) - d) < 1e-9))
            means.append(m)
            cis.append(c)
        _band_plot(ax, deltas, means, cis, color, label=label)
    ax.set_xlabel("Threshold scale, δ")
    ax.set_ylabel("Final gap")
    ax.set_xticks(deltas)
    ax.set_xlim(0.75, 1.25)
    ax.legend(loc="upper left", handlelength=1.3, borderaxespad=0.2)
    _paper_axes(ax)
    _panel_label(ax, "a")

    # b: architecture deleted-rotation fraction
    ax = axes[0, 1]
    for label, rows, color in architectures:
        means, cis = [], []
        for delta in deltas:
            m, c = mean_ci95(
                seed_mean_metric(
                    rows,
                    lambda r, d=delta: abs(float(r["delta"]) - d) < 1e-9,
                    "rotation_prune_ratio",
                )
            )
            means.append(100.0 * m)
            cis.append(100.0 * c)
        _band_plot(ax, deltas, means, cis, color, label=label)
    ax.set_xlabel("Threshold scale, δ")
    ax.set_ylabel("Deleted rotations (%)")
    ax.set_xticks(deltas)
    ax.set_xlim(0.75, 1.25)
    _paper_axes(ax)
    _panel_label(ax, "b")

    # c: path-budget energy gap
    ax = axes[1, 0]
    budgets = [None, 0.5, 1.0, 1.5]
    budget_labels = ["none", "0.5", "1.0", "1.5"]
    positions = list(range(len(budgets)))
    means, cis = [], []
    for budget_val in budgets:
        key = (lambda r: r["path_budget"] is None) if budget_val is None else (
            lambda r, b=budget_val: r["path_budget"] is not None and abs(float(r["path_budget"]) - b) < 1e-9
        )
        m, c = mean_ci95(seed_mean_gaps(budget, key))
        means.append(m)
        cis.append(c)
    _band_plot(ax, positions, means, cis, BLUE)
    ax.axhline(means[0], color="#202020", linestyle="-.", linewidth=0.7, alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(budget_labels)
    ax.set_xlabel("Path budget, B")
    ax.set_ylabel("Final gap")
    _paper_axes(ax)
    _panel_label(ax, "c")

    # d: path-budget deleted rotations
    ax = axes[1, 1]
    means, cis = [], []
    for budget_val in budgets:
        key = (lambda r: r["path_budget"] is None) if budget_val is None else (
            lambda r, b=budget_val: r["path_budget"] is not None and abs(float(r["path_budget"]) - b) < 1e-9
        )
        m, c = mean_ci95(seed_mean_metric(budget, key, "deleted_count"))
        means.append(m)
        cis.append(c)
    _band_plot(ax, positions, means, cis, RED)
    ax.set_xticks(positions)
    ax.set_xticklabels(budget_labels)
    ax.set_xlabel("Path budget, B")
    ax.set_ylabel("Deleted rotations")
    _paper_axes(ax)
    _panel_label(ax, "d")

    # e: calibration-size energy gap
    ax = axes[2, 0]
    sizes = [1, 2, 4, 8]
    log_sizes = [0.0, 1.0, 2.0, 3.0]
    means, cis = [], []
    for m in sizes:
        mean, ci = mean_ci95(seed_mean_gaps(size, lambda r, s=m: int(r["calibration_size"]) == s))
        means.append(mean)
        cis.append(ci)
    _band_plot(ax, log_sizes, means, cis, BLUE)
    ax.axhline(means[0], color="#202020", linestyle="-.", linewidth=0.7, alpha=0.8)
    ax.set_xticks(log_sizes)
    ax.set_xticklabels([str(m) for m in sizes])
    ax.set_xlabel("Calibration size, M")
    ax.set_ylabel("Final gap")
    _paper_axes(ax)
    _panel_label(ax, "e")

    # f: calibration-size violations
    ax = axes[2, 1]
    violations = [
        int(sum(r["physical_threshold_violations"] for r in size if int(r["calibration_size"]) == m))
        for m in sizes
    ]
    ax.plot(log_sizes, violations, "o-", markersize=2.8, markerfacecolor="white",
            markeredgecolor=ORANGE, markeredgewidth=0.7, linewidth=0.9, color=ORANGE)
    ax.set_xticks(log_sizes)
    ax.set_xticklabels([str(m) for m in sizes])
    ax.set_xlabel("Calibration size, M")
    ax.set_ylabel("Path violations")
    _paper_axes(ax)
    _panel_label(ax, "f")

    return _finish(fig, output_dir / "fig_recent_ablations_haug_style")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-17, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        annotation_clip=False,
    )


def _paper_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D7D7D7", linestyle=":", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", pad=1.8)


def _two_panel_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.65), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.92, wspace=0.36)
    return fig, ax1, ax2


def architecture_figure(ring: list[dict[str, Any]], linear: list[dict[str, Any]], all_to_all: list[dict[str, Any]], output_dir: Path) -> plt.Figure:
    fig, ax_gap, ax_viol = _two_panel_figure()
    datasets = [
        ("ring", ring, BLUE),
        ("linear", linear, ORANGE),
        ("all_to_all", all_to_all, GREEN),
    ]
    deltas = [0.8, 1.0, 1.2]
    for label, rows, color in datasets:
        means, cis = [], []
        for delta in deltas:
            m, c = mean_ci95(seed_mean_gaps(rows, lambda r, d=delta: abs(float(r["delta"]) - d) < 1e-9))
            means.append(m)
            cis.append(c)
        ax_gap.errorbar(deltas, means, yerr=cis, fmt="o-", markersize=3.0, markerfacecolor="white",
                        markeredgecolor=color, markeredgewidth=0.8, ecolor=color, elinewidth=0.55,
                        capsize=1.4, linewidth=0.9, label=label, zorder=3)
    ax_gap.set_xlabel("Threshold scale, δ")
    ax_gap.set_ylabel("Final energy gap")
    ax_gap.set_xticks(deltas)
    ax_gap.set_xlim(0.75, 1.25)
    ax_gap.legend(loc="upper left", handlelength=1.4, borderaxespad=0.2)
    _paper_axes(ax_gap)
    _panel_label(ax_gap, "a")
    viol = []
    for _label, rows, _color in datasets:
        total = int(sum(r["physical_threshold_violations"] for r in rows if float(r["delta"]) in deltas))
        viol.append(total)
    ax_viol.bar([0, 1, 2], viol, color=[BLUE, ORANGE, GREEN], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax_viol.set_xticks([0, 1, 2])
    ax_viol.set_xticklabels(["ring", "linear", "all_to_all"])
    ax_viol.set_ylabel("Path threshold violations")
    ax_viol.set_ylim(0, 2)
    _paper_axes(ax_viol)
    _panel_label(ax_viol, "b")
    return _finish(fig, output_dir / "fig_tfim_architecture_compare")


def budget_figure(rows: list[dict[str, Any]], output_dir: Path) -> plt.Figure:
    fig, ax_gap, ax_k = _two_panel_figure()
    budgets = [None, 0.5, 1.0, 1.5]
    labels = ["No budget", "B=0.5", "B=1.0", "B=1.5"]
    gaps, cis, ks = [], [], []
    for budget in budgets:
        key = (lambda r: r["path_budget"] is None) if budget is None else (lambda r, b=budget: r["path_budget"] is not None and abs(float(r["path_budget"]) - b) < 1e-9)
        m, c = mean_ci95(seed_mean_gaps(rows, key))
        gaps.append(m)
        cis.append(c)
        seed_k = []
        for seed in range(3001, 3013):
            g = [r for r in rows if r["seed"] == seed and key(r)]
            seed_k.append(float(np.mean([r["deleted_count"] for r in g])))
        ks.append(float(np.mean(seed_k)))
    x = np.arange(len(budgets))
    ax_gap.errorbar(x, gaps, yerr=cis, fmt="o", markersize=3.4, markerfacecolor="white", markeredgecolor=BLUE,
                    markeredgewidth=0.8, ecolor=BLUE, elinewidth=0.7, capsize=2.0)
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(labels)
    ax_gap.set_ylabel("Final energy gap")
    _paper_axes(ax_gap)
    _panel_label(ax_gap, "a")
    ax_k.bar(x, ks, color=RED, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax_k.set_xticks(x)
    ax_k.set_xticklabels(labels)
    ax_k.set_ylabel("Mean deleted rotations")
    _paper_axes(ax_k)
    _panel_label(ax_k, "b")
    return _finish(fig, output_dir / "fig_tfim_path_budget")


def size_figure(rows: list[dict[str, Any]], output_dir: Path) -> plt.Figure:
    fig, ax_gap, ax_viol = _two_panel_figure()
    sizes = [1, 2, 4, 8]
    gaps, cis = [], []
    viol = []
    for size in sizes:
        m, c = mean_ci95(seed_mean_gaps(rows, lambda r, s=size: int(r["calibration_size"]) == s))
        gaps.append(m)
        cis.append(c)
        viol.append(int(sum(r["physical_threshold_violations"] for r in rows if int(r["calibration_size"]) == size)))
    ax_gap.errorbar(sizes, gaps, yerr=cis, fmt="o-", markersize=3.0, markerfacecolor="white", markeredgecolor=BLUE,
                    markeredgewidth=0.8, ecolor=BLUE, elinewidth=0.55, capsize=1.4, linewidth=0.9)
    ax_gap.set_xlabel("Calibration size, M")
    ax_gap.set_ylabel("Final energy gap")
    ax_gap.set_xticks(sizes)
    _paper_axes(ax_gap)
    _panel_label(ax_gap, "a")
    ax_viol.bar([0, 1, 2, 3], viol, color=ORANGE, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax_viol.set_xticks([0, 1, 2, 3])
    ax_viol.set_xticklabels([str(s) for s in sizes])
    ax_viol.set_xlabel("Calibration size, M")
    ax_viol.set_ylabel("Path threshold violations")
    _paper_axes(ax_viol)
    _panel_label(ax_viol, "b")
    return _finish(fig, output_dir / "fig_tfim_calibration_size")


def _finish(fig: plt.Figure, prefix: Path) -> plt.Figure:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=prefix.with_suffix(".alignment.json"),
        overlay_svg=prefix.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        fig.savefig(prefix.with_suffix(suffix), facecolor="white", **kwargs)
    return fig


def combined_figure(
    ring: list[dict[str, Any]],
    linear: list[dict[str, Any]],
    all_to_all: list[dict[str, Any]],
    budget: list[dict[str, Any]],
    size: list[dict[str, Any]],
    output_dir: Path,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.35), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.11, top=0.94, wspace=0.38, hspace=0.48)
    deltas = [0.8, 1.0, 1.2]

    # a: architecture gap
    ax = axes[0, 0]
    for label, rows, color in (("ring", ring, BLUE), ("linear", linear, ORANGE), ("all-to-all", all_to_all, GREEN)):
        means, cis = [], []
        for delta in deltas:
            m, c = mean_ci95(seed_mean_gaps(rows, lambda r, d=delta: abs(float(r["delta"]) - d) < 1e-9))
            means.append(m)
            cis.append(c)
        ax.errorbar(deltas, means, yerr=cis, fmt="o-", markersize=2.8, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=0.7, ecolor=color, elinewidth=0.5,
                    capsize=1.2, linewidth=0.8, label=label, zorder=3)
    ax.set_xlabel("Threshold, δ")
    ax.set_ylabel("Final gap")
    ax.set_xticks(deltas)
    ax.set_xlim(0.75, 1.25)
    ax.legend(loc="upper left", handlelength=1.2, borderaxespad=0.15)
    _paper_axes(ax)
    _panel_label(ax, "a")

    # b: architecture violations
    ax = axes[0, 1]
    viol = [sum(r["physical_threshold_violations"] for r in rows if float(r["delta"]) in deltas) for _l, rows, _c in (("ring", ring, BLUE), ("linear", linear, ORANGE), ("all-to-all", all_to_all, GREEN))]
    ax.bar([0, 1, 2], viol, color=[BLUE, ORANGE, GREEN], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["ring", "linear", "all-to-all"], fontsize=6.5)
    ax.set_ylabel("Path violations")
    ax.set_ylim(0, 2)
    _paper_axes(ax)
    _panel_label(ax, "b")

    # c: path budget gap
    ax = axes[0, 2]
    budgets = [None, 0.5, 1.0, 1.5]
    labels = ["none", "0.5", "1.0", "1.5"]
    gaps, cis = [], []
    for budget_val in budgets:
        key = (lambda r: r["path_budget"] is None) if budget_val is None else (lambda r, b=budget_val: r["path_budget"] is not None and abs(float(r["path_budget"]) - b) < 1e-9)
        m, c = mean_ci95(seed_mean_gaps(budget, key))
        gaps.append(m)
        cis.append(c)
    ax.errorbar([0, 1, 2, 3], gaps, yerr=cis, fmt="o", markersize=3.0, markerfacecolor="white",
                markeredgecolor=BLUE, markeredgewidth=0.7, ecolor=BLUE, elinewidth=0.6, capsize=1.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_xlabel("Path budget, B")
    ax.set_ylabel("Final gap")
    _paper_axes(ax)
    _panel_label(ax, "c")

    # d: path budget deletions
    ax = axes[1, 0]
    ks = []
    for budget_val in budgets:
        key = (lambda r: r["path_budget"] is None) if budget_val is None else (lambda r, b=budget_val: r["path_budget"] is not None and abs(float(r["path_budget"]) - b) < 1e-9)
        seed_k = []
        for seed in range(3001, 3013):
            g = [r for r in budget if r["seed"] == seed and key(r)]
            seed_k.append(float(np.mean([r["deleted_count"] for r in g])))
        ks.append(float(np.mean(seed_k)))
    ax.bar([0, 1, 2, 3], ks, color=RED, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_xlabel("Path budget, B")
    ax.set_ylabel("Deleted rotations")
    _paper_axes(ax)
    _panel_label(ax, "d")

    # e: calibration size gap
    ax = axes[1, 1]
    sizes = [1, 2, 4, 8]
    gaps, cis = [], []
    for m in sizes:
        mean, ci = mean_ci95(seed_mean_gaps(size, lambda r, s=m: int(r["calibration_size"]) == s))
        gaps.append(mean)
        cis.append(ci)
    ax.errorbar(sizes, gaps, yerr=cis, fmt="o-", markersize=2.8, markerfacecolor="white",
                markeredgecolor=BLUE, markeredgewidth=0.7, ecolor=BLUE, elinewidth=0.5,
                capsize=1.2, linewidth=0.8)
    ax.set_xlabel("Calibration size, M")
    ax.set_ylabel("Final gap")
    ax.set_xticks(sizes)
    _paper_axes(ax)
    _panel_label(ax, "e")

    # f: calibration size violations
    ax = axes[1, 2]
    viol = [int(sum(r["physical_threshold_violations"] for r in size if int(r["calibration_size"]) == m)) for m in sizes]
    ax.bar([0, 1, 2, 3], viol, color=ORANGE, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([str(m) for m in sizes])
    ax.set_xlabel("Calibration size, M")
    ax.set_ylabel("Path violations")
    _paper_axes(ax)
    _panel_label(ax, "f")

    return _finish(fig, output_dir / "fig_recent_ablations_combined")


def main() -> None:
    configure_style()
    ring = load_rows(PROJECT_DIR / "runs" / "tfim_pruning_transition_formal_dense_20260915")
    linear = load_rows(BUNDLE / "runs" / "tfim_n4_linear_formal_20260916")
    all_to_all = load_rows(BUNDLE / "runs" / "tfim_n4_alltoall_formal_20260916")
    fig1 = architecture_figure(ring, linear, all_to_all, PROJECT_DIR / "figures" / "tfim_architecture_compare_20260916")
    plt.close(fig1)
    budget = load_rows(PROJECT_DIR / "runs" / "tfim_path_budget_ablation_20260916")
    fig2 = budget_figure(budget, PROJECT_DIR / "figures" / "tfim_path_budget_20260916")
    plt.close(fig2)
    size = load_rows(PROJECT_DIR / "runs" / "tfim_calibration_size_ablation_20260916")
    fig3 = size_figure(size, PROJECT_DIR / "figures" / "tfim_calibration_size_20260916")
    plt.close(fig3)
    fig4 = combined_figure(ring, linear, all_to_all, budget, size, PROJECT_DIR / "figures" / "recent_ablations_combined_20260916")
    plt.close(fig4)
    fig5 = haug_style_figure(ring, linear, all_to_all, budget, size, PROJECT_DIR / "figures" / "recent_ablations_haug_style_20260916")
    plt.close(fig5)
    print("Saved architecture, path-budget, and calibration-size figures.")


if __name__ == "__main__":
    main()
