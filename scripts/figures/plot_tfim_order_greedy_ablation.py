from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from iprune.figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment


PROJECT_DIR = Path(__file__).resolve().parents[2]
BLUE = "#2C6DB2"
RED = "#D45B52"
GREEN = "#4D9A57"
ORANGE = "#E49B45"
BLACK = "#202020"


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


METHODS = [
    ("forward_dynamic", "Forward"),
    ("global_greedy", "Greedy"),
    ("static_frozen", "Static"),
    ("backward_dynamic", "Backward"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the order-and-greedy ablation.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "tfim_order_greedy_ablation_20260916",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_DIR / "figures" / "tfim_order_greedy_ablation_20260916",
    )
    return parser.parse_args()


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
    halfwidth = float(student_t.ppf(0.975, df=n - 1) * sem)
    return mean, halfwidth


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


def make_figure(rows: list[dict[str, Any]], output_dir: Path) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.92, wspace=0.36)

    deltas = [0.8, 1.0, 1.2]
    colors = [BLUE, GREEN, ORANGE, RED]

    # (a) Final energy gap as a function of the threshold for each rule.
    ax = axes[0]
    for (method, label), color in zip(METHODS, colors):
        means: list[float] = []
        cis: list[float] = []
        for delta in deltas:
            seed_gaps = []
            for seed in range(3001, 3013):
                group = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["seed"] == seed
                    and abs(float(r["delta"]) - delta) < 1e-9
                ]
                seed_gaps.append(
                    float(np.mean([r["energy_gap_finetuned_vs_control"] for r in group]))
                )
            mean, ci = mean_ci95(seed_gaps)
            means.append(mean)
            cis.append(ci)
        ax.errorbar(
            deltas,
            means,
            yerr=cis,
            fmt="o-",
            markersize=3.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=0.55,
            capsize=1.4,
            linewidth=0.9,
            label=label,
            zorder=3,
        )
    ax.set_xlabel("Threshold scale, δ")
    ax.set_ylabel("Final energy gap")
    ax.set_xticks(deltas)
    ax.set_xlim(0.75, 1.25)
    ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.2)
    ax.grid(axis="y", color="#D7D7D7", linestyle=":", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    _panel_label(ax, "a")

    # (b) Realized-path physical-threshold violations.
    ax = axes[1]
    x = np.arange(len(METHODS))
    violations = []
    for method, _label in METHODS:
        total_viol = 0
        for seed in range(3001, 3013):
            group = [r for r in rows if r["method"] == method and r["seed"] == seed]
            total_viol += int(sum(r["physical_threshold_violations"] for r in group))
        violations.append(total_viol)
    ax.bar(x, violations, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for index, value in enumerate(violations):
        ax.text(index, value + 0.08, str(value), ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _m, label in METHODS])
    ax.set_ylabel("Path threshold violations")
    ax.set_ylim(0, max(violations) * 1.28 + 0.5)
    ax.grid(axis="y", color="#D7D7D7", linestyle=":", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    _panel_label(ax, "b")

    # (c) Seed-paired gap difference relative to the dynamic forward rule.
    ax = axes[2]
    reference = "forward_dynamic"
    compared = [
        ("global_greedy", "Greedy", GREEN),
        ("static_frozen", "Static", ORANGE),
        ("backward_dynamic", "Backward", RED),
    ]
    all_differences: list[float] = []
    for index, (method, label, color) in enumerate(compared):
        differences = []
        for seed in range(3001, 3013):
            ref_gaps = [
                float(r["energy_gap_finetuned_vs_control"])
                for r in rows
                if r["method"] == reference and r["seed"] == seed
            ]
            other_gaps = [
                float(r["energy_gap_finetuned_vs_control"])
                for r in rows
                if r["method"] == method and r["seed"] == seed
            ]
            differences.append(float(np.mean(other_gaps) - np.mean(ref_gaps)))
        all_differences.extend(differences)
        jitter = np.random.default_rng(index + 1).normal(0.0, 0.018, size=len(differences))
        ax.scatter(
            np.full(len(differences), index) + jitter,
            differences,
            s=10,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
        )
        mean, ci = mean_ci95(differences)
        ax.errorbar(
            [index],
            [mean],
            yerr=[ci],
            fmt="o",
            markersize=3.6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            ecolor=color,
            elinewidth=0.8,
            capsize=2.0,
            zorder=5,
        )
    ax.axhline(0.0, color="#202020", linestyle=":", linewidth=0.7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Greedy", "Static", "Backward"])
    ax.set_xlim(-0.55, 2.55)
    ymax = float(max(np.max(np.abs(all_differences)) * 1.25, 0.1))
    ax.set_ylim(-ymax, ymax)
    ax.set_ylabel("Gap difference vs Forward")
    ax.grid(axis="y", color="#D7D7D7", linestyle=":", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    _panel_label(ax, "c")

    fig.canvas.draw()
    prefix = output_dir / "fig_tfim_order_greedy_ablation"
    require_matplotlib_panel_alignment(
        fig,
        json_out=prefix.with_suffix(".alignment.json"),
        overlay_svg=prefix.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    return fig


def save_figure(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "fig_tfim_order_greedy_ablation"
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    args = parse_args()
    configure_style()
    rows = load_rows(args.input_dir)
    fig = make_figure(rows, args.output_dir)
    save_figure(fig, args.output_dir)
    plt.close(fig)
    print(f"Saved order/greedy ablation figure to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
