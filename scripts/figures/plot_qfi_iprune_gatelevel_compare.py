from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from iprune.figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment


PROJECT_DIR = Path(__file__).resolve().parents[2]
BLUE = "#2C6DB2"
RED = "#D45B52"
GRAY = "#8A8A8A"
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
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare N=4 and N=6 QFI vs iPrune gate-level evidence."
    )
    parser.add_argument(
        "--n4_dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "qfi_iprune_gatelevel_n4_20260916",
    )
    parser.add_argument(
        "--n6_dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "qfi_iprune_gatelevel_n6_20260916",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_DIR / "figures" / "qfi_iprune_gatelevel_compare_20260916",
    )
    return parser.parse_args()


def load_csv(input_dir: Path) -> list[dict[str, Any]]:
    with (input_dir / "gate_level_results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _correlation_entry(stats: dict[str, Any], key: str) -> tuple[float, tuple[float, float]]:
    entry = stats["within_seed_spearman"][key]
    ci = entry["seed_bootstrap_95_ci_for_mean"]
    return float(entry["mean"]), (float(ci[0]), float(ci[1]))


def make_figure(
    n4_rows: list[dict[str, Any]],
    n6_rows: list[dict[str, Any]],
    n4_stats: dict[str, Any],
    n6_stats: dict[str, Any],
    output_dir: Path,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.05), facecolor="white")
    fig.subplots_adjust(
        left=0.10, right=0.985, bottom=0.12, top=0.94, wspace=0.34, hspace=0.48
    )

    # (a) QFI vs iPrune correlation to true damage, N=4 vs N=6.
    ax = axes[0, 0]
    groups = ["QFI–damage", "iPrune–damage", "FS–damage"]
    keys = [
        "qfi_redundancy_vs_energy_damage",
        "iprune_q_angle_vs_energy_damage",
        "physical_fs_angle_vs_energy_damage",
    ]
    colors = [BLUE, RED, GRAY]
    x = np.arange(len(groups))
    for scale, stats, offset, marker in ((4, n4_stats, -0.11, "o"), (6, n6_stats, 0.11, "s")):
        means = []
        lows = []
        highs = []
        for key in keys:
            mean, (lo, hi) = _correlation_entry(stats, key)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        means = np.asarray(means)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        ax.errorbar(
            x + offset,
            means,
            yerr=[means - lows, highs - means],
            fmt=marker,
            markersize=3.6,
            markerfacecolor="white",
            markeredgecolor=[BLUE, RED, GRAY][0],
            ecolor=BLUE if scale == 4 else RED,
            elinewidth=0.8,
            capsize=2.0,
            label=f"N={scale}",
            zorder=3,
        )
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Within-seed Spearman ρ")
    ax.set_ylim(-0.55, 1.05)
    ax.legend(loc="lower left", handlelength=1.4, borderaxespad=0.2)
    _paper_axes(ax)
    _panel_label(ax, "a")

    # (b) Matched top-4 selection damage.
    ax = axes[0, 1]
    categories = ["N=4 QFI", "N=4 iPrune", "N=6 QFI", "N=6 iPrune"]
    values = []
    for stats, scale in ((n4_stats, 4), (n6_stats, 6)):
        sel = stats["lowest_damage_selection"]
        values.append(float(sel["qfi_highest_redundancy_mean_damage"]["mean"]))
        values.append(float(sel["iprune_lowest_angle_mean_damage"]["mean"]))
    x = np.arange(len(categories))
    bars = ax.bar(
        x,
        values,
        color=[BLUE, RED, BLUE, RED],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    for category, bar in zip(categories, bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=BLACK,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=6.6)
    ax.set_ylabel("Mean true single-gate |ΔE|")
    ax.set_ylim(0, 4.0)
    _paper_axes(ax)
    _panel_label(ax, "b")

    # (c) QFI null-space participation vs true damage, both scales.
    ax = axes[1, 0]
    for rows, color, label in ((n4_rows, BLUE, "N=4"), (n6_rows, RED, "N=6")):
        x = np.asarray([float(r["qfi_null_participation_mean"]) for r in rows])
        y = np.asarray([float(r["single_delete_energy_abs"]) for r in rows])
        ax.scatter(x, y, s=8, color=color, alpha=0.34, linewidths=0, rasterized=False, label=label)
    ax.set_xlabel("QFI null-space participation")
    ax.set_ylabel("True single-gate |ΔE|")
    ax.legend(loc="upper right", handlelength=1.2, borderaxespad=0.2)
    _paper_axes(ax)
    _panel_label(ax, "c")

    # (d) iPrune deletion angle vs true damage, both scales.
    ax = axes[1, 1]
    for rows, color, label in ((n4_rows, BLUE, "N=4"), (n6_rows, RED, "N=6")):
        x = np.asarray([float(r["iprune_q_angle"]) for r in rows])
        y = np.asarray([float(r["single_delete_energy_abs"]) for r in rows])
        ax.scatter(x, y, s=8, color=color, alpha=0.34, linewidths=0, rasterized=False, label=label)
    ax.set_xlabel("iPrune deletion angle")
    ax.set_ylabel("True single-gate |ΔE|")
    ax.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2)
    _paper_axes(ax)
    _panel_label(ax, "d")

    fig.canvas.draw()
    prefix = output_dir / "fig_qfi_iprune_compare"
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
    prefix = output_dir / "fig_qfi_iprune_compare"
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
    n4_rows = load_csv(args.n4_dir)
    n6_rows = load_csv(args.n6_dir)
    n4_stats = load_json(args.n4_dir / "statistical_summary.json")
    n6_stats = load_json(args.n6_dir / "statistical_summary.json")
    fig = make_figure(n4_rows, n6_rows, n4_stats, n6_stats, args.output_dir)
    save_figure(fig, args.output_dir)
    plt.close(fig)
    print(f"Saved QFI/iPrune comparison figure to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
