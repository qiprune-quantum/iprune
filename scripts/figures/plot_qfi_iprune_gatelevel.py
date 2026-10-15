from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from iprune.figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment


PROJECT_DIR = Path(__file__).resolve().parents[2]
BLUE = "#2C6DB2"
RED = "#D45B52"
GREEN = "#4D9A57"
ORANGE = "#E49B45"
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
        description="Build the main-text QFI vs iPrune vs true-damage figure."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "qfi_iprune_gatelevel_n4_20260916",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_DIR / "figures" / "qfi_iprune_gatelevel_20260916",
    )
    return parser.parse_args()


def load_gate_rows(input_dir: Path) -> list[dict[str, Any]]:
    path = input_dir / "gate_level_results.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 192:
        raise ValueError(f"expected 192 gate rows, got {len(rows)}")
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float_rows(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


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


def _annotate_correlation(
    ax: plt.Axes,
    label: str,
    mean: float,
    ci: tuple[float, float],
    *,
    loc: str = "upper left",
) -> None:
    ax.text(
        0.03 if loc == "upper left" else 0.97,
        0.95 if loc in {"upper left", "upper right"} else 0.05,
        f"{label} ρ = {mean:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
        transform=ax.transAxes,
        ha="left" if loc == "upper left" else "right",
        va="top" if loc in {"upper left", "upper right"} else "bottom",
        fontsize=6.8,
        color=BLACK,
    )


def make_figure(
    gate_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    experiment: dict[str, Any],
    output_dir: Path,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.05), facecolor="white")
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.105, top=0.945, wspace=0.36, hspace=0.52
    )

    # (a) QFI rank/nullity stability across seeds and tolerances.
    ax = axes[0, 0]
    ranks = np.asarray(experiment["qfi_ranks"], dtype=float)
    nullities = np.asarray(experiment["qfi_nullities"], dtype=float)
    x = np.arange(1, len(ranks) + 1)
    ax.scatter(x, ranks, s=12, color=BLUE, edgecolors="white", linewidths=0.4, label="rank")
    ax.scatter(x, nullities, s=12, color=ORANGE, edgecolors="white", linewidths=0.4, label="nullity")
    ax.axhline(np.mean(ranks), color=BLUE, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(np.mean(nullities), color=ORANGE, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Training seed index")
    ax.set_ylabel("Dimension")
    ax.set_ylim(0, 50)
    ax.set_xticks([1, 4, 7, 10, 12])
    ax.legend(loc="center right", handlelength=1.3, borderaxespad=0.2)
    ax.text(
        0.03,
        0.97,
        "stable for tol\n1e-6 / 1e-8 / 1e-10",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=GRAY,
    )
    _paper_axes(ax)
    _panel_label(ax, "a")

    # (b) QFI null-space participation vs true single-gate energy damage.
    ax = axes[0, 1]
    qfi = _as_float_rows(gate_rows, "qfi_null_participation_mean")
    damage = _as_float_rows(gate_rows, "single_delete_energy_abs")
    ax.scatter(qfi, damage, s=8, color=BLUE, alpha=0.38, linewidths=0, rasterized=False)
    ax.set_xlabel("QFI null-space participation")
    ax.set_ylabel("True single-gate |ΔE|")
    sp = stats["within_seed_spearman"]["qfi_redundancy_vs_energy_damage"]
    _annotate_correlation(
        ax,
        "QFI–damage",
        float(sp["mean"]),
        (float(sp["seed_bootstrap_95_ci_for_mean"][0]), float(sp["seed_bootstrap_95_ci_for_mean"][1])),
    )
    _paper_axes(ax)
    _panel_label(ax, "b")

    # (c) iPrune deletion angle vs true single-gate energy damage.
    ax = axes[0, 2]
    angle = _as_float_rows(gate_rows, "iprune_q_angle")
    ax.scatter(angle, damage, s=8, color=RED, alpha=0.38, linewidths=0, rasterized=False)
    ax.set_xlabel("iPrune deletion angle")
    ax.set_ylabel("True single-gate |ΔE|")
    sp = stats["within_seed_spearman"]["iprune_q_angle_vs_energy_damage"]
    _annotate_correlation(
        ax,
        "iPrune–damage",
        float(sp["mean"]),
        (float(sp["seed_bootstrap_95_ci_for_mean"][0]), float(sp["seed_bootstrap_95_ci_for_mean"][1])),
    )
    _paper_axes(ax)
    _panel_label(ax, "c")

    # (d) QFI vs iPrune disagreement, colored by true damage.
    ax = axes[1, 0]
    scatter = ax.scatter(
        qfi,
        angle,
        c=damage,
        cmap="viridis",
        norm=Normalize(vmin=float(np.min(damage)), vmax=float(np.max(damage))),
        s=9,
        linewidths=0,
        rasterized=False,
    )
    ax.set_xlabel("QFI null-space participation")
    ax.set_ylabel("iPrune deletion angle")
    colorbar = fig.colorbar(
        ScalarMappable(norm=scatter.norm, cmap=scatter.cmap),
        ax=ax,
        pad=0.02,
        fraction=0.045,
    )
    colorbar.ax.tick_params(labelsize=6.2, pad=1.0)
    colorbar.set_label("True single-gate |ΔE|", fontsize=6.8, labelpad=1.0)
    _paper_axes(ax)
    _panel_label(ax, "d")

    # (e) Within-seed Spearman correlations, aggregated across 12 seeds.
    ax = axes[1, 1]
    comparisons = [
        ("QFI–damage", "qfi_redundancy_vs_energy_damage"),
        ("iPrune–damage", "iprune_q_angle_vs_energy_damage"),
        ("FS–damage", "physical_fs_angle_vs_energy_damage"),
    ]
    for index, (label, key) in enumerate(comparisons):
        entry = stats["within_seed_spearman"][key]
        values = np.asarray(entry["per_seed"], dtype=float)
        ci = entry["seed_bootstrap_95_ci_for_mean"]
        jitter = np.random.default_rng(index + 1).normal(0.0, 0.022, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=11,
            color=[BLUE, RED, GREEN][index],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.85,
        )
        mean = float(np.mean(values))
        ax.plot(
            [index - 0.28, index + 0.28],
            [mean, mean],
            color=[BLUE, RED, GREEN][index],
            linewidth=1.2,
        )
        ax.plot(
            [index, index],
            [float(ci[0]), float(ci[1])],
            color=[BLUE, RED, GREEN][index],
            linewidth=0.9,
        )
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=0.7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["QFI", "iPrune", "FS"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("Within-seed Spearman ρ")
    ax.set_ylim(-0.75, 1.05)
    _paper_axes(ax)
    _panel_label(ax, "e")

    # (f) Matched top-4 selection: QFI redundancy vs iPrune deletion angle.
    ax = axes[1, 2]
    sel = stats["lowest_damage_selection"]
    qfi_damage = np.asarray(sel["qfi_highest_redundancy_mean_damage"]["per_seed"], dtype=float)
    iprune_damage = np.asarray(sel["iprune_lowest_angle_mean_damage"]["per_seed"], dtype=float)
    x = np.arange(1, len(qfi_damage) + 1)
    ax.plot(x, qfi_damage, "-o", color=BLUE, markersize=3.0, linewidth=1.0, label="QFI top-4")
    ax.plot(x, iprune_damage, "-o", color=RED, markersize=3.0, linewidth=1.0, label="iPrune top-4")
    ax.set_xlabel("Training seed index")
    ax.set_ylabel("Mean true single-gate |ΔE|")
    ax.set_xticks([1, 4, 7, 10, 12])
    ax.legend(loc="upper right", handlelength=1.3, borderaxespad=0.2)
    p_value = float(sel["paired_wilcoxon_two_sided_p"])
    ax.text(
        0.03,
        0.97,
        f"paired diff = {float(sel['qfi_minus_iprune_damage']['mean']):.3f}\n"
        f"Wilcoxon p = {p_value:g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=BLACK,
    )
    _paper_axes(ax)
    _panel_label(ax, "f")

    fig.canvas.draw()
    prefix = output_dir / "fig_qfi_iprune_gatelevel"
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
    prefix = output_dir / "fig_qfi_iprune_gatelevel"
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
    gate_rows = load_gate_rows(args.input_dir)
    stats = load_json(args.input_dir / "statistical_summary.json")
    experiment = load_json(args.input_dir / "experiment_summary.json")
    fig = make_figure(gate_rows, stats, experiment, args.output_dir)
    save_figure(fig, args.output_dir)
    plt.close(fig)
    print(f"Saved QFI/iPrune gate-level figure to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
