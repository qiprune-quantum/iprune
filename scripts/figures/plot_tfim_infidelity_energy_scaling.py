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
from scripts.figures.plot_tfim_pruning_transition import (
    BLACK,
    BLUE,
    GRAY,
    RED,
    configure_style,
    group_by_delta,
    load_and_validate,
    mean_ci95,
)


# Explicit publication/export settings are repeated here so this standalone
# source remains auditable even though the shared style helper sets them too.
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relate TFIM pruning-induced infidelity to one-shot energy damage."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("runs/tfim_pruning_transition_formal_dense_20260915"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("figures/tfim_infidelity_energy_scaling_20260916"),
    )
    parser.add_argument("--expected_seeds", type=int, default=12)
    parser.add_argument("--expected_deltas", type=int, default=21)
    return parser.parse_args()


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(x, x))
    if denominator <= 0.0:
        raise ValueError("Cannot fit an origin-constrained line to zero-valued x.")
    slope = float(np.dot(x, y) / denominator)
    prediction = slope * x
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum((y - prediction) ** 2)) / total
    return slope, r_squared


def summarize(
    rows: list[dict[str, Any]], deltas: list[float]
) -> list[dict[str, float]]:
    grouped = group_by_delta(rows)
    records: list[dict[str, float]] = []
    for delta in deltas:
        group = grouped[delta]
        infidelity = [float(row["actual_trace_distance_max"]) ** 2 for row in group]
        damage = [float(row["abs_dE_one_shot_vs_original"]) for row in group]
        x_mean, x_ci, _ = mean_ci95(infidelity)
        y_mean, y_ci, _ = mean_ci95(damage)
        records.append(
            {
                "delta": delta,
                "n_seeds": float(len(group)),
                "infidelity_mean": x_mean,
                "infidelity_ci95_halfwidth": x_ci,
                "one_shot_abs_dE_mean": y_mean,
                "one_shot_abs_dE_ci95_halfwidth": y_ci,
            }
        )
    return records


def export_source_data(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, float]],
    slope: float,
    r_squared: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "source_seed_points.csv"
    with raw_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "delta", "infidelity", "one_shot_abs_dE"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": int(row["seed"]),
                    "delta": float(row["delta"]),
                    "infidelity": float(row["actual_trace_distance_max"]) ** 2,
                    "one_shot_abs_dE": float(row["abs_dE_one_shot_vs_original"]),
                }
            )

    summary_path = output_dir / "source_threshold_means.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    fit_path = output_dir / "fit_summary.json"
    fit_path.write_text(
        json.dumps(
            {
                "fit": "origin-constrained ordinary least squares",
                "fit_unit": "threshold means",
                "excluded_from_fit": "delta=2.0 all-rotations-removed endpoint",
                "n_threshold_means": len(summary) - 1,
                "slope": slope,
                "r_squared_descriptive": r_squared,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def make_figure(
    rows: list[dict[str, Any]],
    summary: list[dict[str, float]],
    slope: float,
    r_squared: float,
    output_dir: Path,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.50, 2.83), constrained_layout=True)

    regular_rows = [row for row in rows if float(row["delta"]) < 2.0]
    raw_x = np.asarray(
        [float(row["actual_trace_distance_max"]) ** 2 for row in regular_rows]
    )
    raw_y = np.asarray(
        [float(row["abs_dE_one_shot_vs_original"]) for row in regular_rows]
    )
    seed_handle = ax.scatter(
        raw_x,
        raw_y,
        s=7,
        color=GRAY,
        alpha=0.20,
        linewidths=0,
        rasterized=False,
        zorder=1,
    )

    regular = [record for record in summary if record["delta"] < 2.0]
    x = np.asarray([record["infidelity_mean"] for record in regular])
    y = np.asarray([record["one_shot_abs_dE_mean"] for record in regular])
    xerr = np.asarray([record["infidelity_ci95_halfwidth"] for record in regular])
    yerr = np.asarray([record["one_shot_abs_dE_ci95_halfwidth"] for record in regular])
    mean_handle = ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        markersize=3.5,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        markeredgewidth=0.85,
        ecolor=BLUE,
        elinewidth=0.55,
        capsize=1.3,
        alpha=0.98,
        zorder=3,
    )

    fit_x = np.linspace(0.0, 1.02, 200)
    ax.plot(
        fit_x,
        slope * fit_x,
        color=RED,
        linewidth=1.15,
        zorder=2,
    )

    endpoint = next(record for record in summary if np.isclose(record["delta"], 2.0))
    ax.errorbar(
        [endpoint["infidelity_mean"]],
        [endpoint["one_shot_abs_dE_mean"]],
        xerr=[endpoint["infidelity_ci95_halfwidth"]],
        yerr=[endpoint["one_shot_abs_dE_ci95_halfwidth"]],
        fmt="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor=BLACK,
        markeredgewidth=0.9,
        ecolor=BLACK,
        elinewidth=0.6,
        capsize=1.5,
        zorder=4,
    )
    ax.annotate(
        "δ=2.0\nall rotations removed",
        (endpoint["infidelity_mean"], endpoint["one_shot_abs_dE_mean"]),
        xytext=(0.71, 0.76),
        textcoords="data",
        fontsize=6.6,
        color=BLACK,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": BLACK, "linewidth": 0.6},
    )

    ax.text(
        0.04,
        0.94,
        f"|ΔE| = {slope:.2f}D²\nR² = {r_squared:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=RED,
    )
    ax.set_xlabel("State infidelity, D²")
    ax.set_ylabel("One-shot energy damage, |ΔE|")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=-0.12)
    ax.grid(False)
    ax.legend(
        [seed_handle, mean_handle],
        ["Individual seeds", "Threshold means ± 95% CI"],
        loc="upper center",
        bbox_to_anchor=(0.64, 0.985),
        handlelength=1.6,
        borderaxespad=0.0,
    )

    fig.canvas.draw()
    prefix = output_dir / "fig2_infidelity_energy_scaling"
    require_matplotlib_panel_alignment(
        fig,
        json_out=prefix.with_suffix(".alignment.json"),
        overlay_svg=prefix.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )
    return fig


def save_figure(fig: plt.Figure, output_dir: Path) -> None:
    prefix = output_dir / "fig2_infidelity_energy_scaling"
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
    mpl.rcParams["axes.labelsize"] = 8.0
    rows, deltas, _ = load_and_validate(
        args.input_dir, args.expected_seeds, args.expected_deltas
    )
    summary = summarize(rows, deltas)
    regular = [record for record in summary if record["delta"] < 2.0]
    x = np.asarray([record["infidelity_mean"] for record in regular])
    y = np.asarray([record["one_shot_abs_dE_mean"] for record in regular])
    slope, r_squared = fit_through_origin(x, y)
    export_source_data(args.output_dir, rows, summary, slope, r_squared)
    fig = make_figure(rows, summary, slope, r_squared, args.output_dir)
    save_figure(fig, args.output_dir)
    plt.close(fig)
    print(f"Validated {len(rows)} seed-threshold rows.")
    print(f"Origin fit: |dE| = {slope:.6f} D^2, R^2 = {r_squared:.6f}")
    print(f"Saved figure bundle to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
