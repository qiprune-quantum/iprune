from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from iprune.figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment


# Keep all text editable in the vector exports.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


BLUE = "#2C6DB2"
RED = "#D45B52"
GREEN = "#4D9A57"
ORANGE = "#E49B45"
GRAY = "#8A8A8A"
LIGHT_GRAY = "#D7D7D7"
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
            "lines.linewidth": 1.05,
            "lines.markersize": 2.8,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the six-panel TFIM pruning-transition figure."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("runs/tfim_pruning_transition_formal_dense_20260915"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("figures/tfim_pruning_transition_20260915"),
    )
    parser.add_argument("--expected_seeds", type=int, default=12)
    parser.add_argument("--expected_deltas", type=int, default=21)
    return parser.parse_args()


def load_and_validate(
    input_dir: Path, expected_seeds: int, expected_deltas: int
) -> tuple[list[dict[str, Any]], list[float], list[int]]:
    files = sorted(input_dir.glob("seed*.json"))
    if len(files) != expected_seeds:
        raise ValueError(f"Expected {expected_seeds} seed files, found {len(files)}.")

    rows: list[dict[str, Any]] = []
    seed_grids: dict[int, tuple[float, ...]] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload["meta"]
        if meta.get("status") != "formal":
            raise ValueError(f"Non-formal input: {path}")
        if not bool(meta["vqe_converged"]):
            raise ValueError(f"Unconverged VQE input: {path}")
        seed = int(meta["seed"])
        current = tuple(float(value) for value in meta["deltas"])
        seed_grids[seed] = current
        for row in payload["results"]:
            if int(row["seed"]) != seed:
                raise ValueError(f"Seed mismatch in {path}")
            if not bool(row["vqe_converged"]):
                raise ValueError(f"Unconverged row in {path}")
            path_steps = row["accepted_path"]
            if len(path_steps) != int(row["pruned_positions"]):
                raise ValueError("Accepted path length does not match K.")
            history = row["finetune_gap_vs_control_history"]
            if len(history) != int(row["finetune_steps"]) + 1:
                raise ValueError("Fine-tuning trajectory has the wrong length.")
            previous = -np.inf
            for expected_step, step in enumerate(path_steps, start=1):
                if int(step["step"]) != expected_step:
                    raise ValueError("Deletion steps are not consecutive.")
                if float(step["physical_angle"]) > float(row["epsilon_q"]) + 1e-10:
                    raise ValueError("An accepted local angle exceeds epsilon.")
                cumulative = float(step["cumulative_trace_cost"])
                if cumulative + 1e-12 < previous:
                    raise ValueError("Cumulative path cost is not monotone.")
                previous = cumulative
            if path_steps and not np.isclose(
                float(path_steps[-1]["cumulative_trace_cost"]),
                float(row["realized_path_trace_sum_max"]),
                atol=1e-10,
            ):
                raise ValueError("Stored path endpoint does not match the audit.")
            if float(row["actual_trace_distance_max"]) > min(
                1.0, float(row["realized_path_trace_sum_max"])
            ) + 1e-9:
                raise ValueError("Actual trace distance exceeds the path bound.")
            rows.append(row)

    grids = set(seed_grids.values())
    if len(grids) != 1:
        raise ValueError("Seeds do not share one threshold grid.")
    deltas = list(next(iter(grids)))
    if len(deltas) != expected_deltas or np.any(np.diff(deltas) <= 0.0):
        raise ValueError("Threshold grid is incomplete or not strictly increasing.")
    if len(rows) != expected_seeds * expected_deltas:
        raise ValueError("The seed-threshold table is incomplete.")
    seeds = sorted(seed_grids)
    return rows, deltas, seeds


def mean_ci95(values: Sequence[float]) -> tuple[float, float, int]:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    n = len(sample)
    if n == 0:
        return math.nan, math.nan, 0
    mean = float(np.mean(sample))
    if n == 1:
        return mean, 0.0, n
    sem = float(np.std(sample, ddof=1) / np.sqrt(n))
    halfwidth = float(student_t.ppf(0.975, df=n - 1) * sem)
    return mean, halfwidth, n


def group_by_delta(rows: Iterable[dict[str, Any]]) -> dict[float, list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["delta"])].append(row)
    return dict(grouped)


def summary_series(
    grouped: dict[float, list[dict[str, Any]]],
    deltas: Sequence[float],
    getter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats = [mean_ci95([getter(row) for row in grouped[delta]]) for delta in deltas]
    return tuple(np.asarray(part, dtype=float) for part in zip(*stats))  # type: ignore[return-value]


def write_csv(path: Path, fieldnames: Sequence[str], records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def export_source_data(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    grouped: dict[float, list[dict[str, Any]]],
    deltas: Sequence[float],
) -> None:
    threshold_fields = [
        "seed",
        "delta",
        "epsilon_q",
        "vqe_steps_run",
        "pruned_positions",
        "rotation_count_before",
        "rotation_prune_ratio",
        "realized_path_trace_sum_max",
        "actual_trace_distance_max",
        "energy_gap_one_shot_vs_control",
        "energy_gap_finetuned_vs_control",
        "abs_dE_one_shot_vs_original",
        "abs_dE_finetuned_vs_original",
        "selection_time_ms",
    ]
    write_csv(
        output_dir / "fig1_source_threshold_rows.csv",
        threshold_fields,
        [{key: row[key] for key in threshold_fields} for row in rows],
    )

    local_records: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    for row in rows:
        for step in row["accepted_path"]:
            local_records.append(
                {
                    "seed": row["seed"],
                    "delta": row["delta"],
                    "epsilon_q": row["epsilon_q"],
                    **step,
                }
            )
        for tune_step, (energy, gap) in enumerate(
            zip(row["finetune_energy_history"], row["finetune_gap_vs_control_history"])
        ):
            trajectory_records.append(
                {
                    "seed": row["seed"],
                    "delta": row["delta"],
                    "finetune_step": tune_step,
                    "energy": energy,
                    "energy_gap_vs_control": gap,
                }
            )
    write_csv(
        output_dir / "fig1_source_local_angles.csv",
        [
            "seed",
            "delta",
            "epsilon_q",
            "step",
            "position",
            "q_angle",
            "physical_angle",
            "local_trace_cost",
            "cumulative_trace_cost",
            "trace_distance_upper_bound",
        ],
        local_records,
    )
    write_csv(
        output_dir / "fig1_source_finetune_trajectories.csv",
        ["seed", "delta", "finetune_step", "energy", "energy_gap_vs_control"],
        trajectory_records,
    )

    summary_records: list[dict[str, Any]] = []
    metrics = {
        "rotation_prune_ratio": lambda row: float(row["rotation_prune_ratio"]),
        "path_cost": lambda row: float(row["realized_path_trace_sum_max"]),
        "path_trace_bound": lambda row: min(
            1.0, float(row["realized_path_trace_sum_max"])
        ),
        "actual_trace_distance": lambda row: float(row["actual_trace_distance_max"]),
        "one_shot_abs_dE_vs_original": lambda row: float(row["abs_dE_one_shot_vs_original"]),
        "finetuned_energy_gap": lambda row: float(row["energy_gap_finetuned_vs_control"]),
    }
    for delta in deltas:
        record: dict[str, Any] = {"delta": delta, "n_seeds": len(grouped[delta])}
        for name, getter in metrics.items():
            mean, ci, n = mean_ci95([getter(row) for row in grouped[delta]])
            record[f"{name}_mean"] = mean
            record[f"{name}_ci95_halfwidth"] = ci
            record[f"{name}_n"] = n
        summary_records.append(record)
    write_csv(
        output_dir / "fig1_source_summary.csv",
        list(summary_records[0]),
        summary_records,
    )


def panel_label(ax: plt.Axes, label: str) -> None:
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


def paper_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=LIGHT_GRAY, linestyle=":", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", pad=1.8)


def plot_mean_ci(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    ci: np.ndarray,
    *,
    color: str,
    label: str | None = None,
    linestyle: str = "-",
    marker: str = "o",
    zorder: int = 3,
) -> None:
    lower = mean - ci
    upper = mean + ci
    ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0, zorder=zorder - 1)
    ax.plot(
        x,
        mean,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markerfacecolor="white" if marker != "None" else color,
        markeredgewidth=0.65,
        label=label,
        zorder=zorder,
    )


def add_transition_guides(ax: plt.Axes) -> None:
    ax.axvline(1.1, color=GRAY, linewidth=0.55, linestyle=":", alpha=0.65, zorder=0)
    ax.axvline(1.2, color=GRAY, linewidth=0.55, linestyle=":", alpha=0.65, zorder=0)


def make_figure(
    rows: Sequence[dict[str, Any]], deltas: Sequence[float], output_dir: Path
) -> plt.Figure:
    grouped = group_by_delta(rows)
    x = np.asarray(deltas, dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.85), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.95, wspace=0.39, hspace=0.43)

    # (a) Gate-removal response.
    ax = axes[0, 0]
    prune, prune_ci, _ = summary_series(
        grouped, deltas, lambda row: 100.0 * float(row["rotation_prune_ratio"])
    )
    plot_mean_ci(ax, x, prune, prune_ci, color=BLUE)
    ax.axhline(100.0, color=GREEN, linestyle=":", linewidth=0.9)
    add_transition_guides(ax)
    ax.set_ylabel("Pruned rotations (%)")
    ax.set_ylim(-4.0, 106.0)
    panel_label(ax, "a")
    paper_axes(ax)

    # (b) Distribution of accepted local physical angles.
    ax = axes[0, 1]
    representative = [(0.8, BLUE), (1.2, ORANGE), (1.6, GREEN)]
    for delta, color in representative:
        angles = sorted(
            float(step["physical_angle"])
            for row in grouped[delta]
            for step in row["accepted_path"]
        )
        if not angles:
            raise ValueError(f"No accepted local angles at delta={delta}.")
        angle_x = np.asarray(angles, dtype=float)
        cumulative = np.arange(1, len(angle_x) + 1, dtype=float) / len(angle_x)
        ax.plot(
            angle_x,
            cumulative,
            color=color,
            linewidth=1.05,
            label=f"δ={delta:g}",
        )
        epsilon = float(grouped[delta][0]["epsilon_q"])
        ax.axvline(epsilon, color=color, linestyle=":", linewidth=0.7, alpha=0.8)
    ax.set_xlabel("Accepted local angle, θ")
    ax.set_ylabel("Cumulative probability")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.03)
    ax.legend(loc="lower right", handlelength=1.45, borderaxespad=0.25)
    panel_label(ax, "b")
    paper_axes(ax)

    # (c) Sequential accumulation of locally bounded deviations.
    ax = axes[0, 2]
    path_mean, path_ci, _ = summary_series(
        grouped, deltas, lambda row: float(row["realized_path_trace_sum_max"])
    )
    plot_mean_ci(ax, x, path_mean, path_ci, color=BLUE)
    add_transition_guides(ax)
    ax.set_ylabel("Path cost, Σ sin θ")
    ax.set_ylim(bottom=0.0)
    panel_label(ax, "c")
    paper_axes(ax)

    # (d) Orthogonal state-space validation of the accumulated path cost.
    ax = axes[1, 0]
    trace_mean, trace_ci, _ = summary_series(
        grouped, deltas, lambda row: float(row["actual_trace_distance_max"])
    )
    bound_mean, bound_ci, _ = summary_series(
        grouped,
        deltas,
        lambda row: min(1.0, float(row["realized_path_trace_sum_max"])),
    )
    plot_mean_ci(ax, x, trace_mean, trace_ci, color=BLUE, label="Actual")
    plot_mean_ci(
        ax,
        x,
        bound_mean,
        bound_ci,
        color=RED,
        label="Path bound",
        linestyle="--",
        marker="s",
    )
    add_transition_guides(ax)
    ax.set_ylabel("Trace distance")
    ax.set_ylim(-0.03, 1.06)
    ax.legend(loc="lower right", handlelength=1.55, borderaxespad=0.25)
    panel_label(ax, "d")
    paper_axes(ax)

    # (e) Immediate task-level consequence.
    ax = axes[1, 1]
    one_mean, one_ci, _ = summary_series(
        grouped, deltas, lambda row: float(row["abs_dE_one_shot_vs_original"])
    )
    plot_mean_ci(ax, x[:-1], one_mean[:-1], one_ci[:-1], color=BLUE)
    add_transition_guides(ax)
    ax.scatter(
        [x[-1]],
        [one_mean[-1]],
        s=18,
        facecolors="white",
        edgecolors=BLACK,
        linewidths=0.75,
        zorder=5,
    )
    ax.set_ylabel("One-shot energy gap, ΔE")
    ax.set_ylim(bottom=0.0)
    panel_label(ax, "e")
    paper_axes(ax)

    # (f) Recovery boundary after the matched fine-tuning budget.
    ax = axes[1, 2]
    tuned_mean, tuned_ci, _ = summary_series(
        grouped, deltas, lambda row: float(row["energy_gap_finetuned_vs_control"])
    )
    plot_mean_ci(ax, x[:-1], tuned_mean[:-1], tuned_ci[:-1], color=RED)
    ax.axhline(0.0, color=GREEN, linestyle=":", linewidth=0.9)
    add_transition_guides(ax)
    ax.scatter(
        [x[-1]],
        [tuned_mean[-1]],
        s=18,
        facecolors="white",
        edgecolors=BLACK,
        linewidths=0.75,
        zorder=5,
    )
    ax.set_ylabel("Fine-tuned energy gap, ΔE")
    ax.set_ylim(bottom=0.0)
    panel_label(ax, "f")
    paper_axes(ax)

    threshold_axes = [axes[0, 0], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for ax in threshold_axes:
        ax.set_xlabel("Threshold scale, δ")
        ax.set_xlim(0.15, 2.05)
        ax.set_xticks([0.2, 0.6, 1.0, 1.4, 1.8, 2.0])

    fig.canvas.draw()
    prefix = output_dir / "fig1_tfim_pruning_transition"
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
    prefix = output_dir / "fig1_tfim_pruning_transition"
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
    rows, deltas, _seeds = load_and_validate(
        args.input_dir, args.expected_seeds, args.expected_deltas
    )
    grouped = group_by_delta(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_source_data(args.output_dir, rows, grouped, deltas)
    figure = make_figure(rows, deltas, args.output_dir)
    save_figure(figure, args.output_dir)
    plt.close(figure)
    print(f"Validated {len(rows)} seed-threshold rows.")
    print(f"Saved figure bundle to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
