from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from iprune.figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[2]
BLUE = "#2C6DB2"
BLUE_LIGHT = "#8CB7DD"
RED = "#D45B52"
RED_LIGHT = "#EBA59E"
GREEN = "#3B9563"
GRAY = "#858585"
BLACK = "#202020"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.2,
            "axes.labelsize": 7.6,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.fontsize": 6.3,
            "legend.frameon": False,
            "lines.linewidth": 0.9,
            "figure.dpi": 180,
            "savefig.dpi": 600,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="N4/N6 full-QFI versus iPrune mechanism figure")
    parser.add_argument(
        "--n4-dir", type=Path, default=ROOT / "runs" / "qfi_iprune_gatelevel_n4_20260916"
    )
    parser.add_argument(
        "--n6-dir", type=Path, default=ROOT / "runs" / "qfi_iprune_gatelevel_n6_20260916"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "figures" / "qfi_iprune_multiscale_20260916"
    )
    return parser.parse_args()


def read_bundle(path: Path, n_qubits: int) -> dict:
    frame = pd.read_csv(path / "gate_level_results.csv")
    stats = json.loads((path / "statistical_summary.json").read_text(encoding="utf-8"))
    experiment = json.loads((path / "experiment_summary.json").read_text(encoding="utf-8"))
    expected = 12 * n_qubits * 4
    if len(frame) != expected or frame.seed.nunique() != 12:
        raise ValueError(f"N{n_qubits}: expected {expected} rows and 12 seeds")
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError(f"N{n_qubits}: non-finite numerical value")
    frame = frame.copy()
    frame["system"] = f"N={n_qubits}"
    return {"frame": frame, "stats": stats, "experiment": experiment}


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.annotate(
        text,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-16, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        annotation_clip=False,
    )


def paper_axes(ax: plt.Axes, *, xgrid: bool = False) -> None:
    ax.grid(axis="both" if xgrid else "y", color="#D8D8D8", linestyle=":", linewidth=0.42)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", pad=1.7)


def corr_entry(bundle: dict, key: str) -> tuple[float, float, float, np.ndarray]:
    entry = bundle["stats"]["within_seed_spearman"][key]
    lo, hi = entry["seed_bootstrap_95_ci_for_mean"]
    return float(entry["mean"]), float(lo), float(hi), np.asarray(entry["per_seed"], float)


def correlation_note(ax: plt.Axes, n4: dict, n6: dict, key: str) -> None:
    m4, lo4, hi4, _ = corr_entry(n4, key)
    m6, lo6, hi6, _ = corr_entry(n6, key)
    ax.text(
        0.03,
        0.97,
        f"N=4  ρ={m4:.3f}  [{lo4:.3f}, {hi4:.3f}]\n"
        f"N=6  ρ={m6:.3f}  [{lo6:.3f}, {hi6:.3f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        linespacing=1.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )


def make_figure(n4: dict, n6: dict, output_dir: Path) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.12), facecolor="white")
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.105, top=0.95, wspace=0.35, hspace=0.50)

    # a — numerical structure and tolerance robustness.
    ax = axes[0, 0]
    tolerance_keys = ["1e-06", "1e-08", "1e-10"]
    x = np.arange(3)
    for bundle, color, marker, parameter_count in (
        (n4, BLUE, "o", 48),
        (n6, RED, "^", 72),
    ):
        rank_matrix = np.asarray(
            [[row[key] for key in tolerance_keys] for row in bundle["experiment"]["rank_sensitivity"]],
            dtype=float,
        ) / parameter_count
        for row in rank_matrix:
            ax.plot(x, row, color=color, alpha=0.16, linewidth=0.55)
        mean = rank_matrix.mean(axis=0)
        ax.plot(x, mean, color=color, marker=marker, markersize=3.7, linewidth=1.25)
    ax.set_xticks(x, [r"$10^{-6}$", r"$10^{-8}$", r"$10^{-10}$"])
    ax.set_xlabel("Relative eigenvalue tolerance")
    ax.set_ylabel("QFI rank / parameter count")
    ax.set_ylim(0.55, 0.96)
    ax.text(0.03, 0.06, "thin lines: individual seeds", transform=ax.transAxes, fontsize=5.7, color=GRAY)
    ax.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", label="N=4 (48 parameters)"),
            Line2D([], [], color=RED, marker="^", label="N=6 (72 parameters)"),
        ],
        loc="upper left",
        handlelength=1.5,
    )
    paper_axes(ax)
    panel_label(ax, "a")

    # b — structural score does not identify low-damage gate deletion.
    ax = axes[0, 1]
    for bundle, color, marker, label in (
        (n4, BLUE_LIGHT, "o", "N=4"),
        (n6, BLUE, "^", "N=6"),
    ):
        frame = bundle["frame"]
        ax.scatter(
            frame.qfi_null_participation_mean,
            frame.single_delete_energy_abs,
            s=8,
            color=color,
            marker=marker,
            alpha=0.43,
            linewidths=0,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("QFI null-space participation")
    ax.set_ylabel(r"True single-gate $|\Delta E|$")
    ax.set_ylim(2e-7, 20)
    correlation_note(ax, n4, n6, "qfi_redundancy_vs_energy_damage")
    ax.legend(loc="lower left", handletextpad=0.2, borderaxespad=0.2)
    paper_axes(ax, xgrid=True)
    panel_label(ax, "b")

    # c — task-conditioned deletion angle predicts true task damage.
    ax = axes[0, 2]
    for bundle, color, marker, label in (
        (n4, RED_LIGHT, "o", "N=4"),
        (n6, RED, "^", "N=6"),
    ):
        frame = bundle["frame"]
        ax.scatter(
            frame.iprune_q_angle,
            frame.single_delete_energy_abs,
            s=8,
            color=color,
            marker=marker,
            alpha=0.43,
            linewidths=0,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("iPrune deletion angle")
    ax.set_ylabel(r"True single-gate $|\Delta E|$")
    ax.set_ylim(2e-7, 20)
    correlation_note(ax, n4, n6, "iprune_q_angle_vs_energy_damage")
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    paper_axes(ax, xgrid=True)
    panel_label(ax, "c")

    # d — our two-axis structural/task redundancy map.
    ax = axes[1, 0]
    all_damage = pd.concat([n4["frame"], n6["frame"]]).single_delete_energy_abs
    norm = LogNorm(vmin=float(all_damage.min()), vmax=float(all_damage.max()))
    for bundle, marker, label in ((n4, "o", "N=4"), (n6, "^", "N=6")):
        frame = bundle["frame"]
        ax.scatter(
            frame.qfi_null_participation_mean,
            frame.iprune_q_angle,
            c=frame.single_delete_energy_abs,
            cmap="viridis",
            norm=norm,
            s=9,
            marker=marker,
            alpha=0.72,
            linewidths=0,
        )
    ax.set_xlabel("QFI null-space participation")
    ax.set_ylabel("iPrune deletion angle")
    ax.text(0.03, 0.96, "structure–task redundancy map", transform=ax.transAxes, va="top", fontsize=5.8)
    cax = inset_axes(ax, width="4%", height="42%", loc="lower right", borderpad=0.8)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax)
    cb.ax.tick_params(labelsize=5.3, width=0.45, length=1.8, pad=1)
    cb.set_label(r"$|\Delta E|$", fontsize=5.5, labelpad=1)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=GRAY, label="N=4", markersize=3.5),
            Line2D([], [], marker="^", linestyle="none", color=GRAY, label="N=6", markersize=3.5),
        ],
        loc="upper right",
        handletextpad=0.2,
    )
    paper_axes(ax, xgrid=True)
    panel_label(ax, "d")

    # e — independent seed-level correlations.
    ax = axes[1, 1]
    keys = [
        "qfi_redundancy_vs_energy_damage",
        "iprune_q_angle_vs_energy_damage",
        "physical_fs_angle_vs_energy_damage",
    ]
    labels = ["QFI", "iPrune", "FS"]
    rng = np.random.default_rng(20260916)
    for index, key in enumerate(keys):
        for bundle, offset, color, marker in (
            (n4, -0.13, BLUE, "o"),
            (n6, 0.13, RED, "^"),
        ):
            mean, lo, hi, values = corr_entry(bundle, key)
            jitter = rng.normal(0, 0.018, len(values))
            ax.scatter(
                np.full(len(values), index + offset) + jitter,
                values,
                s=9,
                color=color,
                marker=marker,
                alpha=0.55,
                linewidths=0,
            )
            ax.errorbar(
                index + offset,
                mean,
                yerr=np.asarray([[mean - lo], [hi - mean]]),
                fmt=marker,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=4.2,
                capsize=2.2,
                linewidth=1.0,
                zorder=4,
            )
    ax.axhline(0, color=GRAY, linestyle=":", linewidth=0.7)
    ax.set_xticks(np.arange(3), labels)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.72, 1.05)
    ax.set_ylabel("Within-seed Spearman ρ")
    ax.text(0.03, 0.04, "points: 12 training seeds\nmean ± seed-bootstrap 95% CI", transform=ax.transAxes, fontsize=5.5, color=GRAY)
    ax.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", linestyle="none", label="N=4"),
            Line2D([], [], color=RED, marker="^", linestyle="none", label="N=6"),
        ],
        loc="center left",
        bbox_to_anchor=(0.02, 0.39),
        handletextpad=0.2,
    )
    paper_axes(ax)
    panel_label(ax, "e")

    # f — matched top-4 gate selection, paired within seed.
    ax = axes[1, 2]
    positions = {"n4_qfi": 0.0, "n4_i": 0.75, "n6_qfi": 1.8, "n6_i": 2.55}
    for bundle, x0, x1, color in (
        (n4, positions["n4_qfi"], positions["n4_i"], BLUE),
        (n6, positions["n6_qfi"], positions["n6_i"], RED),
    ):
        sel = bundle["stats"]["lowest_damage_selection"]
        qfi = np.asarray(sel["qfi_highest_redundancy_mean_damage"]["per_seed"], float)
        iprune = np.asarray(sel["iprune_lowest_angle_mean_damage"]["per_seed"], float)
        for left, right in zip(qfi, iprune):
            ax.plot([x0, x1], [left, right], color=color, alpha=0.28, linewidth=0.65)
        ax.scatter(np.full(len(qfi), x0), qfi, color=color, marker="o", s=10, alpha=0.65, linewidths=0)
        ax.scatter(np.full(len(iprune), x1), iprune, color=color, marker="o", s=10, alpha=0.65, linewidths=0)
        ax.plot([x0 - 0.14, x0 + 0.14], [qfi.mean(), qfi.mean()], color=BLACK, linewidth=1.2)
        ax.plot([x1 - 0.14, x1 + 0.14], [iprune.mean(), iprune.mean()], color=BLACK, linewidth=1.2)
        p = float(sel["paired_wilcoxon_two_sided_p"])
        ax.text((x0 + x1) / 2, 0.96, f"p={p:.4g}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.6)
    ax.set_xticks(
        [positions["n4_qfi"], positions["n4_i"], positions["n6_qfi"], positions["n6_i"]],
        ["QFI", "iPrune", "QFI", "iPrune"],
        rotation=18,
    )
    ax.text(0.375, -0.22, "N=4", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.2)
    ax.text(2.175, -0.22, "N=6", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.2)
    ax.set_xlim(-0.4, 2.95)
    ax.set_ylabel(r"Top-4 mean true $|\Delta E|$")
    ax.text(0.03, 0.04, "paired by trained circuit", transform=ax.transAxes, fontsize=5.7, color=GRAY)
    paper_axes(ax)
    panel_label(ax, "f")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "fig_qfi_iprune_multiscale"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes.flat),
        panel_ids=list("abcdef"),
        row_groups=[["a", "b", "c"], ["d", "e", "f"]],
        column_groups=[["a", "d"], ["b", "e"], ["c", "f"]],
        exclude_axes=[cax],
        json_out=prefix.with_suffix(".alignment.json"),
        overlay_svg=prefix.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    return fig


def save(fig: plt.Figure, output_dir: Path) -> None:
    prefix = output_dir / "fig_qfi_iprune_multiscale"
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})


def write_notes(output_dir: Path) -> None:
    text = """# Figure contract and QA notes

- Core conclusion: iPrune gate deletion angles predict true task damage across N=4 and N=6, whereas full-QFI null-space participation does not identify the same safely removable gates.
- Statistical unit: 12 independently trained circuits per system size; gate rows are clustered within seed and are not treated as independent replicates.
- Data integrity: all 192 N4 and 288 N6 gate rows are shown; no smoothing, interpolation, filtering, or downsampling.
- Panel roles: (a) numerical structure/robustness; (b) structural-score control; (c) primary task-damage relation; (d) structure-task redundancy map; (e) seed-level statistical validation; (f) matched-budget practical consequence.
- Boundary: q-angle and ordinary FS angle remain similar; the figure supports distinction from QFI, not superiority of q deformation over q=1.
- Reviewer risk: N6 QFI rank is tolerance-sensitive; panel a exposes rather than hides this limitation.
"""
    (output_dir / "FIGURE_CONTRACT_AND_QA.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    n4 = read_bundle(args.n4_dir, 4)
    n6 = read_bundle(args.n6_dir, 6)
    fig = make_figure(n4, n6, args.output_dir)
    save(fig, args.output_dir)
    write_notes(args.output_dir)
    plt.close(fig)
    print(f"Saved multiscale QFI/iPrune figure to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
