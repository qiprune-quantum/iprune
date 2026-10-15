from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from iprune.figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment


PROJECT = Path(__file__).resolve().parents[2]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load(csv_path: Path) -> dict[tuple[int, int], list[float]]:
    agg = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["layer"]), int(row["qubit"]))
            agg[key].append(float(row["physical_fs_angle"]))
    return agg


def _panel_label(ax, label):
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


def make_figure(output_dir: Path) -> plt.Figure:
    datasets = [
        (4, PROJECT / "runs" / "qfi_iprune_gatelevel_n4_20260916" / "gate_level_results.csv"),
        (6, PROJECT / "runs" / "qfi_iprune_gatelevel_n6_20260916" / "gate_level_results.csv"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.16, top=0.90, wspace=0.32)
    images = []
    for ax, (n, csv_path) in zip(axes, datasets):
        agg = load(csv_path)
        layers = sorted({k[0] for k in agg})
        qubits = sorted({k[1] for k in agg})
        mat = np.full((len(layers), len(qubits)), np.nan)
        for (l, q), vals in agg.items():
            mat[l, q] = float(np.mean(vals))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
        images.append(im)
        ax.set_xticks(range(len(qubits)))
        ax.set_xticklabels([str(q) for q in qubits])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(l) for l in layers])
        ax.set_xlabel("Qubit")
        ax.set_ylabel("Layer")
        ax.set_title(f"N={n}")
        _panel_label(ax, "a" if n == 4 else "b")
    fig.colorbar(images[-1], ax=axes, shrink=0.9, label="Mean iPrune deletion angle")
    fig.canvas.draw()
    prefix = output_dir / "fig_pruning_heatmap_n4_n6"
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


def save(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "fig_pruning_heatmap_n4_n6"
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})


def main() -> None:
    configure_style()
    out = PROJECT / "figures" / "pruning_heatmap_n4_n6_20260917"
    fig = make_figure(out)
    save(fig, out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
