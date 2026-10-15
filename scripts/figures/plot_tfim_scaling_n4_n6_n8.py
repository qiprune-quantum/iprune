from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from iprune.figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment


PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / "n6_n8_lab_bundle_20260916"
BLUE = "#2C6DB2"
RED = "#D45B52"
GREEN = "#4D9A57"


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
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_rows(path: Path) -> list[dict]:
    rows = []
    for f in sorted(path.glob("seed*.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        if payload["meta"].get("selection_geometry") != "physical_fubini_study":
            raise ValueError(f"Expected a physical-only run: {f}")
        if any(abs(float(row["q"]) - 1.0) > 1e-12 for row in payload["results"]):
            raise ValueError(f"Non-unit q in physical-only run: {f}")
        rows.extend(payload["results"])
    if not rows:
        raise ValueError(f"No seed results found: {path}")
    return rows


def mean_ci(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    n = len(a)
    m = float(a.mean())
    if n <= 1:
        return m, 0.0
    sem = float(a.std(ddof=1) / np.sqrt(n))
    return m, float(student_t.ppf(0.975, df=n - 1) * sem)


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope = float(np.dot(x, y) / np.dot(x, x))
    pred = slope * x
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    return slope, r2


def summary(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = defaultdict(list)
    for r in rows:
        if float(r["delta"]) < 2.0:
            grouped[float(r["delta"])].append(
                (float(r["actual_trace_distance_max"]) ** 2,
                 float(r["abs_dE_one_shot_vs_original"]))
            )
    xs, ys, cis = [], [], []
    for d in sorted(grouped):
        x, y = zip(*grouped[d])
        xm, _ = mean_ci(list(x))
        ym, yc = mean_ci(list(y))
        xs.append(xm)
        ys.append(ym)
        cis.append(yc)
    return np.asarray(xs), np.asarray(ys), np.asarray(cis)


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


def make_figure(output_dir: Path, data: dict[int, tuple[Path, str]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.4, 2.8), facecolor="white")
    for n, (path, color) in data.items():
        x, y, ci = summary(load_rows(path))
        slope, r2 = fit_through_origin(x, y)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, y, "o", markersize=2.8, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=0.7, color=color, linewidth=0.9,
                label=f"N={n}, k={slope:.2f}, R²={r2:.2f}")
    ax.set_xlabel("State infidelity, D²")
    ax.set_ylabel("One-shot energy damage, |ΔE|")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=-0.1)
    ax.legend(loc="upper left", handlelength=1.3, borderaxespad=0.2)
    _panel_label(ax, "a")
    fig.tight_layout()
    fig.canvas.draw()
    prefix = output_dir / "fig_tfim_scaling_n4_n6_n8"
    require_matplotlib_panel_alignment(
        fig,
        json_out=prefix.with_suffix(".alignment.json"),
        overlay_svg=prefix.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    return fig


def save(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "fig_tfim_scaling_n4_n6_n8"
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Physical-only N=4,6,8 TFIM scaling plot")
    parser.add_argument("--n4-dir", type=Path, default=BUNDLE / "runs" / "tfim_n4_physical_q1_20260924")
    parser.add_argument("--n6-dir", type=Path, default=BUNDLE / "runs" / "tfim_n6_physical_q1_20260924")
    parser.add_argument("--n8-dir", type=Path, default=BUNDLE / "runs" / "tfim_n8_physical_q1_20260924")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "figures" / "tfim_scaling_physical_q1_20260924")
    args = parser.parse_args()
    configure_style()
    data = {4: (args.n4_dir, BLUE), 6: (args.n6_dir, GREEN), 8: (args.n8_dir, RED)}
    fig = make_figure(args.output_dir, data)
    save(fig, args.output_dir)
    plt.close(fig)
    print(args.output_dir)


if __name__ == "__main__":
    main()
