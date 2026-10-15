"""Haug-style figure: QFI does not predict gate damage, iPrune does."""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
})

GRAY = "#8a8a8a"
BLUE = "#2166ac"


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "qfi": float(r["qfi_null_participation_mean"]),
                "angle": float(r["physical_fs_angle"]),
                "damage": float(r["single_delete_energy_abs"]),
                "seed": int(r["seed"]),
            })
    return rows


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def within_seed_spearman(rows, key):
    vals = []
    for seed in sorted({r["seed"] for r in rows}):
        sub = [r for r in rows if r["seed"] == seed]
        x = np.array([r[key] for r in sub])
        y = np.array([r["damage"] for r in sub])
        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
            vals.append(spearman(x, y))
    return float(np.mean(vals)) if vals else float("nan")


project = Path(__file__).resolve().parents[2]
n4 = load(project / "runs" / "qfi_iprune_gatelevel_n4_20260916" / "gate_level_results.csv")
n6 = load(project / "runs" / "qfi_iprune_gatelevel_n6_20260916" / "gate_level_results.csv")

fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2), sharex="col", sharey="row")

for row, (data, N) in enumerate([(n4, 4), (n6, 6)]):
    # QFI panel
    ax = axes[row, 0]
    ax.scatter([r["qfi"] for r in data], [r["damage"] for r in data],
               s=12, color=GRAY, alpha=0.55, lw=0, zorder=3)
    rho = within_seed_spearman(data, "qfi")
    ax.text(0.03, 0.97, rf"$N={N}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(0.03, 0.88, rf"$\rho={rho:.2f}$", transform=ax.transAxes,
            ha="left", va="top", color="black")

    # iPrune panel
    ax = axes[row, 1]
    ax.scatter([r["angle"] for r in data], [r["damage"] for r in data],
               s=12, color=BLUE, alpha=0.55, lw=0, zorder=3)
    rho = within_seed_spearman(data, "angle")
    ax.text(0.03, 0.97, rf"$N={N}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(0.03, 0.88, rf"$\rho={rho:.2f}$", transform=ax.transAxes,
            ha="left", va="top", color="black")

for ax in axes[:, 1]:
    ax.set_yscale("log")

axes[1, 0].set_xlabel("QFI null-space participation", fontsize=10)
axes[1, 1].set_xlabel(r"iPrune deletion angle $\theta$", fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel(r"true gate damage  $|\Delta E|$", fontsize=10)

for ax in axes.flat:
    ax.tick_params(direction="in", top=True, right=True)
    for s in ax.spines.values():
        s.set_linewidth(0.7)

axes[0, 0].set_title("QFI (structural)", fontsize=10)
axes[0, 1].set_title("iPrune (task-conditioned)", fontsize=10)

for ax, lab in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
    ax.text(0.03, 0.97, lab, transform=ax.transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold")

fig.tight_layout(pad=0.6, h_pad=1.2, w_pad=1.0)
base = project / "figures" / "fig_qfi_iprune_predictive"
base.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(base.with_suffix(".svg"), facecolor="white")
fig.savefig(base.with_suffix(".pdf"), facecolor="white")
print("saved", base.with_suffix(".svg"), base.with_suffix(".pdf"))
