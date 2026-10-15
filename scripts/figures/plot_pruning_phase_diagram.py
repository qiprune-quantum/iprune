"""Haug-style pruning phase diagram: safe / recoverable / failure regions."""

import json
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

BLUE = "#2C6DB2"
RED = "#D45B52"
GREEN = "#4D9A57"
ORANGE = "#E49B45"
GRAY = "#8A8A8A"


PROJECT = Path(__file__).resolve().parents[2]
RDIR = PROJECT / "runs" / "tfim_pruning_transition_formal_dense_20260915"


def collect(field):
    """Return {delta: [values across seeds]}."""
    agg = {}
    for fp in sorted(RDIR.glob("seed*.json")):
        d = json.load(open(fp, encoding="utf-8"))
        for r in d["results"]:
            agg.setdefault(float(r["delta"]), []).append(float(r[field]))
    return agg


one = collect("energy_gap_one_shot_vs_control")
ft = collect("energy_gap_finetuned_vs_control")
deltas = np.array(sorted(one))


def stat(agg):
    m = np.array([np.mean(agg[d]) for d in deltas])
    s = np.array([np.std(agg[d], ddof=1) for d in deltas])
    return m, s


m1, s1 = stat(one)
m2, s2 = stat(ft)

fig, ax = plt.subplots(figsize=(3.5, 2.8))

# three background regions
ax.axvspan(0.2, 0.8, color=GREEN, alpha=0.10, lw=0)
ax.axvspan(0.8, 1.4, color=ORANGE, alpha=0.12, lw=0)
ax.axvspan(1.4, 2.0, color=RED, alpha=0.10, lw=0)
ax.axvline(1.4, color=RED, lw=0.8, ls="--", alpha=0.7)

ax.plot(deltas, m1, "-o", color=RED, ms=3, lw=1.2,
        label="one-shot")
ax.fill_between(deltas, m1 - s1, m1 + s1, color=RED, alpha=0.18, lw=0)
ax.plot(deltas, m2, "-s", color=BLUE, ms=3, lw=1.2,
        label="fine-tuned")
ax.fill_between(deltas, m2 - s2, m2 + s2, color=BLUE, alpha=0.18, lw=0)

ax.set_xlabel(r"threshold $\delta$")
ax.set_ylabel(r"energy gap $|\Delta E|$")
ax.set_yscale("log")
ax.set_xlim(0.2, 2.0)
ax.set_ylim(1e-5, 8)
ax.tick_params(direction="in", top=True, right=True)
for s in ax.spines.values():
    s.set_linewidth(0.7)

ax.text(0.5, 6.5, "safe", ha="center", va="top", fontsize=9, color=GREEN,
        fontweight="bold")
ax.text(1.1, 6.5, "recoverable", ha="center", va="top", fontsize=9,
        color="#b07a20", fontweight="bold")
ax.text(1.7, 6.5, "failure", ha="center", va="top", fontsize=9, color=RED,
        fontweight="bold")
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout(pad=0.4)
base = PROJECT / "figures" / "fig_pruning_phase_diagram"
base.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(base.with_suffix(".svg"), facecolor="white")
fig.savefig(base.with_suffix(".pdf"), facecolor="white")
print("saved", base.with_suffix(".svg"), base.with_suffix(".pdf"))
