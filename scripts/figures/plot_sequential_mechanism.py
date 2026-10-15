"""Sequential hybrid-prefix deletion, drawn as a clean single-qubit scan."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
})

GREEN = "#4D9A57"
GRAY = "#8A8A8A"
LGRAY = "#D7D7D7"


def _rot(ax, x, y, label, deleted=False):
    w, h = 0.085, 0.22
    color = GREEN if deleted else GRAY
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                           facecolor="white", edgecolor=color, lw=1.1,
                           linestyle="--" if deleted else "-", zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
            color=color, zorder=4)


fig, ax = plt.subplots(figsize=(6.4, 2.1))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

yq = 0.48
ax.plot([0.04, 0.96], [yq, yq], color="black", lw=1.0, zorder=1)

xs = [0.16, 0.32, 0.48, 0.64, 0.80]
labels = [r"$R_1$", r"$R_2$", r"$R_3$", r"$R_4$", r"$R_5$"]
deleted = [False, True, False, True, False]

for x, lab, dele in zip(xs, labels, deleted):
    _rot(ax, x, yq, r"$I$" if dele else lab, deleted=dele)

# scan arrow above the circuit
ax.add_patch(FancyArrowPatch((0.08, 0.80), (0.92, 0.80),
                             arrowstyle="-|>", mutation_scale=14,
                             lw=1.1, color=GRAY, zorder=5))
ax.text(0.5, 0.91, "scan in circuit order", ha="center", va="center",
        fontsize=9.5, color="black")

# per-gate decision labels
for x, dele in zip(xs, deleted):
    if dele:
        ax.text(x, 0.66, "delete", ha="center", va="center", fontsize=8,
                color=GREEN, fontweight="bold")
    else:
        ax.text(x, 0.66, "keep", ha="center", va="center", fontsize=8,
                color=GRAY)

ax.text(0.5, 0.12, "accepted deletions become identity and update the prefix",
        ha="center", va="center", fontsize=9.5, color=GREEN)

fig.tight_layout(pad=0.3)
base = Path(__file__).resolve().parents[2] / "figures" / "fig_sequential_mechanism"
base.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(base.with_suffix(".svg"), facecolor="white")
fig.savefig(base.with_suffix(".pdf"), facecolor="white")
print("saved", base.with_suffix(".svg"), base.with_suffix(".pdf"))
