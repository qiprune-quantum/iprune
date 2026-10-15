"""Meyer-style flow diagram of the iPrune pipeline."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
})

BLUE = "#2C6DB2"
GREEN = "#4D9A57"
GRAY = "#4d4d4d"
LIGHT = "#eef3f8"


def box(ax, x, y, w, h, text, fc="white", ec=GRAY):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.04",
                                facecolor=fc, edgecolor=ec, lw=1.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5,
            color="black", zorder=4)


def diamond(ax, x, y, w, h, text, fc="white", ec=GRAY):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2),
                          (x - w / 2, y)],
                         closed=True, facecolor=fc, edgecolor=ec, lw=1.2,
                         zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5,
            color="black", zorder=4)


def arrow(ax, xy0, xy1, color=GRAY):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>",
                                 mutation_scale=13, lw=1.2, color=color,
                                 zorder=2))


fig, ax = plt.subplots(figsize=(5.4, 6.4))
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-0.3, 6.8)
ax.axis("off")

# input
box(ax, 0, 6.25, 4.8, 0.9,
    "trained circuit $U(\\theta)$\ncalibration $\\mathcal{D}_\\mathrm{cal}$, tolerance $\\varepsilon$",
    fc=LIGHT, ec=BLUE)
# scan
box(ax, 0, 5.15, 3.6, 0.7, "scan rotations in circuit order")
# compare
box(ax, 0, 4.10, 4.4, 0.7,
    "compare $R_j$ with identity on the hybrid prefix")
# decision
diamond(ax, 0, 3.05, 2.6, 1.1, "angle $\\leq\\varepsilon$?")
# branches
box(ax, -1.55, 1.75, 2.5, 0.9, "$R_j\\to I$\nupdate prefix",
    fc=LIGHT, ec=GREEN)
box(ax, 1.55, 1.75, 1.8, 0.7, "keep $R_j$")
# output
box(ax, 0, 0.55, 5.0, 0.9,
    "compacted circuit + deletion path\n+ trace-distance certificate",
    fc=LIGHT, ec=BLUE)

# arrows
arrow(ax, (0, 5.80), (0, 5.50))
arrow(ax, (0, 4.80), (0, 4.45))
arrow(ax, (0, 3.75), (0, 3.62))
arrow(ax, (-0.6, 2.60), (-1.55, 2.25))
arrow(ax, (0.6, 2.60), (1.55, 2.15))
arrow(ax, (-1.55, 1.30), (0, 1.05))
arrow(ax, (1.55, 1.40), (0, 1.05))

ax.text(-0.75, 2.52, "yes", ha="left", va="center", fontsize=8.5,
        color=GREEN)
ax.text(0.70, 2.52, "no", ha="left", va="center", fontsize=8.5,
        color=GRAY)

fig.tight_layout(pad=0.3)
base = Path(__file__).resolve().parents[2] / "figures" / "fig_flow_diagram"
base.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(base.with_suffix(".svg"), facecolor="white")
fig.savefig(base.with_suffix(".pdf"), facecolor="white")
print("saved", base.with_suffix(".svg"), base.with_suffix(".pdf"))
