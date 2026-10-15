"""Clean, minimal APS-style circuit concept figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})

BLUE = "#2166ac"
RED = "#b2182b"
GREEN = "#1b7837"
GRAY = "#4d4d4d"


def _qline(ax, x0, x1, y):
    ax.plot([x0, x1], [y, y], color="black", lw=1.0, zorder=1)


def _rot(ax, x, y, label, color=GRAY, fill=False):
    w, h = 0.085, 0.20
    fc = color if fill else "white"
    tc = "white" if fill else color
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                           facecolor=fc, edgecolor=color, lw=1.0, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
            color=tc, zorder=4)


def _cnot(ax, x, yc, yt):
    ax.plot([x, x], [yt, yc], color="black", lw=1.0, zorder=2)
    ax.add_patch(Circle((x, yc), 0.02, facecolor="black",
                        edgecolor="black", zorder=4))
    r = 0.035
    ax.plot([x - r, x + r], [yt, yt], color="black", lw=1.0, zorder=4)
    ax.plot([x, x], [yt - r, yt + r], color="black", lw=1.0, zorder=4)


def _circuit(ax, hl_color):
    """Two qubits, one CNOT on the left, three rotations on the right."""
    y0, y1 = 0.74, 0.60
    xc = 0.55
    _qline(ax, 0.26, 0.86, y0)
    _qline(ax, 0.26, 0.86, y1)
    # CNOT (control top, target bottom) on the left
    _cnot(ax, 0.33, y0, y1)
    # three rotations on each qubit
    for x in (0.45, 0.55, 0.65):
        for i, y in enumerate((y0, y1)):
            hl = (i == 0 and x == 0.55)
            _rot(ax, x, y, r"$R_j$" if hl else r"$R$",
                 color=hl_color if hl else GRAY, fill=hl)


def _arrow(ax, x, y0, y1, color):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1),
                                 arrowstyle="-|>", mutation_scale=13,
                                 lw=1.5, color=color, zorder=6))


fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# left
ax = axes[0]
ax.text(0.04, 0.96, "(a)  QFI structural redundancy",
        ha="left", va="top", fontsize=11, fontweight="bold")
_circuit(ax, RED)
ax.text(0.5, 0.40, r"$R_j$ is in a QFI null direction",
        ha="center", va="center", fontsize=9.5, color=GRAY)
ax.text(0.5, 0.34, r"remove $R_j$", ha="center", va="center",
        fontsize=10, color="black")
_arrow(ax, 0.5, 0.28, 0.10, RED)
ax.text(0.5, 0.06, "large task damage", ha="center", va="center",
        fontsize=10, fontweight="bold", color=RED)

# right
ax = axes[1]
ax.text(0.04, 0.96, "(b)  iPrune task redundancy",
        ha="left", va="top", fontsize=11, fontweight="bold")
_circuit(ax, GREEN)
ax.text(0.5, 0.40, r"$R_j|\psi_{\mathrm{task}}\rangle\approx|\psi_{\mathrm{task}}\rangle$",
        ha="center", va="center", fontsize=9.5, color=GRAY)
ax.text(0.5, 0.34, r"remove $R_j$", ha="center", va="center",
        fontsize=10, color="black")
_arrow(ax, 0.5, 0.28, 0.10, GREEN)
ax.text(0.5, 0.06, "small task damage", ha="center", va="center",
        fontsize=10, fontweight="bold", color=GREEN)

fig.tight_layout(pad=0.4)
base = Path(__file__).resolve().parents[2] / "figures" / "fig_concept_structural_vs_task"
base.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(base.with_suffix(".svg"), facecolor="white")
fig.savefig(base.with_suffix(".pdf"), facecolor="white")
print("saved", base.with_suffix(".svg"), base.with_suffix(".pdf"))
