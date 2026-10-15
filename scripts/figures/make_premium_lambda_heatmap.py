from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "results" / "lambda_noise_grid_20260812"
SOURCE = ROOT / "tfim" / "seed_level.csv"
CHOICES = ROOT / "tfim" / "calibration_selected_lambda.csv"
OUTPUT = ROOT / "teacher_consultation_v2"

REGULAR = fm.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
BOLD = fm.FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")

NAVY = "#10233F"
INK = "#17243A"
SLATE = "#66758A"
GRID = "#DCE3EA"
WHITE = "#FFFFFF"
RED = "#C64E4E"
GREEN = "#18856E"
SELECT = "#F0B429"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": REGULAR.get_name(),
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
        }
    )


def heldout_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(SOURCE)
    data = data[data.seed >= 47].copy()
    keys = ["seed", "channel", "noise_strength", "delta"]
    control = data[np.isclose(data["lambda"], 1.0)].set_index(keys)["abs_energy_drift"]
    data = data.join(control.rename("control_abs_energy_drift"), on=keys)
    data["difference"] = data.abs_energy_drift - data.control_abs_energy_drift
    summary = (
        data.groupby(["channel", "delta", "lambda", "noise_strength"], as_index=False)
        .difference.mean()
        .sort_values(["channel", "delta", "lambda", "noise_strength"])
    )
    choices = pd.read_csv(CHOICES)
    return summary, choices


def panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    choices: pd.DataFrame,
    channel: str,
    delta: float,
    title: str,
    norm: colors.Normalize,
    cmap,
    show_y: bool,
) -> None:
    group = data[(data.channel == channel) & np.isclose(data.delta, delta)]
    strengths = sorted(group.noise_strength.unique())
    lambdas = sorted(group["lambda"].unique(), reverse=True)
    pivot = group.pivot(index="lambda", columns="noise_strength", values="difference").loc[
        lambdas, strengths
    ]
    image = ax.imshow(pivot.to_numpy(float), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(strengths)), [f"{value:g}" for value in strengths], fontproperties=REGULAR)
    ax.set_yticks(range(len(lambdas)), [f"{value:g}" for value in lambdas], fontproperties=REGULAR)
    ax.set_xlabel("噪声强度 p", fontproperties=BOLD, fontsize=9.5)
    if show_y:
        ax.set_ylabel("λ", fontproperties=BOLD, fontsize=10)
    else:
        ax.set_ylabel("")
    ax.set_title(title, loc="left", fontproperties=BOLD, fontsize=12.5, color=NAVY, pad=11)
    ax.tick_params(length=0, labelsize=9.2)

    values = pivot.to_numpy(float)
    for row_index, lam in enumerate(lambdas):
        for column_index, strength in enumerate(strengths):
            value = float(values[row_index, column_index])
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = WHITE if luminance < 0.55 else INK
            label = "0" if abs(value) < 0.0005 else f"{value:+.3f}"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontproperties=BOLD if lam in (1.0, 0.5) else REGULAR,
                fontsize=9,
                color=text_color,
                zorder=3,
            )

    selected = choices[(choices.channel == channel) & np.isclose(choices.delta, delta)]
    for _, row in selected.iterrows():
        if float(row.noise_strength) not in strengths or float(row.selected_lambda) not in lambdas:
            continue
        x = strengths.index(float(row.noise_strength))
        y = lambdas.index(float(row.selected_lambda))
        ax.add_patch(
            Rectangle(
                (x - 0.46, y - 0.46),
                0.92,
                0.92,
                fill=False,
                edgecolor=SELECT,
                linewidth=2.8,
                zorder=4,
            )
        )

    ax.set_xticks(np.arange(-0.5, len(strengths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(lambdas), 1), minor=True)
    ax.grid(which="minor", color=WHITE, linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def validate(fig: plt.Figure, stem: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for artist in fig.findobj(match=lambda item: isinstance(item, Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer=renderer)
        if box.width > 0 and box.height > 0:
            texts.append((artist.get_text().replace("\n", " / "), box))
    outside = []
    for label, box in texts:
        if box.x0 < fig.bbox.x0 - 1 or box.y0 < fig.bbox.y0 - 1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
            outside.append(label)
    overlaps = []
    for i, (left_label, left_box) in enumerate(texts):
        for right_label, right_box in texts[i + 1 :]:
            w = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            h = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if w > 3 and h > 3:
                overlaps.append((left_label, right_label, w * h))
    report = [
        f"figure={stem}",
        f"visible_text_objects={len(texts)}",
        f"out_of_canvas={len(outside)}",
        f"text_overlaps={len(overlaps)}",
    ]
    report.extend(f"OUTSIDE: {label}" for label in outside)
    report.extend(f"OVERLAP area={area:.1f}: {left} <> {right}" for left, right, area in overlaps)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{stem}_layout_qa.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    if outside or overlaps:
        raise RuntimeError(f"Layout QA failed: outside={len(outside)} overlaps={len(overlaps)}")


def create_figure(data: pd.DataFrame, choices: pd.DataFrame) -> None:
    stem = "lambda_noise_heatmap_premium_cn"
    fig = plt.figure(figsize=(14.4, 8.1), facecolor=WHITE)
    fig.add_artist(Rectangle((0, 0.855), 1, 0.145, transform=fig.transFigure, color=NAVY, zorder=0))
    fig.text(0.06, 0.938, "λ × 真实噪声：相对 q=1 的性能差值", ha="left", va="center",
             fontproperties=BOLD, fontsize=21, color=WHITE)
    fig.text(
        0.06,
        0.89,
        "TFIM · held-out seeds 47–51 · matched-K 真实删门 · 单元格 = 自适应方案绝对能量误差 − q=1",
        ha="left",
        va="center",
        fontproperties=REGULAR,
        fontsize=10.2,
        color="#D9E4F2",
    )

    maximum = float(data.difference.max())
    norm = colors.TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    cmap = colors.LinearSegmentedColormap.from_list("premium_diverging", ["#16836E", "#F7F7F5", "#B33D45"])

    positions = [
        [0.07, 0.52, 0.39, 0.245],
        [0.54, 0.52, 0.39, 0.245],
        [0.07, 0.19, 0.39, 0.245],
        [0.54, 0.19, 0.39, 0.245],
    ]
    axes = [fig.add_axes(position) for position in positions]
    panel(axes[0], data, choices, "depolarizing", 1.2, "退极化｜较低删门预算 δ=1.2", norm, cmap, True)
    panel(axes[1], data, choices, "amplitude_damping", 1.2, "振幅阻尼｜较低删门预算 δ=1.2", norm, cmap, False)
    panel(axes[2], data, choices, "depolarizing", 1.5, "退极化｜较高删门预算 δ=1.5", norm, cmap, True)
    image = panel(axes[3], data, choices, "amplitude_damping", 1.5, "振幅阻尼｜较高删门预算 δ=1.5", norm, cmap, False)

    cax = fig.add_axes([0.94, 0.21, 0.012, 0.54])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=8.5, colors=SLATE, length=3)
    colorbar.outline.set_visible(False)
    colorbar.set_label("误差差值", fontproperties=BOLD, fontsize=9.2, color=INK, labelpad=7)

    fig.text(0.07, 0.115, "■", fontproperties=BOLD, fontsize=13, color=SELECT, ha="left", va="center")
    fig.text(0.088, 0.115, "黄色边框：仅用 calibration seeds 42–46 选出的 λ*", fontproperties=REGULAR,
             fontsize=9.5, color=SLATE, ha="left", va="center")
    fig.text(0.54, 0.115, "0 = 与 q=1 完全相同；红色 = 更差；绿色 = 更好", fontproperties=REGULAR,
             fontsize=9.5, color=SLATE, ha="left", va="center")

    fig.add_artist(Rectangle((0.06, 0.035), 0.88, 0.052, transform=fig.transFigure, color="#F4F6F8", zorder=0))
    fig.text(
        0.08,
        0.061,
        "核心发现：全部单元格均为 0 或正值——λ 改变后没有任何 held-out 条件优于 q=1。",
        ha="left",
        va="center",
        fontproperties=BOLD,
        fontsize=11.5,
        color=RED,
    )
    validate(fig, stem)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=320, facecolor=WHITE)
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor=WHITE)
    plt.close(fig)


def main() -> None:
    style()
    data, choices = heldout_grid()
    create_figure(data, choices)
    print(f"Saved premium heatmap to {OUTPUT}")


if __name__ == "__main__":
    main()
