import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局样式配置（学术论文风格） =====================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 干净的白色背景
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.edgecolor'] = 'gray'
plt.rcParams['legend.frameon'] = True

CSV_PATH = "./results/qiprune_sweep_results.csv"

df = pd.read_csv(CSV_PATH)

# 转数值
for c in ["delta", "rot_pool_sigma", "replace_ratio_over_candidates", "L", "dE_mean"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 多 seed 平均
group_cols = ["task", "dataset", "delta", "rot_pool_sigma"]
metrics_all = [c for c in ["replace_ratio_over_candidates", "L", "dE_mean"] if c in df.columns]

g = (df.groupby(group_cols, dropna=False)[metrics_all]
     .mean()
     .reset_index())

# 固定的 sigma 顺序
SIGMA_ORDER = [0.001, 0.003, 0.006, 0.01]
DELTA_ORDER = [0.01, 0.02]


def _grouped_bar(ax, sub, metric, sigmas, deltas, annotate=False):
    """
    画分组柱：x=sigma，组内按 delta
    使用纯色填充，无纹理（hatch）
    """
    x = np.arange(len(sigmas))
    width = 0.35  # 调整柱子宽度，避免重叠

    # 使用清晰的配色方案（去除纹理hatch）
    color_map = {
        deltas[0]: '#4A90E2',  # 蓝色 - delta=0.01
        deltas[1]: '#F5A623'  # 橙色 - delta=0.02
    }

    for j, d in enumerate(deltas):
        s = sub[sub["delta"] == d].copy()
        y = []
        for sig in sigmas:
            v = s.loc[np.isclose(s["rot_pool_sigma"], sig), metric]
            y.append(float(v.iloc[0]) if len(v) else np.nan)
        y = np.array(y, dtype=float)

        xpos = x + (j - (len(deltas) - 1) / 2) * width

        # 去除 hatch，使用纯色填充
        bars = ax.bar(xpos, y, width=width,
                      label=f"δ={d:g}",  # 使用希腊字母
                      color=color_map.get(d, '#2ca02c'),
                      edgecolor='black',  # 黑色边框
                      linewidth=0.8,
                      alpha=0.85)

        # 数值标注（replace_ratio 和 L 都标注）
        if annotate:
            for i, b in enumerate(bars):
                h = b.get_height()
                if np.isfinite(h):
                    # 根据指标类型选择格式
                    if metric == "replace_ratio_over_candidates":
                        label_text = f"{h:.3f}"
                        fontsize = 7.5
                        offset = 3
                    else:  # L 指标
                        label_text = f"{int(h)}"
                        fontsize = 8
                        offset = 4

                    # 交错标注位置，避免重叠
                    # 对于相邻的柱子，第二个柱子的标注稍微向上偏移
                    if j == 1:  # delta=0.02 的柱子
                        offset += 8  # 额外上移

                    ax.annotate(label_text,
                                (b.get_x() + b.get_width() / 2, h),
                                textcoords="offset points",
                                xytext=(0, offset),
                                ha="center",
                                fontsize=fontsize,
                                color='black',
                                weight='normal')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}" for s in sigmas])

    # 网格样式
    ax.grid(True, axis="y", linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)  # 网格在柱子后面

    # 隐藏顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def advanced_panel_for_task(task, save_path=None):
    sub_task = g[g["task"] == task].copy()
    datasets = sorted(sub_task["dataset"].dropna().unique().tolist())
    if not datasets:
        print(f"[WARN] No datasets for task={task}")
        return

    # 决定指标行
    candidate_rows = [
        ("replace_ratio_over_candidates", "Replace Ratio"),
        ("L", "Number of Gates Replaced (L)"),
        ("dE_mean", "Mean Energy Deviation (ΔE)"),
    ]

    rows = []
    for metric, ylab in candidate_rows:
        if metric in sub_task.columns and not sub_task[metric].dropna().empty:
            rows.append((metric, ylab))

    nrows = len(rows)
    ncols = len(datasets)

    # 调整图形大小
    fig_w = max(10, 3.5 * ncols)
    fig_h = 2.5 * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    # 手动调整子图间距，为图例留出空间
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08,
                        wspace=0.25, hspace=0.3)

    legend_handles, legend_labels = None, None

    for c, dataset in enumerate(datasets):
        sub = sub_task[sub_task["dataset"] == dataset].copy()

        sigmas_present = sorted(sub["rot_pool_sigma"].dropna().unique().tolist())
        sigmas = [s for s in SIGMA_ORDER if any(np.isclose(sigmas_present, s))]
        if len(sigmas) == 0:
            sigmas = sigmas_present

        deltas_present = sorted(sub["delta"].dropna().unique().tolist())
        deltas = [d for d in DELTA_ORDER if d in deltas_present] or deltas_present

        for r, (metric, ylab) in enumerate(rows):
            ax = axes[r][c]

            # replace_ratio 和 L 都需要标注
            should_annotate = (metric in ["replace_ratio_over_candidates", "L"])

            _grouped_bar(
                ax=ax,
                sub=sub,
                metric=metric,
                sigmas=sigmas,
                deltas=deltas,
                annotate=should_annotate
            )

            # 数据集标题（只在第一行）
            if r == 0:
                ax.set_title(dataset.upper(), fontsize=12, fontweight='bold', pad=8)

            # y轴标签（只在第一列）
            if c == 0:
                ax.set_ylabel(ylab, fontsize=11)

            # x轴标签（只在最后一行）
            if r == nrows - 1:
                ax.set_xlabel("σ (Candidate Pool)", fontsize=11)

            # 统一 replace_ratio 轴范围，为标注留出空间
            if metric == "replace_ratio_over_candidates":
                ax.set_ylim(0, 0.70)
            elif metric == "L":
                # 为 L 指标的标注留出空间
                current_ylim = ax.get_ylim()
                ax.set_ylim(current_ylim[0], current_ylim[1] * 1.08)

            # 收集 legend
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    # 图例放在右上角，垂直排列，更靠右
    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc="upper right",
                   bbox_to_anchor=(0.995, 0.98),  # 更靠右（从0.98改为0.995）
                   frameon=True,
                   fontsize=10,
                   ncol=1,  # 垂直排列
                   columnspacing=1.0,
                   handlelength=1.5,
                   borderpad=0.5,
                   labelspacing=0.5)

    # 保存图片
    if save_path:
        out_png = save_path.format(task=task)
        out_pdf = out_png.replace(".png", ".pdf")

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")

    plt.show()


# 生成图表
for t in sorted(g["task"].dropna().unique().tolist()):
    advanced_panel_for_task(t, save_path="panel_{task}_clean.png")