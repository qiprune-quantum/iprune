from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "lambda_noise_grid_20260812" / "tfim"
CHANNEL_LABELS = {
    "depolarizing": "Depolarizing",
    "amplitude_damping": "Amplitude damping",
}


def _bootstrap(values: Sequence[float], seed: int, draws: int = 20000) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _summarize_grid(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    group_columns = ["split", "channel", "noise_strength", "delta", "lambda", "q"]
    for index, (keys, group) in enumerate(data.groupby(group_columns, sort=True)):
        split, channel, strength, delta, lam, q = keys
        item: Dict[str, float | int | str] = {
            "split": split,
            "channel": channel,
            "noise_strength": float(strength),
            "delta": float(delta),
            "lambda": float(lam),
            "q": float(q),
            "seeds": int(group.seed.nunique()),
            "matched_k_mean": float(group.matched_k.mean()),
            "prune_ratio_mean": float(group.actual_gate_prune_ratio.mean()),
        }
        for metric_index, metric in enumerate(("abs_energy_drift", "trace_distance")):
            mean, low, high = _bootstrap(group[metric].to_numpy(float), 20260812 + 100 * index + metric_index)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci_low"] = low
            item[f"{metric}_ci_high"] = high
        rows.append(item)
    return pd.DataFrame(rows)


def _choose_lambdas(calibration: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (channel, strength, delta), group in calibration.groupby(
        ["channel", "noise_strength", "delta"], sort=True
    ):
        means = group.groupby("lambda", sort=True).abs_energy_drift.mean()
        best_value = float(means.min())
        tied = [float(lam) for lam, value in means.items() if np.isclose(value, best_value, atol=1e-12)]
        # Conservative tie break: prefer lambda closest to the undeformed q=1 control.
        selected = max(tied)
        rows.append(
            {
                "channel": channel,
                "noise_strength": float(strength),
                "delta": float(delta),
                "selected_lambda": selected,
                "selected_q": float(np.exp(1.0 - selected)),
                "calibration_abs_energy_drift": best_value,
                "tied_best_lambdas": ",".join(f"{value:g}" for value in sorted(tied)),
            }
        )
    return pd.DataFrame(rows)


def _test_adaptive(test: pd.DataFrame, choices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = test.merge(choices, on=["channel", "noise_strength", "delta"], how="inner")
    selected = selected[np.isclose(selected["lambda"], selected.selected_lambda)].copy()
    control = test[np.isclose(test["lambda"], 1.0)].copy()
    keys = ["seed", "channel", "noise_strength", "delta"]
    paired = selected.merge(
        control[keys + ["abs_energy_drift", "trace_distance"]],
        on=keys,
        suffixes=("_adaptive", "_control"),
        validate="one_to_one",
    )
    paired["energy_difference_adaptive_minus_control"] = (
        paired.abs_energy_drift_adaptive - paired.abs_energy_drift_control
    )
    paired["trace_difference_adaptive_minus_control"] = (
        paired.trace_distance_adaptive - paired.trace_distance_control
    )
    rows = []
    for index, ((channel, strength, delta), group) in enumerate(
        paired.groupby(["channel", "noise_strength", "delta"], sort=True)
    ):
        item: Dict[str, float | int | str] = {
            "channel": channel,
            "noise_strength": float(strength),
            "delta": float(delta),
            "selected_lambda": float(group.selected_lambda.iloc[0]),
            "selected_q": float(group.selected_q.iloc[0]),
            "test_seeds": int(group.seed.nunique()),
        }
        for metric_index, metric in enumerate(
            ("energy_difference_adaptive_minus_control", "trace_difference_adaptive_minus_control")
        ):
            mean, low, high = _bootstrap(group[metric].to_numpy(float), 20261812 + 100 * index + metric_index)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci_low"] = low
            item[f"{metric}_ci_high"] = high
        energy = group.energy_difference_adaptive_minus_control
        item["energy_better"] = int((energy < -1e-12).sum())
        item["energy_worse"] = int((energy > 1e-12).sum())
        item["energy_tie"] = int((energy.abs() <= 1e-12).sum())
        rows.append(item)
    return paired, pd.DataFrame(rows)


def _plot_heatmaps(summary: pd.DataFrame, choices: pd.DataFrame, destination: Path) -> None:
    test = summary[summary.split == "test"]
    deltas = sorted(test.delta.unique())
    channels = ["depolarizing", "amplitude_damping"]
    figure, axes = plt.subplots(len(deltas) * 2, len(channels), figsize=(11, 12), constrained_layout=True)
    for delta_index, delta in enumerate(deltas):
        for channel_index, channel in enumerate(channels):
            group = test[(test.delta == delta) & (test.channel == channel)]
            strengths = sorted(group.noise_strength.unique())
            lambdas = sorted(group["lambda"].unique())
            for metric_index, (metric, label) in enumerate(
                (("abs_energy_drift_mean", "|Energy drift|"), ("trace_distance_mean", "Trace distance"))
            ):
                axis = axes[2 * delta_index + metric_index, channel_index]
                pivot = group.pivot(index="lambda", columns="noise_strength", values=metric).loc[
                    lambdas, strengths
                ]
                image = axis.imshow(pivot.to_numpy(float), aspect="auto", origin="lower", cmap="viridis")
                axis.set_xticks(range(len(strengths)), [f"{value:g}" for value in strengths])
                axis.set_yticks(range(len(lambdas)), [f"{value:g}" for value in lambdas])
                axis.set_xlabel("Noise strength")
                axis.set_ylabel("lambda")
                axis.set_title(f"{CHANNEL_LABELS[channel]}, delta={delta:g}: {label}")
                for row in range(len(lambdas)):
                    for column in range(len(strengths)):
                        value = pivot.iloc[row, column]
                        axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=7,
                                  color="white" if value > np.nanmedian(pivot.to_numpy(float)) else "black")
                selected = choices[(choices.channel == channel) & (choices.delta == delta)]
                for _, choice in selected.iterrows():
                    x = strengths.index(float(choice.noise_strength))
                    y = lambdas.index(float(choice.selected_lambda))
                    axis.scatter([x], [y], marker="s", facecolors="none", edgecolors="red", s=180, linewidths=2)
                figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("TFIM matched-K true deletion: lambda x physical-noise grid\n(red box: calibration-selected lambda; colors: held-out test seeds 47-51)")
    figure.savefig(destination / "fig_lambda_noise_heatmaps.png", dpi=220)
    figure.savefig(destination / "fig_lambda_noise_heatmaps.pdf")
    plt.close(figure)


def _write_summary(
    destination: Path,
    choices: pd.DataFrame,
    adaptive: pd.DataFrame,
    paired: pd.DataFrame,
    noise_semantics: str,
) -> None:
    monotonic = []
    for (channel, delta), group in choices.groupby(["channel", "delta"], sort=True):
        ordered = group.sort_values("noise_strength")
        values = ordered.selected_lambda.to_numpy(float)
        monotonic.append(
            (
                channel,
                float(delta),
                bool(np.all(np.diff(values) <= 1e-12)),
                bool(np.any(np.diff(values) < -1e-12)),
                values,
            )
        )
    # Preserve seed as the independent unit by averaging repeated channel/noise/delta
    # conditions within each held-out seed before bootstrapping.
    seed_means = paired.groupby("seed", sort=True)[
        ["energy_difference_adaptive_minus_control", "trace_difference_adaptive_minus_control"]
    ].mean()
    overall_energy = seed_means.energy_difference_adaptive_minus_control.to_numpy(float)
    overall_trace = seed_means.trace_difference_adaptive_minus_control.to_numpy(float)
    e_mean, e_low, e_high = _bootstrap(overall_energy, 20262812)
    t_mean, t_low, t_high = _bootstrap(overall_trace, 20262813)
    lines = [
        "# TFIM lambda x 真实噪声 matched-K 实验",
        "",
        "更新时间：2026-08-12（Asia/Shanghai）",
        "",
        "## 协议",
        "",
        "- 10 个独立种子（42--51）；42--46 只用于校准选 lambda，47--51 只用于最终测试。",
        "- lambda={0.2,0.5,0.8,1.0}，q(lambda)=exp(1-lambda)。其中 lambda=1 与 q=1 完全相同，是同一个严格对照。",
        "- 退极化强度 {0.001,0.005,0.01,0.05}；振幅阻尼强度 {0.001,0.01,0.05}。",
        "- delta={1.2,1.5}，各 lambda 使用相同 checkpoint、相同物理证书、相同实际删门数 K。",
        "- 指标为噪声基线与噪声删门电路之间的绝对能量漂移、混态 trace distance；未微调。",
        (
            "- 固定噪声预算稳健性协议：删门后仍保留该位置原计划的噪声事件。"
            if noise_semantics == "fixed_scheduled_noise"
            else "- 每个保留门后施加门局部噪声；删门同时移除该门对应的噪声事件。"
        ),
        "",
        "## 校准集选择的 lambda",
        "",
        "| 通道 | 噪声强度 | delta | lambda | q |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in choices.iterrows():
        lines.append(
            f"| {CHANNEL_LABELS[row.channel]} | {row.noise_strength:g} | {row.delta:g} | {row.selected_lambda:g} | {row.selected_q:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 独立测试：自适应 lambda 相对 lambda=1/q=1",
            "",
            "负值代表自适应方案误差更小。",
            "",
            "| 通道 | 噪声强度 | delta | 选定 lambda | 能量差值 [95% CI] | trace 差值 [95% CI] | 胜/负/平 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in adaptive.iterrows():
        lines.append(
            f"| {CHANNEL_LABELS[row.channel]} | {row.noise_strength:g} | {row.delta:g} | {row.selected_lambda:g} | "
            f"{row.energy_difference_adaptive_minus_control_mean:+.4f} "
            f"[{row.energy_difference_adaptive_minus_control_ci_low:+.4f}, {row.energy_difference_adaptive_minus_control_ci_high:+.4f}] | "
            f"{row.trace_difference_adaptive_minus_control_mean:+.4f} "
            f"[{row.trace_difference_adaptive_minus_control_ci_low:+.4f}, {row.trace_difference_adaptive_minus_control_ci_high:+.4f}] | "
            f"{int(row.energy_better)}/{int(row.energy_worse)}/{int(row.energy_tie)} |"
        )
    lines.extend(
        [
            "",
            "## 总体描述性汇总",
            "",
            f"- 跨全部独立测试条件，自适应 lambda 的能量差值为 {e_mean:+.4f} [{e_low:+.4f}, {e_high:+.4f}]。",
            f"- 跨全部独立测试条件，自适应 lambda 的 trace-distance 差值为 {t_mean:+.4f} [{t_low:+.4f}, {t_high:+.4f}]。",
            "- 上述区间先在每个 held-out seed 内跨条件平均，再以 5 个 seed 为独立单位 bootstrap。",
            "",
            "## 单调性核验",
            "",
        ]
    )
    for channel, delta, is_monotone, has_strict_drop, values in monotonic:
        lines.append(
            f"- {CHANNEL_LABELS[channel]}、delta={delta:g}：lambda 序列 "
            f"{', '.join(f'{value:g}' for value in values)}；"
            f"非增条件={'满足' if is_monotone else '不满足'}，"
            f"但随噪声增强出现严格下降={'是' if has_strict_drop else '否'}。"
        )
    lines.extend(
        [
            "",
            "## 表述边界",
            "",
            "只有当独立测试上的配对能量差值稳定为负、区间不跨 0，且 lambda 随噪声强度呈预期趋势时，才支持当前 lambda 映射的噪声自适应价值。否则应把结果报告为负结果或任务依赖现象，不能从校准集热力图直接宣称有效。",
        ]
    )
    (destination / "RESULTS_SUMMARY_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TFIM lambda x noise grid.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    destination = args.results_dir.resolve()
    checkpoint = pd.read_json(destination / "checkpoint.json", typ="series")
    meta = checkpoint["meta"]
    noise_semantics = str(meta.get("noise_semantics", "gate_removed_noise_removed"))
    data = pd.read_csv(destination / "seed_level.csv")
    data["split"] = np.where(data.seed <= 46, "calibration", "test")
    summary = _summarize_grid(data)
    choices = _choose_lambdas(data[data.split == "calibration"])
    paired, adaptive = _test_adaptive(data[data.split == "test"], choices)
    summary.to_csv(destination / "summary_grid.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(destination / "calibration_selected_lambda.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(destination / "adaptive_test_paired_rows.csv", index=False, encoding="utf-8-sig")
    adaptive.to_csv(destination / "adaptive_test_summary.csv", index=False, encoding="utf-8-sig")
    _plot_heatmaps(summary, choices, destination)
    _write_summary(destination, choices, adaptive, paired, noise_semantics)
    print(f"Saved analysis to {destination}")


if __name__ == "__main__":
    main()
