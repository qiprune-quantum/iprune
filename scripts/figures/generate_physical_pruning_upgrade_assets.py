from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.analyze_existing_statistics import (
    _exact_sign_flip_p,
    _holm_adjust,
    _paired_effect_dz,
    _seed_bootstrap_ci,
    _wilcoxon_p,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TFIM = (
    PROJECT_DIR
    / "results"
    / "supplementary_upgrade_20260807"
    / "tfim_physical_baselines_seed42_46"
    / "physical_gate_baselines_checkpoint.json"
)
DEFAULT_MNIST = (
    PROJECT_DIR
    / "results"
    / "supplementary_upgrade_20260807"
    / "trained_qnn_true_pruning_mnist49_seed42_46"
    / "trained_qnn_pruning_checkpoint.json"
)

METHODS = [
    "qiprune_true_identity",
    "gradient_integrated_energy_matched_k",
    "parameter_circular_magnitude_matched_k",
    "random_rotation_matched_k",
]
LABELS = {
    "qiprune_true_identity": "q-iPrune",
    "gradient_integrated_energy_matched_k": "Gradient",
    "parameter_circular_magnitude_matched_k": "Parameter magnitude",
    "random_rotation_matched_k": "Random",
}
COLORS = {
    "qiprune_true_identity": "#1f4e79",
    "gradient_integrated_energy_matched_k": "#c55a11",
    "parameter_circular_magnitude_matched_k": "#548235",
    "random_rotation_matched_k": "#7f6000",
}
MARKERS = {
    "qiprune_true_identity": "o",
    "gradient_integrated_energy_matched_k": "s",
    "parameter_circular_magnitude_matched_k": "^",
    "random_rotation_matched_k": "D",
}


def _rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not rows:
        raise ValueError(f"No results found in {path}")
    return rows


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def _tfim_analysis(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator, bootstrap_draws: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[Tuple[str, float], List[Mapping[str, Any]]] = defaultdict(list)
    paired: Dict[Tuple[int, float], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["method"]), float(row["delta"]))].append(row)
        paired[(int(row["seed"]), float(row["delta"]))][str(row["method"])] = row

    curves = []
    for method in METHODS:
        for delta in sorted({float(row["delta"]) for row in rows}):
            method_rows = grouped[(method, delta)]
            prune_mean, prune_sd = _mean_sd(
                [float(row["actual_gate_prune_ratio"]) for row in method_rows]
            )
            gap_mean, gap_sd = _mean_sd(
                [abs(float(row["energy_gap_finetuned_vs_control"])) for row in method_rows]
            )
            one_mean, one_sd = _mean_sd(
                [abs(float(row["energy_gap_one_shot_vs_control"])) for row in method_rows]
            )
            curves.append(
                {
                    "method": method,
                    "method_label": LABELS[method],
                    "pruning_scale": delta,
                    "seed_n": len(method_rows),
                    "gate_prune_ratio_mean": prune_mean,
                    "gate_prune_ratio_sd": prune_sd,
                    "one_shot_abs_gap_mean": one_mean,
                    "one_shot_abs_gap_sd": one_sd,
                    "finetuned_abs_gap_mean": gap_mean,
                    "finetuned_abs_gap_sd": gap_sd,
                }
            )

    useful = {
        key: methods
        for key, methods in paired.items()
        if key[1] < 2.0
        and int(methods["qiprune_true_identity"]["matched_delete_count"]) > 0
    }
    q_method = "qiprune_true_identity"
    tests = []
    wins: Dict[str, Dict[str, int]] = {}
    for comparator in METHODS[1:]:
        by_seed: Dict[int, List[float]] = defaultdict(list)
        config_differences = []
        q_wins = comparator_wins = ties = 0
        for (seed, _delta), method_rows in useful.items():
            q_row = method_rows[q_method]
            comparator_row = method_rows[comparator]
            difference = abs(float(q_row["energy_gap_finetuned_vs_control"])) - abs(
                float(comparator_row["energy_gap_finetuned_vs_control"])
            )
            by_seed[seed].append(difference)
            config_differences.append(difference)
            energy_difference = float(q_row["energy_pruned_finetuned"]) - float(
                comparator_row["energy_pruned_finetuned"]
            )
            if energy_difference < -1e-12:
                q_wins += 1
            elif energy_difference > 1e-12:
                comparator_wins += 1
            else:
                ties += 1
        seed_differences = [float(np.mean(by_seed[seed])) for seed in sorted(by_seed)]
        ci_low, ci_high = _seed_bootstrap_ci(
            seed_differences, draws=bootstrap_draws, rng=rng
        )
        tests.append(
            {
                "comparison": f"q-iPrune minus {LABELS[comparator]}",
                "comparator": comparator,
                "metric": "absolute_finetuned_energy_gap_to_control",
                "seed_n": len(seed_differences),
                "paired_configurations": len(config_differences),
                "mean_seed_aggregated_difference": float(np.mean(seed_differences)),
                "cluster_bootstrap_ci95_low": ci_low,
                "cluster_bootstrap_ci95_high": ci_high,
                "paired_effect_dz": _paired_effect_dz(seed_differences),
                "exact_seed_sign_flip_p": _exact_sign_flip_p(seed_differences),
                "configuration_level_wilcoxon_p_sensitivity": _wilcoxon_p(config_differences),
                "direction_note": "negative favors q-iPrune",
            }
        )
        wins[comparator] = {
            "qiprune_wins": q_wins,
            "comparator_wins": comparator_wins,
            "ties": ties,
        }
    _holm_adjust(tests, "exact_seed_sign_flip_p", "holm_adjusted_seed_p")

    method_seed_means = []
    for method in METHODS:
        by_seed: Dict[int, List[float]] = defaultdict(list)
        for (seed, _), method_rows in useful.items():
            by_seed[seed].append(
                abs(float(method_rows[method]["energy_gap_finetuned_vs_control"]))
            )
        seed_means = [float(np.mean(by_seed[seed])) for seed in sorted(by_seed)]
        mean, sd = _mean_sd(seed_means)
        method_seed_means.append(
            {
                "method": method,
                "method_label": LABELS[method],
                "seed_n": len(seed_means),
                "mean_abs_gap_over_useful_region": mean,
                "sd_seed_mean_abs_gap": sd,
            }
        )
    audit = {
        "rows": len(rows),
        "paired_configurations": len(paired),
        "primary_useful_configurations": len(useful),
        "primary_rule": "pruning_scale < 2.0 and matched_delete_count > 0",
        "wins_useful_region": wins,
        "method_seed_means": method_seed_means,
    }
    return curves, tests, audit


def _mnist_analysis(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator, bootstrap_draws: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["delta"])].append(row)
    curves = []
    tests = []
    for delta, delta_rows in sorted(grouped.items()):
        record: Dict[str, Any] = {"pruning_scale": delta, "seed_n": len(delta_rows)}
        fields = {
            "gate_prune_ratio": "actual_gate_prune_ratio",
            "control_accuracy": "acc_full_control_after_extra_steps",
            "one_shot_accuracy": "acc_pruned_one_shot",
            "finetuned_accuracy": "acc_pruned_finetuned",
        }
        for output_name, source_name in fields.items():
            mean, sd = _mean_sd([float(row[source_name]) for row in delta_rows])
            record[f"{output_name}_mean"] = mean
            record[f"{output_name}_sd"] = sd
        differences = [
            float(row["acc_pruned_finetuned"])
            - float(row["acc_full_control_after_extra_steps"])
            for row in delta_rows
        ]
        difference_mean, difference_sd = _mean_sd(differences)
        ci_low, ci_high = _seed_bootstrap_ci(
            differences, draws=bootstrap_draws, rng=rng
        )
        record["finetuned_minus_control_accuracy_mean"] = difference_mean
        record["finetuned_minus_control_accuracy_sd"] = difference_sd
        curves.append(record)
        tests.append(
            {
                "pruning_scale": delta,
                "metric": "finetuned_accuracy_minus_matched_control",
                "seed_n": len(differences),
                "mean_difference": difference_mean,
                "cluster_bootstrap_ci95_low": ci_low,
                "cluster_bootstrap_ci95_high": ci_high,
                "paired_effect_dz": _paired_effect_dz(differences),
                "exact_seed_sign_flip_p": _exact_sign_flip_p(differences),
                "direction_note": "positive favors the pruned circuit",
            }
        )
    _holm_adjust(tests, "exact_seed_sign_flip_p", "holm_adjusted_seed_p")
    audit = {
        "rows": len(rows),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "dataset": "mnist49",
        "safe_operating_summary": "Mean tuned accuracy is preserved through pruning_scale=0.5 (mean total-gate pruning 21.7%); pruning_scale=0.8 loses 4.14 percentage points on average.",
    }
    return curves, tests, audit


def _plot_tfim(curves: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in METHODS:
        rows = sorted(
            [row for row in curves if row["method"] == method],
            key=lambda row: row["pruning_scale"],
        )
        x = 100 * np.asarray([row["gate_prune_ratio_mean"] for row in rows])
        y = np.asarray([row["finetuned_abs_gap_mean"] for row in rows])
        yerr = np.asarray([row["finetuned_abs_gap_sd"] for row in rows])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            markersize=5,
            capsize=3,
        )
    ax.axvspan(45, 52, color="#d9d9d9", alpha=0.6, label="Over-pruning boundary")
    ax.set_xlabel("Total logical-gate pruning ratio (%)")
    ax.set_ylabel("Absolute fine-tuned energy gap to control")
    ax.set_title("TFIM VQE: matched-$K$ physical gate deletion")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_tfim_physical_baselines.png", dpi=300)
    fig.savefig(output_dir / "fig_tfim_physical_baselines.pdf")
    plt.close(fig)


def _plot_mnist(curves: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    rows = sorted(curves, key=lambda row: row["pruning_scale"])
    x = 100 * np.asarray([row["gate_prune_ratio_mean"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    series = [
        ("control_accuracy", "Full control", "#595959", "--"),
        ("one_shot_accuracy", "One-shot pruned", "#c55a11", "-"),
        ("finetuned_accuracy", "Fine-tuned pruned", "#1f4e79", "-"),
    ]
    for prefix, label, color, linestyle in series:
        mean = 100 * np.asarray([row[f"{prefix}_mean"] for row in rows])
        sd = 100 * np.asarray([row[f"{prefix}_sd"] for row in rows])
        axes[0].plot(x, mean, label=label, color=color, linestyle=linestyle, marker="o")
        axes[0].fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12)
    axes[0].set_xlabel("Total logical-gate pruning ratio (%)")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("(a) Accuracy retention")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    difference = 100 * np.asarray(
        [row["finetuned_minus_control_accuracy_mean"] for row in rows]
    )
    difference_sd = 100 * np.asarray(
        [row["finetuned_minus_control_accuracy_sd"] for row in rows]
    )
    axes[1].axhline(0.0, color="#595959", linewidth=1.2, linestyle="--")
    axes[1].errorbar(
        x,
        difference,
        yerr=difference_sd,
        color="#1f4e79",
        marker="o",
        linewidth=1.8,
        capsize=3,
    )
    axes[1].axvspan(0, 28, color="#e2f0d9", alpha=0.35)
    axes[1].set_xlabel("Total logical-gate pruning ratio (%)")
    axes[1].set_ylabel("Fine-tuned $\Delta$accuracy (percentage points)")
    axes[1].set_title("(b) Difference from matched control")
    axes[1].grid(alpha=0.25)
    fig.suptitle("MNIST 4 vs 9: true gate deletion over five seeds", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_mnist49_true_pruning.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_mnist49_true_pruning.pdf", bbox_inches="tight")
    plt.close(fig)


def _latex(
    tfim_tests: Sequence[Mapping[str, Any]],
    tfim_audit: Mapping[str, Any],
    mnist_curves: Sequence[Mapping[str, Any]],
    mnist_tests: Sequence[Mapping[str, Any]],
) -> Tuple[str, str, str]:
    test_lookup = {row["comparator"]: row for row in tfim_tests}
    seed_mean_lookup = {row["method"]: row for row in tfim_audit["method_seed_means"]}
    tfim_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Matched-$K$ physical gate deletion on TFIM VQE in the useful operating region. The error metric is the absolute fine-tuned energy gap to the matched full-circuit control, averaged within each seed and then across five seeds. Lower is better.}",
        r"\label{tab:tfim_physical_baselines}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Mean gap & q-iPrune W/L/T & Exact seed $p$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        summary = seed_mean_lookup[method]
        if method == "qiprune_true_identity":
            comparison = "--"
            p_value = "--"
        else:
            record = tfim_audit["wins_useful_region"][method]
            comparison = f"{record['qiprune_wins']}/{record['comparator_wins']}/{record['ties']}"
            p_value = f"{test_lookup[method]['exact_seed_sign_flip_p']:.4f}"
        tfim_lines.append(
            f"{LABELS[method]} & {summary['mean_abs_gap_over_useful_region']:.3f} $\\pm$ "
            f"{summary['sd_seed_mean_abs_gap']:.3f} & {comparison} & {p_value} \\\\"
        )
    tfim_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    mnist_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{True gate deletion on MNIST 4 vs 9 over five seeds. Accuracy values are percentages.}",
        r"\label{tab:mnist49_true_pruning}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$s_{\mathrm{prune}}$ & Pruned gates & Control acc. & Fine-tuned acc. & $\Delta$acc. \\",
        r"\midrule",
    ]
    for row in mnist_curves:
        mnist_lines.append(
            f"{row['pruning_scale']:.2f} & {100*row['gate_prune_ratio_mean']:.1f} $\\pm$ {100*row['gate_prune_ratio_sd']:.1f} & "
            f"{100*row['control_accuracy_mean']:.2f} & {100*row['finetuned_accuracy_mean']:.2f} & "
            f"{100*row['finetuned_minus_control_accuracy_mean']:+.2f} \\\\"
        )
    mnist_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    physical_random = test_lookup["random_rotation_matched_k"]
    physical_parameter = test_lookup["parameter_circular_magnitude_matched_k"]
    delta_08 = next(row for row in mnist_curves if abs(row["pruning_scale"] - 0.8) < 1e-12)
    text = rf"""\paragraph{{Matched physical-deletion baselines.}}
Figure~\ref{{fig:tfim_physical_baselines}} and Table~\ref{{tab:tfim_physical_baselines}} compare q-iPrune with integrated-gradient, circular parameter-magnitude, and random deletion on the same optimized TFIM circuits, with identical deleted-gate counts and fine-tuning budgets. Across the useful operating region, q-iPrune has the lowest mean absolute energy gap ({seed_mean_lookup['qiprune_true_identity']['mean_abs_gap_over_useful_region']:.3f}). The conservative seed-level exact tests give $p={physical_random['exact_seed_sign_flip_p']:.4f}$ against random deletion and $p={physical_parameter['exact_seed_sign_flip_p']:.4f}$ against parameter magnitude; these results support a stable numerical advantage but are not described as universal significance with only five independent seeds.

\paragraph{{Classification true-pruning curve.}}
Figure~\ref{{fig:mnist49_true_pruning}} and Table~\ref{{tab:mnist49_true_pruning}} report true gate deletion on MNIST 4 vs 9 using disjoint training, pruning-calibration, and test sets. Mean fine-tuned accuracy is preserved through $s_{{\mathrm{{prune}}}}=0.5$, corresponding to {100*next(row for row in mnist_curves if abs(row['pruning_scale']-0.5)<1e-12)['gate_prune_ratio_mean']:.1f}\% total-gate pruning. At $s_{{\mathrm{{prune}}}}=0.8$, the mean pruning ratio reaches {100*delta_08['gate_prune_ratio_mean']:.1f}\% and the mean accuracy loss is {-100*delta_08['finetuned_minus_control_accuracy_mean']:.2f} percentage points, marking the aggressive-pruning regime.
"""
    return "\n".join(tfim_lines), "\n".join(mnist_lines), text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper assets for the physical-pruning upgrade.")
    parser.add_argument("--tfim_checkpoint", type=Path, default=DEFAULT_TFIM)
    parser.add_argument("--mnist_checkpoint", type=Path, default=DEFAULT_MNIST)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=20000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260807)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    tfim_rows = _rows(args.tfim_checkpoint)
    mnist_rows = _rows(args.mnist_checkpoint)
    tfim_curves, tfim_tests, tfim_audit = _tfim_analysis(
        tfim_rows, rng, args.bootstrap_draws
    )
    mnist_curves, mnist_tests, mnist_audit = _mnist_analysis(
        mnist_rows, rng, args.bootstrap_draws
    )

    _plot_tfim(tfim_curves, output_dir)
    _plot_mnist(mnist_curves, output_dir)
    tfim_tex, mnist_tex, results_text = _latex(
        tfim_tests, tfim_audit, mnist_curves, mnist_tests
    )
    _write_text(output_dir / "tab_tfim_physical_baselines.tex", tfim_tex)
    _write_text(output_dir / "tab_mnist49_true_pruning.tex", mnist_tex)
    _write_text(output_dir / "results_text_physical_upgrade.tex", results_text)
    _write_csv(output_dir / "tfim_physical_baseline_curves.csv", tfim_curves)
    _write_csv(output_dir / "tfim_physical_baseline_tests.csv", tfim_tests)
    _write_csv(output_dir / "mnist49_true_pruning_curves.csv", mnist_curves)
    _write_csv(output_dir / "mnist49_true_pruning_tests.csv", mnist_tests)

    folder = output_dir.name
    paper_ready = rf"""% Auto-generated physical-pruning upgrade
\input{{{folder}/results_text_physical_upgrade}}
\input{{{folder}/tab_tfim_physical_baselines}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=0.78\textwidth]{{{folder}/fig_tfim_physical_baselines.pdf}}
  \caption{{Matched-$K$ physical deletion on TFIM VQE over five seeds. Error bars denote one standard deviation. The shaded endpoint is the all-rotation-deleted failure boundary.}}
  \label{{fig:tfim_physical_baselines}}
\end{{figure*}}

\input{{{folder}/tab_mnist49_true_pruning}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=0.94\textwidth]{{{folder}/fig_mnist49_true_pruning.pdf}}
  \caption{{Physical gate deletion and post-pruning fine-tuning on MNIST 4 vs 9 over five seeds. Bands and error bars denote one standard deviation.}}
  \label{{fig:mnist49_true_pruning}}
\end{{figure*}}
"""
    _write_text(output_dir / "paper_ready_physical_upgrade.tex", paper_ready)
    _write_json(
        output_dir / "physical_pruning_upgrade_summary.json",
        {
            "meta": {
                "protocol": "physical_pruning_upgrade_seed_clustered_statistics",
                "tfim_checkpoint": str(args.tfim_checkpoint.resolve()),
                "mnist_checkpoint": str(args.mnist_checkpoint.resolve()),
                "bootstrap_draws": args.bootstrap_draws,
                "bootstrap_seed": args.bootstrap_seed,
            },
            "tfim_curves": tfim_curves,
            "tfim_tests": tfim_tests,
            "tfim_audit": tfim_audit,
            "mnist49_curves": mnist_curves,
            "mnist49_tests": mnist_tests,
            "mnist49_audit": mnist_audit,
        },
    )
    readme = f"""# 物理删门升级结果（2026-08-07）

本目录包含两项新增论文实验：

1. TFIM VQE 上 q-iPrune、真实积分梯度、参数幅值和随机删门的 matched-K 物理比较。
2. MNIST 4 vs 9 上五个随机种子的训练后真实删门与微调曲线。

所有统计以随机种子为独立重复单位。配置级 Wilcoxon 只作为敏感性分析，主检验为种子内聚合后的精确符号翻转检验。

论文插入方式：

```latex
\\input{{{folder}/paper_ready_physical_upgrade}}
```

不要与旧版仅含 q-iPrune/Gradient 的图重复插入；新版 TFIM 四方法图应替换旧梯度对比主图。
"""
    _write_text(output_dir / "README_CN.md", readme)
    print(f"Saved physical-pruning upgrade assets to: {output_dir}")


if __name__ == "__main__":
    main()
