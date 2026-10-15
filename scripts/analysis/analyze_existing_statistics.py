from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_TRUE_PRUNING = (
    RESULTS_DIR
    / "supplementary_upgrade_20260806"
    / "true_pruning_vqe_finetune_seed42_46"
    / "true_pruning_finetune_checkpoint.json"
)
DEFAULT_GRADIENT = (
    RESULTS_DIR
    / "supplementary_upgrade_20260806"
    / "vqe_qiprune_vs_real_gradient_seed42_46"
    / "qiprune_vs_gradient_checkpoint.json"
)
DEFAULT_FAIR = [
    RESULTS_DIR
    / "supplementary_upgrade_20260805"
    / "fair_baselines_seed42"
    / "fair_baseline_checkpoint.json",
    RESULTS_DIR
    / "supplementary_upgrade_20260805"
    / "fair_baselines_seed43_46"
    / "fair_baseline_checkpoint.json",
]

QIPRUNE = "qiprune_positionaware"
QIPRUNE_PHYSICAL = "qiprune_true_identity"
GRADIENT_PHYSICAL = "gradient_integrated_energy_matched_k"
METHOD_LABELS = {
    QIPRUNE: "q-iPrune",
    "q1_positionaware_matched_selected_L": "q=1",
    "qiprune_no_prefix_matched_selected_L": "No Prefix",
    "unitary_overlap_matched_selected_L": "Unitary",
    "parameter_euclidean_matched_selected_L": "Parameter",
    "random_reference_matched_selected_L": "Random",
}


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Missing result rows in {path}")
    return rows


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _mean_sd_t_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sample")
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    if array.size > 1:
        half = float(
            stats.t.ppf((1.0 + confidence) / 2.0, df=array.size - 1)
            * sd
            / math.sqrt(array.size)
        )
    else:
        half = 0.0
    return {
        "n": int(array.size),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _seed_bootstrap_ci(
    seed_values: Sequence[float], *, draws: int, rng: np.random.Generator
) -> Tuple[float, float]:
    values = np.asarray(seed_values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_flip_p(seed_differences: Sequence[float]) -> float:
    differences = np.asarray(seed_differences, dtype=np.float64)
    observed = abs(float(np.mean(differences)))
    null = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(differences)):
        null.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def _paired_effect_dz(seed_differences: Sequence[float]) -> float:
    differences = np.asarray(seed_differences, dtype=np.float64)
    sd = float(np.std(differences, ddof=1))
    if sd <= 1e-15:
        return 0.0 if abs(float(np.mean(differences))) <= 1e-15 else math.copysign(math.inf, float(np.mean(differences)))
    return float(np.mean(differences) / sd)


def _wilcoxon_p(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if np.all(np.abs(values) <= 1e-15):
        return 1.0
    return float(stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)


def _holm_adjust(rows: List[Dict[str, Any]], p_key: str, output_key: str) -> None:
    indexed = sorted(
        ((index, float(row[p_key])) for index, row in enumerate(rows)),
        key=lambda item: item[1],
    )
    running = 0.0
    adjusted: Dict[int, float] = {}
    count = len(indexed)
    for rank, (index, p_value) in enumerate(indexed):
        running = max(running, (count - rank) * p_value)
        adjusted[index] = min(1.0, running)
    for index, row in enumerate(rows):
        row[output_key] = adjusted[index]


def _true_pruning_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["delta"])].append(row)
    output = []
    metrics: Dict[str, Callable[[Mapping[str, Any]], float]] = {
        "gate_prune_ratio": lambda row: float(row["actual_gate_prune_ratio"]),
        "one_shot_abs_gap_vs_control": lambda row: abs(float(row["energy_gap_one_shot_vs_control"])),
        "finetuned_abs_gap_vs_control": lambda row: abs(float(row["energy_gap_finetuned_vs_control"])),
        "recovery_fraction_vs_control": lambda row: float(row["recovery_fraction_vs_control"] or 0.0),
        "finetuned_trace_distance": lambda row: float(row["trace_dist_finetuned_vs_control_max"]),
    }
    for delta, delta_rows in sorted(grouped.items()):
        record: Dict[str, Any] = {"pruning_scale": delta, "seeds": len(delta_rows)}
        for metric, getter in metrics.items():
            summary = _mean_sd_t_ci([getter(row) for row in delta_rows])
            for field, value in summary.items():
                if field != "n":
                    record[f"{metric}_{field}"] = value
        output.append(record)
    return output


def _gradient_statistics(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_draws: int, rng: np.random.Generator
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    paired: Dict[Tuple[int, float], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(int(row["seed"]), float(row["delta"]))][str(row["method"])] = row

    all_pairs = []
    useful_pairs = []
    for key, methods in sorted(paired.items()):
        q_row = methods[QIPRUNE_PHYSICAL]
        gradient_row = methods[GRADIENT_PHYSICAL]
        item = {
            "seed": key[0],
            "pruning_scale": key[1],
            "matched_delete_count": int(q_row["matched_delete_count"]),
            "final_energy_difference_q_minus_gradient": float(q_row["energy_pruned_finetuned"]) - float(gradient_row["energy_pruned_finetuned"]),
            "absolute_control_gap_difference_q_minus_gradient": abs(float(q_row["energy_gap_finetuned_vs_control"])) - abs(float(gradient_row["energy_gap_finetuned_vs_control"])),
        }
        all_pairs.append(item)
        if key[1] < 2.0 and item["matched_delete_count"] > 0:
            useful_pairs.append(item)

    wins = sum(row["final_energy_difference_q_minus_gradient"] < -1e-12 for row in all_pairs)
    losses = sum(row["final_energy_difference_q_minus_gradient"] > 1e-12 for row in all_pairs)
    ties = len(all_pairs) - wins - losses

    tests = []
    for metric in [
        "final_energy_difference_q_minus_gradient",
        "absolute_control_gap_difference_q_minus_gradient",
    ]:
        by_seed: Dict[int, List[float]] = defaultdict(list)
        for row in useful_pairs:
            by_seed[int(row["seed"])].append(float(row[metric]))
        seed_differences = [float(np.mean(by_seed[seed])) for seed in sorted(by_seed)]
        ci_low, ci_high = _seed_bootstrap_ci(
            seed_differences, draws=bootstrap_draws, rng=rng
        )
        config_differences = [float(row[metric]) for row in useful_pairs]
        tests.append(
            {
                "comparison": "q-iPrune minus Gradient",
                "metric": metric,
                "independent_unit": "seed",
                "seed_n": len(seed_differences),
                "paired_configurations": len(config_differences),
                "mean_seed_aggregated_difference": float(np.mean(seed_differences)),
                "median_seed_aggregated_difference": float(np.median(seed_differences)),
                "cluster_bootstrap_ci95_low": ci_low,
                "cluster_bootstrap_ci95_high": ci_high,
                "paired_effect_dz": _paired_effect_dz(seed_differences),
                "exact_seed_sign_flip_p": _exact_sign_flip_p(seed_differences),
                "configuration_level_wilcoxon_p_sensitivity": _wilcoxon_p(config_differences),
                "direction_note": "negative favors q-iPrune",
            }
        )
    audit = {
        "all_paired_configurations": len(all_pairs),
        "primary_useful_paired_configurations": len(useful_pairs),
        "primary_rule": "pruning_scale < 2.0 and matched_delete_count > 0",
        "wins_qiprune": wins,
        "wins_gradient": losses,
        "ties": ties,
        "warning": "Thresholds within a seed are correlated; seed-aggregated exact sign-flip tests are primary.",
    }
    return tests, audit


def _fair_metric(row: Mapping[str, Any], metric: str) -> float | None:
    if metric == "absolute_accuracy_change":
        return abs(float(row["acc_drop"])) if row["task"] == "classification" else None
    if metric == "absolute_vqe_energy_change":
        return abs(float(row["dE_max"])) if row["task"] == "vqe" else None
    if metric == "maximum_trace_distance":
        # Keep this aligned with the paper figure: state drift is summarized on
        # the three classification tasks, while TFIM has its own energy metric.
        return float(row["trace_dist_max"]) if row["task"] == "classification" else None
    raise KeyError(metric)


def _fair_statistics(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_draws: int, rng: np.random.Generator
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    metrics = [
        "absolute_accuracy_change",
        "absolute_vqe_energy_change",
        "maximum_trace_distance",
    ]
    methods = sorted({str(row["baseline_type"]) for row in rows})
    summary = []
    for method in methods:
        method_rows = [row for row in rows if row["baseline_type"] == method]
        record: Dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
        }
        for metric in metrics:
            values = [value for row in method_rows if (value := _fair_metric(row, metric)) is not None]
            stats_row = _mean_sd_t_ci(values)
            record[f"{metric}_n"] = stats_row["n"]
            record[f"{metric}_mean"] = stats_row["mean"]
            record[f"{metric}_sd"] = stats_row["sd"]
        summary.append(record)

    key_fields = ("task", "dataset", "seed", "delta", "rot_pool_sigma")
    paired: Dict[Tuple[Any, ...], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[tuple(row[field] for field in key_fields)][str(row["baseline_type"])] = row

    tests = []
    for comparator in methods:
        if comparator == QIPRUNE:
            continue
        for metric in metrics:
            config_differences = []
            by_seed: Dict[int, List[float]] = defaultdict(list)
            for key, method_rows in paired.items():
                q_value = _fair_metric(method_rows[QIPRUNE], metric)
                comparator_value = _fair_metric(method_rows[comparator], metric)
                if q_value is None or comparator_value is None:
                    continue
                difference = q_value - comparator_value
                config_differences.append(difference)
                by_seed[int(key[2])].append(difference)
            seed_differences = [float(np.mean(by_seed[seed])) for seed in sorted(by_seed)]
            ci_low, ci_high = _seed_bootstrap_ci(
                seed_differences, draws=bootstrap_draws, rng=rng
            )
            tests.append(
                {
                    "comparison": f"q-iPrune minus {METHOD_LABELS[comparator]}",
                    "comparator": comparator,
                    "metric": metric,
                    "independent_unit": "seed",
                    "seed_n": len(seed_differences),
                    "paired_configurations": len(config_differences),
                    "mean_seed_aggregated_difference": float(np.mean(seed_differences)),
                    "median_seed_aggregated_difference": float(np.median(seed_differences)),
                    "cluster_bootstrap_ci95_low": ci_low,
                    "cluster_bootstrap_ci95_high": ci_high,
                    "paired_effect_dz": _paired_effect_dz(seed_differences),
                    "exact_seed_sign_flip_p": _exact_sign_flip_p(seed_differences),
                    "configuration_level_wilcoxon_p_sensitivity": _wilcoxon_p(config_differences),
                    "direction_note": "negative favors q-iPrune",
                }
            )
    _holm_adjust(tests, "exact_seed_sign_flip_p", "holm_adjusted_seed_p")
    _holm_adjust(
        tests,
        "configuration_level_wilcoxon_p_sensitivity",
        "holm_adjusted_configuration_p_sensitivity",
    )

    signatures: Dict[Tuple[Any, ...], Dict[str, str]] = defaultdict(dict)
    for row in rows:
        signatures[tuple(row[field] for field in key_fields)][str(row["baseline_type"])] = str(row["selected_circuit_signature"])
    q1_same = sum(
        method_signatures[QIPRUNE] == method_signatures["q1_positionaware_matched_selected_L"]
        for method_signatures in signatures.values()
    )
    audit = {
        "configurations": len(signatures),
        "method_rows": len(rows),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "qiprune_q1_identical_selected_circuit_signatures": q1_same,
        "warning": "Seed-level exact tests are primary; configuration-level Wilcoxon tests are sensitivity analyses because configurations within a seed are correlated.",
    }
    return summary, tests, audit


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    number = float(value)
    if math.isinf(number):
        return r"$\infty$" if number > 0 else r"$-\infty$"
    return f"{number:.{digits}f}"


def _latex(
    true_summary: Sequence[Mapping[str, Any]],
    gradient_tests: Sequence[Mapping[str, Any]],
    gradient_audit: Mapping[str, Any],
    fair_summary: Sequence[Mapping[str, Any]],
    fair_tests: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "% Auto-generated by analyze_existing_statistics.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Seed-level summary of physical gate deletion on TFIM VQE. Values are mean $\pm$ standard deviation over five seeds.}",
        r"\label{tab:true_pruning_statistics}",
        r"\begin{tabular}{ccccc}",
        r"\hline",
        r"$s_{\mathrm{prune}}$ & Pruned gates (\%) & One-shot gap & Fine-tuned gap & Recovery (\%) \\",
        r"\hline",
    ]
    for row in true_summary:
        lines.append(
            f"{row['pruning_scale']:.1f} & "
            f"{100*row['gate_prune_ratio_mean']:.1f} $\\pm$ {100*row['gate_prune_ratio_sd']:.1f} & "
            f"{row['one_shot_abs_gap_vs_control_mean']:.3f} $\\pm$ {row['one_shot_abs_gap_vs_control_sd']:.3f} & "
            f"{row['finetuned_abs_gap_vs_control_mean']:.3f} $\\pm$ {row['finetuned_abs_gap_vs_control_sd']:.3f} & "
            f"{100*row['recovery_fraction_vs_control_mean']:.1f} $\\pm$ {100*row['recovery_fraction_vs_control_sd']:.1f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])

    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Matched-budget candidate-selection summary. Lower is better.}",
            r"\label{tab:fair_baseline_statistics}",
            r"\begin{tabular}{lccc}",
            r"\hline",
            r"Method & $|\Delta\mathrm{Acc}|$ & $|\Delta E|$ & Max. trace distance \\",
            r"\hline",
        ]
    )
    order = [QIPRUNE, "q1_positionaware_matched_selected_L", "qiprune_no_prefix_matched_selected_L", "unitary_overlap_matched_selected_L", "parameter_euclidean_matched_selected_L", "random_reference_matched_selected_L"]
    lookup = {row["method"]: row for row in fair_summary}
    for method in order:
        row = lookup[method]
        lines.append(
            f"{row['method_label']} & {row['absolute_accuracy_change_mean']:.6f} & "
            f"{row['absolute_vqe_energy_change_mean']:.6f} & {row['maximum_trace_distance_mean']:.6f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])

    lines.append(r"\paragraph{Paired statistical analysis.}")
    energy_test = next(row for row in gradient_tests if row["metric"] == "final_energy_difference_q_minus_gradient")
    gap_test = next(row for row in gradient_tests if row["metric"] == "absolute_control_gap_difference_q_minus_gradient")
    lines.append(
        "Across the matched physical-deletion configurations, q-iPrune records "
        f"{gradient_audit['wins_qiprune']} wins, {gradient_audit['wins_gradient']} losses, "
        f"and {gradient_audit['ties']} ties. After excluding the all-rotation-deleted boundary and zero-deletion cases, "
        "the seed-aggregated exact sign-flip test gives "
        f"$p={energy_test['exact_seed_sign_flip_p']:.4f}$ for final energy and "
        f"$p={gap_test['exact_seed_sign_flip_p']:.4f}$ for the absolute gap to the matched control. "
        "The corresponding configuration-level Wilcoxon values are reported only as sensitivity analyses because thresholds within a seed are correlated."
    )

    random_tests = [row for row in fair_tests if row["comparator"] == "random_reference_matched_selected_L"]
    lines.append(
        " In the candidate-selection baselines, the conservative seed-level family-wise tests do not reach the conventional 0.05 threshold with only five independent seeds; "
        "the numerical advantage over random selection is therefore reported with effect estimates and confidence intervals rather than as universal statistical dominance."
    )
    lines.append("")
    lines.append("% Random-baseline seed-level p-values: " + ", ".join(f"{row['metric']}={row['exact_seed_sign_flip_p']:.4f}" for row in random_tests))
    return "\n".join(lines) + "\n"


def _readme(
    gradient_audit: Mapping[str, Any], fair_audit: Mapping[str, Any]
) -> str:
    return f"""# 现有结果正式统计说明

本目录由 `analyze_existing_statistics.py` 从已有 JSON 检查点生成，不会重新训练，也不会修改原始结果。

## 统计口径

- 独立重复单位为随机种子，而不是同一种子下的每个阈值或参数配置。
- 主检验先在种子内聚合，再对 5 个种子执行精确配对符号翻转检验。
- 95% 置信区间通过种子簇 bootstrap 计算。
- 配置级 Wilcoxon 仅作为敏感性分析，不能解释为 24 或 160 个完全独立重复。
- 公平基线的多重比较使用 Holm 校正。
- 原候选池敏感性实验只有 seed 42，因此保留为描述性机制验证，不对其伪造多种子推断统计。

## 数据审计

- 梯度物理删门：{gradient_audit['all_paired_configurations']} 组配对配置，主分析使用 {gradient_audit['primary_useful_paired_configurations']} 组有效配置。
- 胜负统计：q-iPrune {gradient_audit['wins_qiprune']} 胜，梯度 {gradient_audit['wins_gradient']} 胜，{gradient_audit['ties']} 平。
- 公平候选基线：{fair_audit['configurations']} 个配置、{fair_audit['method_rows']} 条方法结果。
- q-iPrune 与 q=1 的最终线路签名相同：{fair_audit['qiprune_q1_identical_selected_circuit_signatures']}/{fair_audit['configurations']}。

## 文件

- `true_pruning_by_scale.csv`：真实删门逐阈值均值、标准差和 t 置信区间。
- `gradient_paired_tests.csv`：梯度对比的种子级主检验和配置级敏感性分析。
- `fair_baseline_summary.csv`：六种候选选择方法的汇总指标。
- `fair_baseline_paired_tests.csv`：公平基线的配对检验与 Holm 校正。
- `statistical_results.tex`：可供论文引用的 LaTeX 表格和统计表述。
- `statistics_summary.json`：完整机器可读结果。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal statistics for saved q-iPrune checkpoints.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--true_pruning", type=Path, default=DEFAULT_TRUE_PRUNING)
    parser.add_argument("--gradient", type=Path, default=DEFAULT_GRADIENT)
    parser.add_argument("--fair", nargs="+", type=Path, default=DEFAULT_FAIR)
    parser.add_argument("--bootstrap_draws", type=int, default=20000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260807)
    args = parser.parse_args()

    rng = np.random.default_rng(args.bootstrap_seed)
    true_rows = _read_rows(args.true_pruning)
    gradient_rows = _read_rows(args.gradient)
    fair_rows = list(itertools.chain.from_iterable(_read_rows(path) for path in args.fair))

    true_summary = _true_pruning_summary(true_rows)
    gradient_tests, gradient_audit = _gradient_statistics(
        gradient_rows, bootstrap_draws=args.bootstrap_draws, rng=rng
    )
    fair_summary, fair_tests, fair_audit = _fair_statistics(
        fair_rows, bootstrap_draws=args.bootstrap_draws, rng=rng
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "true_pruning_by_scale.csv", true_summary)
    _atomic_csv(output_dir / "gradient_paired_tests.csv", gradient_tests)
    _atomic_csv(output_dir / "fair_baseline_summary.csv", fair_summary)
    _atomic_csv(output_dir / "fair_baseline_paired_tests.csv", fair_tests)
    _atomic_text(
        output_dir / "statistical_results.tex",
        _latex(true_summary, gradient_tests, gradient_audit, fair_summary, fair_tests),
    )
    _atomic_text(output_dir / "README_CN.md", _readme(gradient_audit, fair_audit))
    _atomic_json(
        output_dir / "statistics_summary.json",
        {
            "meta": {
                "protocol": "seed_clustered_formal_statistics_from_saved_checkpoints",
                "independent_unit": "seed",
                "bootstrap_draws": args.bootstrap_draws,
                "bootstrap_seed": args.bootstrap_seed,
                "inputs": {
                    "true_pruning": str(args.true_pruning.resolve()),
                    "gradient": str(args.gradient.resolve()),
                    "fair": [str(path.resolve()) for path in args.fair],
                },
            },
            "true_pruning": true_summary,
            "gradient_tests": gradient_tests,
            "gradient_audit": gradient_audit,
            "fair_baseline_summary": fair_summary,
            "fair_baseline_tests": fair_tests,
            "fair_baseline_audit": fair_audit,
        },
    )
    print(f"Saved formal statistics to: {output_dir}")


if __name__ == "__main__":
    main()
