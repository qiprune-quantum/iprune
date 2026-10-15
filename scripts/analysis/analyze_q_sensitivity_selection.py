from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_DIR
    / "results"
    / "q_sensitivity_upgrade_20260810"
    / "selection_diagnostic"
    / "checkpoint.json"
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _mean(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def _summarize(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["qbeta"])].append(row)
    summary = []
    for qbeta in sorted(grouped):
        group = grouped[qbeta]
        summary.append(
            {
                "qbeta": qbeta,
                "q": float(group[0]["q"]),
                "configurations": len(group),
                "rank_changed_configurations": sum(
                    float(row["spearman_vs_q1"]) < 1.0 - 1e-12 for row in group
                ),
                "matched_k_changed_configurations": sum(
                    int(row["matched_different_positions_vs_q1"]) > 0 for row in group
                ),
                "native_k_changed_configurations": sum(
                    int(row["native_q_threshold_count"]) != int(row["matched_k"])
                    for row in group
                ),
                "mean_matched_jaccard_vs_q1": _mean(group, "matched_jaccard_vs_q1"),
                "minimum_matched_jaccard_vs_q1": min(
                    float(row["matched_jaccard_vs_q1"]) for row in group
                ),
                "mean_spearman_vs_q1": _mean(group, "spearman_vs_q1"),
                "mean_kendall_vs_q1": _mean(group, "kendall_vs_q1"),
                "q_pass_physical_fail_total": sum(
                    int(row["q_pass_physical_fail_count"]) for row in group
                ),
                "physical_pass_q_fail_total": sum(
                    int(row["physical_pass_q_fail_count"]) for row in group
                ),
            }
        )
    return summary


def _summarize_by_dataset(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), float(row["qbeta"]))].append(row)
    summary = []
    for dataset, qbeta in sorted(grouped, key=lambda item: (item[0], item[1])):
        group = grouped[(dataset, qbeta)]
        summary.append(
            {
                "dataset": dataset,
                "qbeta": qbeta,
                "q": float(group[0]["q"]),
                "configurations": len(group),
                "rank_changed_fraction": sum(
                    float(row["spearman_vs_q1"]) < 1.0 - 1e-12 for row in group
                )
                / len(group),
                "matched_k_changed_fraction": sum(
                    int(row["matched_different_positions_vs_q1"]) > 0 for row in group
                )
                / len(group),
                "native_k_changed_fraction": sum(
                    int(row["native_q_threshold_count"]) != int(row["matched_k"])
                    for row in group
                )
                / len(group),
                "mean_matched_jaccard_vs_q1": _mean(group, "matched_jaccard_vs_q1"),
                "mean_spearman_vs_q1": _mean(group, "spearman_vs_q1"),
            }
        )
    return summary


def _write_tex(path: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Selection sensitivity to the $q$ deformation over 160 historical configurations.}",
        r"\label{tab:q_selection_sensitivity}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\beta_q$ & $q$ & Rank changed & Matched-$K$ changed & Native $K$ changed & Mean Jaccard \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['qbeta']:g} & {row['q']:.3f} & "
            f"{row['rank_changed_configurations']}/160 & "
            f"{row['matched_k_changed_configurations']}/160 & "
            f"{row['native_k_changed_configurations']}/160 & "
            f"{row['mean_matched_jaccard_vs_q1']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _plot(
    path_base: Path,
    summary: Sequence[Mapping[str, Any]],
    by_dataset: Sequence[Mapping[str, Any]],
) -> None:
    qbetas = np.asarray([float(row["qbeta"]) for row in summary])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    for dataset in sorted({str(row["dataset"]) for row in by_dataset}):
        selected = [row for row in by_dataset if row["dataset"] == dataset]
        axes[0].plot(
            [float(row["qbeta"]) for row in selected],
            [float(row["matched_k_changed_fraction"]) for row in selected],
            marker="o",
            label=dataset,
        )
    axes[0].set_title("Matched-K set changes")
    axes[0].set_xlabel(r"$\beta_q$")
    axes[0].set_ylabel("Fraction of configurations")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        qbetas,
        [float(row["mean_matched_jaccard_vs_q1"]) for row in summary],
        marker="o",
        label="Matched-K Jaccard",
    )
    axes[1].plot(
        qbetas,
        [float(row["mean_spearman_vs_q1"]) for row in summary],
        marker="s",
        label="Spearman rank corr.",
    )
    axes[1].set_title("Similarity to q=1")
    axes[1].set_xlabel(r"$\beta_q$")
    axes[1].set_ylim(0.94, 1.005)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(
        qbetas,
        [
            float(row["native_k_changed_configurations"])
            / float(row["configurations"])
            for row in summary
        ],
        marker="o",
        color="#b04a5a",
    )
    axes[2].set_title("Native threshold changes K")
    axes[2].set_xlabel(r"$\beta_q$")
    axes[2].set_ylabel("Fraction of configurations")
    axes[2].set_ylim(-0.02, 1.02)

    for axis in axes:
        axis.grid(alpha=0.22)
    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze q selection-sensitivity checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_root", type=Path)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    rows = payload["results"]
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else checkpoint_path.parent.parent.resolve()
    )
    analysis_dir = output_root / "analysis"
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"

    summary = _summarize(rows)
    by_dataset = _summarize_by_dataset(rows)
    _atomic_csv(analysis_dir / "selection_summary_by_q.csv", summary)
    _atomic_csv(analysis_dir / "selection_summary_by_dataset_q.csv", by_dataset)
    _atomic_json(
        analysis_dir / "selection_summary.json",
        {
            "source_checkpoint": str(checkpoint_path),
            "historical_reproduction": {
                "passed": payload["historical_reproduction_passed_configs"],
                "completed": payload["completed_configs"],
            },
            "summary_by_q": summary,
            "summary_by_dataset_q": by_dataset,
            "conclusion_gate": {
                "q_changes_rankings": any(
                    row["rank_changed_configurations"] > 0 for row in summary
                ),
                "q_changes_matched_k_sets": any(
                    row["matched_k_changed_configurations"] > 0 for row in summary
                ),
                "task_performance_benefit_tested": False,
                "classification": "q controls selection rankings and matched-K paths; task benefit remains untested",
            },
        },
    )
    _write_tex(tables_dir / "tab_q_selection_sensitivity.tex", summary)
    _plot(figures_dir / "fig_q_selection_sensitivity", summary, by_dataset)
    print(f"Saved analysis: {analysis_dir}")
    print(f"Saved figure: {figures_dir / 'fig_q_selection_sensitivity.png'}")
    print(f"Saved table: {tables_dir / 'tab_q_selection_sensitivity.tex'}")


if __name__ == "__main__":
    main()
