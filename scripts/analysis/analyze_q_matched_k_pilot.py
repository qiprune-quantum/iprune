from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT = PROJECT_DIR / "results" / "q_sensitivity_upgrade_20260810"
DEFAULT_TFIM = ROOT / "tfim_matched_k" / "checkpoint.json"
DEFAULT_MNIST = ROOT / "mnist49_matched_k_selection" / "checkpoint.json"


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


def _exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(n, count) for count in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * probability))


def _bootstrap_mean_ci(
    values: np.ndarray, draws: int, rng: np.random.Generator
) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _tfim_summary(
    rows: Sequence[Mapping[str, Any]], draws: int, bootstrap_seed: int
) -> List[Dict[str, Any]]:
    baseline = {
        (int(row["seed"]), float(row["delta"])): float(
            row["abs_dE_finetuned_vs_original"]
        )
        for row in rows
        if float(row["qbeta"]) == 0.0
    }
    grouped: Dict[Tuple[float, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["delta"]), float(row["qbeta"]))].append(row)

    summary = []
    rng = np.random.default_rng(bootstrap_seed)
    for delta, qbeta in sorted(grouped):
        group = sorted(grouped[(delta, qbeta)], key=lambda row: int(row["seed"]))
        values = np.asarray(
            [float(row["abs_dE_finetuned_vs_original"]) for row in group]
        )
        differences = np.asarray(
            [
                float(row["abs_dE_finetuned_vs_original"])
                - baseline[(int(row["seed"]), delta)]
                for row in group
            ]
        )
        wins = int(np.sum(differences < -1e-12))
        losses = int(np.sum(differences > 1e-12))
        ties = int(len(differences) - wins - losses)
        ci_low, ci_high = _bootstrap_mean_ci(differences, draws, rng)
        summary.append(
            {
                "delta": delta,
                "qbeta": qbeta,
                "q": float(group[0]["q"]),
                "seeds": len(group),
                "mean_abs_energy_error": float(np.mean(values)),
                "std_abs_energy_error": float(np.std(values, ddof=1)),
                "mean_paired_difference_vs_q1": float(np.mean(differences)),
                "paired_difference_ci95_low": ci_low,
                "paired_difference_ci95_high": ci_high,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "exact_sign_p": _exact_sign_p(wins, losses),
                "selection_changed_seeds": sum(
                    int(row["matched_different_positions_vs_q1"]) > 0
                    for row in group
                ),
                "mean_matched_jaccard_vs_q1": float(
                    np.mean([float(row["matched_jaccard_vs_q1"]) for row in group])
                ),
                "k_shortfall_seeds": sum(
                    int(row["physical_certificate_k_shortfall"]) > 0 for row in group
                ),
            }
        )
    return summary


def _mnist_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[float, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["delta"]), float(row["qbeta"]))].append(row)
    summary = []
    for delta, qbeta in sorted(grouped):
        group = grouped[(delta, qbeta)]
        summary.append(
            {
                "delta": delta,
                "qbeta": qbeta,
                "q": float(group[0]["q"]),
                "seeds": len(group),
                "selection_changed_seeds": sum(
                    int(row["matched_different_positions_vs_q1"]) > 0
                    for row in group
                ),
                "mean_matched_jaccard_vs_q1": float(
                    np.mean([float(row["matched_jaccard_vs_q1"]) for row in group])
                ),
                "native_k_changed_seeds": sum(
                    int(row["native_threshold_delete_count"])
                    != int(row["matched_delete_count"])
                    for row in group
                ),
                "k_shortfall_seeds": sum(
                    int(row["physical_certificate_k_shortfall"]) > 0 for row in group
                ),
            }
        )
    return summary


def _write_tex(path: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{TFIM matched-$K$ pilot comparing each $q$ deformation with $q=1$. Positive paired differences indicate larger error.}",
        r"\label{tab:q_matched_k_tfim_pilot}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$\delta$ & $\beta_q$ & Mean error & Paired diff. & 95\% CI & W/L/T & Changed \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['delta']:.1f} & {row['qbeta']:g} & "
            f"{row['mean_abs_energy_error']:.4f} & "
            f"{row['mean_paired_difference_vs_q1']:+.4f} & "
            f"[{row['paired_difference_ci95_low']:+.4f}, "
            f"{row['paired_difference_ci95_high']:+.4f}] & "
            f"{row['wins']}/{row['losses']}/{row['ties']} & "
            f"{row['selection_changed_seeds']}/{row['seeds']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _plot(
    path_base: Path,
    tfim: Sequence[Mapping[str, Any]],
    mnist: Sequence[Mapping[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    colors = {1.2: "#2878b5", 1.5: "#c45a34"}
    for delta in sorted({float(row["delta"]) for row in tfim}):
        selected = [row for row in tfim if float(row["delta"]) == delta]
        x = [float(row["qbeta"]) for row in selected]
        axes[0].errorbar(
            x,
            [float(row["mean_abs_energy_error"]) for row in selected],
            yerr=[
                float(row["std_abs_energy_error"]) / math.sqrt(float(row["seeds"]))
                for row in selected
            ],
            marker="o",
            capsize=3,
            color=colors[delta],
            label=fr"$\delta={delta:g}$",
        )
        means = np.asarray(
            [float(row["mean_paired_difference_vs_q1"]) for row in selected]
        )
        lower = np.asarray(
            [float(row["paired_difference_ci95_low"]) for row in selected]
        )
        upper = np.asarray(
            [float(row["paired_difference_ci95_high"]) for row in selected]
        )
        axes[1].errorbar(
            x,
            means,
            yerr=np.vstack([means - lower, upper - means]),
            marker="o",
            capsize=3,
            color=colors[delta],
            label=fr"$\delta={delta:g}$",
        )
    axes[0].set_title("TFIM fine-tuned error (mean ± SEM)")
    axes[0].set_xlabel(r"$\beta_q$")
    axes[0].set_ylabel("Absolute energy error")
    axes[0].legend(frameon=False)
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_title("Paired difference vs q=1")
    axes[1].set_xlabel(r"$\beta_q$")
    axes[1].set_ylabel("q error minus q=1 error")
    axes[1].legend(frameon=False)

    for label, rows, linestyle in (
        ("TFIM 10%", [row for row in tfim if float(row["delta"]) == 1.2], "-"),
        ("TFIM 20%", [row for row in tfim if float(row["delta"]) == 1.5], "--"),
        ("MNIST49 12%", [row for row in mnist if float(row["delta"]) == 0.3], "-"),
        ("MNIST49 22%", [row for row in mnist if float(row["delta"]) == 0.5], "--"),
    ):
        axes[2].plot(
            [float(row["qbeta"]) for row in rows],
            [
                float(row["selection_changed_seeds"]) / float(row["seeds"])
                for row in rows
            ],
            marker="o",
            linestyle=linestyle,
            label=label,
        )
    axes[2].set_title("Matched-K set changes")
    axes[2].set_xlabel(r"$\beta_q$")
    axes[2].set_ylabel("Fraction of seeds")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched-K q contribution pilot.")
    parser.add_argument("--tfim_checkpoint", type=Path, default=DEFAULT_TFIM)
    parser.add_argument("--mnist_checkpoint", type=Path, default=DEFAULT_MNIST)
    parser.add_argument("--output_root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap_draws", type=int, default=20000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260810)
    args = parser.parse_args()

    tfim_payload = json.loads(args.tfim_checkpoint.resolve().read_text(encoding="utf-8"))
    mnist_payload = json.loads(args.mnist_checkpoint.resolve().read_text(encoding="utf-8"))
    tfim_summary = _tfim_summary(
        tfim_payload["results"], args.bootstrap_draws, args.bootstrap_seed
    )
    mnist_summary = _mnist_summary(mnist_payload["results"])
    output_root = args.output_root.resolve()
    analysis_dir = output_root / "analysis"
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"

    _atomic_csv(analysis_dir / "tfim_matched_k_paired_summary.csv", tfim_summary)
    _atomic_csv(analysis_dir / "mnist49_matched_k_selection_summary.csv", mnist_summary)
    conclusion = {
        "classification": "result_2_q_changes_selection_without_repeatable_performance_improvement",
        "tfim": "q changes matched-K sets in a minority of seeds; paired effects are mixed and confidence intervals include zero",
        "mnist49": "fixed-certificate matched-K sets remain identical through qbeta=20 for all five reconstructed seeds",
        "independent_q_performance_benefit_supported": False,
        "formal_significance_claim": False,
    }
    _atomic_json(
        analysis_dir / "matched_k_pilot_summary.json",
        {
            "sources": {
                "tfim": str(args.tfim_checkpoint.resolve()),
                "mnist49": str(args.mnist_checkpoint.resolve()),
            },
            "bootstrap": {
                "draws": args.bootstrap_draws,
                "seed": args.bootstrap_seed,
                "unit": "random seed within each task workpoint",
            },
            "tfim_summary": tfim_summary,
            "mnist49_summary": mnist_summary,
            "conclusion": conclusion,
        },
    )
    _write_tex(tables_dir / "tab_q_matched_k_tfim_pilot.tex", tfim_summary)
    _plot(figures_dir / "fig_q_matched_k_pilot", tfim_summary, mnist_summary)
    print(f"Saved analysis: {analysis_dir / 'matched_k_pilot_summary.json'}")
    print(f"Saved figure: {figures_dir / 'fig_q_matched_k_pilot.png'}")
    print(f"Saved table: {tables_dir / 'tab_q_matched_k_tfim_pilot.tex'}")


if __name__ == "__main__":
    main()
