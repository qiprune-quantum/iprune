from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "results" / "lambda_noise_grid_20260812" / "tfim" / "adaptive_test_summary.csv"
OUTPUT = PROJECT.parent / "Shao_Paper_4_v2" / "generated_lambda_noise_table_20260812"


CHANNELS = {
    "depolarizing": "Depolarizing",
    "amplitude_damping": "Amplitude damping",
}


def estimate(mean: float, low: float, high: float, digits: int) -> str:
    threshold = 0.5 * 10 ** (-digits)
    if abs(mean) < threshold and abs(low) < threshold and abs(high) < threshold:
        return "$0$"
    return (
        f"${mean:+.{digits}f}$ "
        f"$[{low:+.{digits}f},{high:+.{digits}f}]$"
    )


def build_table(data: pd.DataFrame) -> str:
    rows = data[data.delta.eq(1.5)].copy()
    rows["channel_order"] = rows.channel.map({"depolarizing": 0, "amplitude_damping": 1})
    rows = rows.sort_values(["channel_order", "noise_strength"])
    lines = [
        "% Generated from qiprune_pl-mainV2/results/lambda_noise_grid_20260812/tfim/",
        "% Do not edit numerical values by hand; regenerate from the seed-level records.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Held-out evaluation of the noise-calibrated deformation for matched-$K$ TFIM gate deletion at the more aggressive pruning budget ($\delta=1.5$). The calibration seeds 42--46 select $\lambda^*$ independently for each channel and noise strength; all reported results use held-out seeds 47--51. Entries are paired differences, adaptive $\lambda^*$ minus the undeformed control $\lambda=1$ ($q=1$), with 95\% seed-level bootstrap confidence intervals. Negative values favor the adaptive deformation.}",
        r"\label{tab:lambda_noise_adaptive_heldout}",
        r"\small",
        r"\setlength{\tabcolsep}{5.2pt}",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        "Noise channel & Strength & $\\lambda^*$ & $\\Delta |E|$ $[95\\%\\,\\mathrm{CI}]$ & $\\Delta D_{\\mathrm{tr}}$ $[95\\%\\,\\mathrm{CI}]$ & W/L/T \\\\",
        r"\midrule",
    ]
    current = None
    for _, row in rows.iterrows():
        channel = str(row.channel)
        if current is not None and channel != current:
            lines.append(r"\addlinespace[2pt]")
        current = channel
        lines.append(
            f"{CHANNELS[channel]} & ${float(row.noise_strength):g}$ & ${float(row.selected_lambda):g}$ & "
            f"{estimate(float(row.energy_difference_adaptive_minus_control_mean), float(row.energy_difference_adaptive_minus_control_ci_low), float(row.energy_difference_adaptive_minus_control_ci_high), 3)} & "
            f"{estimate(float(row.trace_difference_adaptive_minus_control_mean), float(row.trace_difference_adaptive_minus_control_ci_low), float(row.trace_difference_adaptive_minus_control_ci_high), 4)} & "
            f"{int(row.energy_better)}/{int(row.energy_worse)}/{int(row.energy_tie)} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{6}{p{0.94\textwidth}}{\footnotesize At the lower pruning budget ($\delta=1.2$), all four tested values $\lambda\in\{0.2,0.5,0.8,1.0\}$ selected the same matched-$K$ circuits as $q=1$ for every held-out seed and noise condition, giving exactly zero paired differences. On MNIST 4-vs-9, the matched-$K$ deletion set likewise remained unchanged in all 40 seed--budget--$\lambda$ conditions. W/L/T counts held-out seeds for which the adaptive deformation has lower/higher/equal absolute energy drift than $q=1$.} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def preview_document(table_text: str) -> str:
    return "\n".join(
        [
            r"\documentclass[a4paper]{article}",
            r"\usepackage{booktabs}",
            r"\usepackage{geometry}",
            r"\geometry{margin=0.8in}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            table_text,
            r"\end{document}",
            "",
        ]
    )


def main() -> None:
    data = pd.read_csv(SOURCE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table_text = build_table(data)
    (OUTPUT / "tab_lambda_noise_adaptive_heldout.tex").write_text(table_text, encoding="utf-8")
    (OUTPUT / "preview_lambda_noise_table.tex").write_text(preview_document(table_text), encoding="utf-8")
    print(f"Saved paper table to {OUTPUT}")


if __name__ == "__main__":
    main()
