from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_DIR / "results" / "lambda_noise_grid_20260812"
LAMBDA_QBETA = {
    1.0: 0.0,
    0.8: 6.6666666667,
    0.5: 16.6666666667,
    0.2: 26.6666666667,
}


def _mnist_audit(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "mnist49_selection" / "rows").glob("mnist49_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = pd.DataFrame(payload["results"])
        for lam, qbeta in LAMBDA_QBETA.items():
            subset = result[result.qbeta.sub(qbeta).abs() < 1e-6]
            for _, row in subset.iterrows():
                rows.append(
                    {
                        "seed": int(row.seed),
                        "delta": float(row.delta),
                        "lambda": lam,
                        "q": float(row.q),
                        "matched_delete_count": int(row.matched_delete_count),
                        "matched_different_positions_vs_q1": int(
                            row.matched_different_positions_vs_q1
                        ),
                        "native_threshold_delete_count": int(
                            row.native_threshold_delete_count
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the lambda-noise experiment family.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    mnist = _mnist_audit(root)
    mnist.to_csv(root / "mnist49_selection_audit.csv", index=False, encoding="utf-8-sig")

    primary = pd.read_csv(root / "tfim" / "adaptive_test_summary.csv")
    fixed = pd.read_csv(root / "tfim_fixed_noise_budget" / "adaptive_test_summary.csv")
    mnist_changed = int((mnist.matched_different_positions_vs_q1 > 0).sum())
    mnist_total = len(mnist)
    lines = [
        "# q-iPrune 新补实验总汇：lambda 噪声网格",
        "",
        "更新时间：2026-08-12（Asia/Shanghai）",
        "",
        "## 一句话结论",
        "",
        "新的 matched-K 真实删门实验没有支持当前 lambda 到 q 映射的噪声自适应收益：TFIM 独立测试中，校准选择的 lambda 平均不优于 lambda=1/q=1；MNIST49 的四个 lambda 在所有检查点都选择了完全相同的门。",
        "",
        "## TFIM",
        "",
        "- 10 个独立种子；42--46 校准、47--51 独立测试。",
        "- lambda={0.2,0.5,0.8,1.0}；退极化 4 个强度，振幅阻尼 3 个强度；两个真实删门预算。",
        "- 各 lambda 保持相同 checkpoint、相同物理可行集合和相同 K。",
        "- 主协议和固定噪声预算稳健性协议都已完成，共 1120 个种子级条件行。",
        "- 主协议的完整数值、热力图和独立测试结果见 tfim/RESULTS_SUMMARY_CN.md。",
        "- 固定噪声预算结果见 tfim_fixed_noise_budget/RESULTS_SUMMARY_CN.md。",
        "",
        "## MNIST 4 vs 9 停止审计",
        "",
        f"- 共检查 {mnist_total} 个 seed x delta x lambda 条件，matched-K 集合变化 {mnist_changed}/{mnist_total}。",
        "- 因所有 lambda 的删除集合与 lambda=1 完全相同，噪声准确率和 trace distance 必然相同；按无信息增益停止条件，没有运行昂贵的 8-qubit mixed-state 重复评估。",
        "- 原生阈值下少数 lambda 会改变 K，但这混入删门数量差异，不能用来证明选门质量或噪声自适应。",
        "",
        "## 方法学修正",
        "",
        "- 按当前公式 q(lambda)=exp(1-lambda)，lambda=1 与 q=1 是同一方法，不能作为两个独立基线。",
        "- 自适应 lambda 在校准种子上选择，最终只在未参与选择的种子上报告，避免测试集挑点。",
        "- 最优 lambda 在各删门预算下随噪声强度保持恒定，没有出现‘噪声越强，lambda 越小’的严格下降趋势。",
        "",
        "## 论文可用表述",
        "",
        "在相同删门预算和物理证书下，我们扫描 lambda={0.2,0.5,0.8,1.0} 并在独立随机种子上评估两类门局部噪声。校准选择的 lambda 在 TFIM held-out 测试上未优于 undeformed lambda=1/q=1，对固定噪声预算的稳健性检查得到相同结论；MNIST49 中所有 lambda 的 matched-K 门集合完全一致。因此，当前 lambda 到 q 映射应被视为可调建模选择，而非已验证的噪声自适应机制。",
        "",
        "## 不可写的表述",
        "",
        "- 不可写‘噪声越大最优 lambda 越小’。",
        "- 不可写‘自适应 lambda 在高噪声下显著优于 q=1’。",
        "- 不可把 lambda=1 和 q=1 算成两个不同对比方法。",
    ]
    (root / "OVERALL_RESULTS_SUMMARY_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summary to {root}")


if __name__ == "__main__":
    main()
