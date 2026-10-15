from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_rows(rows: List[Dict[str, Any]], task: str, dataset: str, n_qubits: int) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("task") != task:
            continue
        if r.get("dataset") != dataset:
            continue
        if int(r.get("n_qubits", -1)) != int(n_qubits):
            continue
        out.append(r)
    return out


def mean_over_seeds(points: List[Dict[str, Any]], key: str) -> float:
    vals = []
    for p in points:
        if key not in p:
            continue
        v = p[key]
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def plot_lines(
        data_by_label: List[Tuple[str, List[float], List[float]]],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: str,
):
    plt.figure()
    for label, xs, ys in data_by_label:
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    in_path = "./results/qiprune_sweep.json"
    payload = load(in_path)
    rows = payload["results"]

    n_qubits = int(payload["meta"]["defaults"]["n_qubits"])

    # 从meta中获取参数网格
    sigmas = [float(s) for s in payload["meta"]["grid"]["sigmas"]]
    deltas = [float(d) for d in payload["meta"]["grid"]["deltas"]]

    # 检测baseline_type是否存在
    baseline_types = ["paper", "random_matchedL"]
    if not any(r.get("baseline_type") for r in rows):
        baseline_types = ["paper"]

    out_dir = "./results/plots"
    os.makedirs(out_dir, exist_ok=True)

    # 从实际数据中提取唯一的L值（作为x轴）
    # L值对应不同的候选池设置
    all_L_values = sorted(list(set(r.get("L") for r in rows if r.get("L") is not None)))

    print(f"发现的L值: {all_L_values}")
    print(f"Deltas: {deltas}")
    print(f"Baseline types: {baseline_types}")

    # -----------------------
    # Classification datasets
    # -----------------------
    for ds in ["mnist49", "fashion_sb", "bas"]:
        ds_rows = filter_rows(rows, task="classification", dataset=ds, n_qubits=n_qubits)

        if not ds_rows:
            print(f"警告: 没有找到数据集 {ds} 的数据")
            continue

        for metric_key, ylabel, fname in [
            ("replace_ratio_over_candidates", "replace_ratio", f"{ds}_replace_ratio.png"),
            ("trace_dist_max", "trace_drift_max", f"{ds}_drift_max.png"),
            ("acc_drop", "accuracy_drop", f"{ds}_acc_drop.png"),
        ]:
            data_by_label: List[Tuple[str, List[float], List[float]]] = []

            # 按 (baseline_type, delta) 组合绘制不同的线
            for bt in baseline_types:
                for d in deltas:
                    # 筛选匹配的数据点
                    filtered = [
                        p for p in ds_rows
                        if p.get("baseline_type", "paper") == bt and abs(float(p["delta"]) - d) < 1e-9
                    ]

                    if not filtered:
                        continue

                    # 按L值排序并提取对应的y值
                    L_to_value = {}
                    for p in filtered:
                        L_val = p.get("L")
                        if L_val is not None and metric_key in p:
                            val = p[metric_key]
                            if val is not None:
                                L_to_value[L_val] = float(val)

                    # 构建x和y列表
                    x_vals = sorted(L_to_value.keys())
                    y_vals = [L_to_value[L] for L in x_vals]

                    if x_vals and y_vals:
                        label = f"{bt}, delta={d}"
                        data_by_label.append((label, x_vals, y_vals))

            if data_by_label:
                plot_lines(
                    data_by_label=data_by_label,
                    title=f"{ds} (n_qubits={n_qubits})",
                    xlabel="L (number of gates replaced)",
                    ylabel=ylabel,
                    save_path=os.path.join(out_dir, fname),
                )

    # ---- VQE ----
    vqe_rows = filter_rows(rows, task="vqe", dataset="tfim", n_qubits=n_qubits)

    if not vqe_rows:
        print("警告: 没有找到VQE任务的数据")
    else:
        for metric_key, ylabel, fname in [
            ("replace_ratio_over_candidates", "replace_ratio", "vqe_replace_ratio.png"),
            ("trace_dist_max", "trace_drift_max", "vqe_drift_max.png"),
            ("dE_max", "dE_max", "vqe_dE_max.png"),
            ("dE_mean", "dE_mean", "vqe_dE_mean.png"),
        ]:
            data_by_label: List[Tuple[str, List[float], List[float]]] = []

            for bt in baseline_types:
                for d in deltas:
                    filtered = [
                        p for p in vqe_rows
                        if p.get("baseline_type", "paper") == bt and abs(float(p["delta"]) - d) < 1e-9
                    ]

                    if not filtered:
                        continue

                    L_to_value = {}
                    for p in filtered:
                        L_val = p.get("L")
                        if L_val is not None and metric_key in p:
                            val = p[metric_key]
                            if val is not None:
                                L_to_value[L_val] = float(val)

                    x_vals = sorted(L_to_value.keys())
                    y_vals = [L_to_value[L] for L in x_vals]

                    if x_vals and y_vals:
                        label = f"{bt}, delta={d}"
                        data_by_label.append((label, x_vals, y_vals))

            if data_by_label:
                plot_lines(
                    data_by_label=data_by_label,
                    title=f"TFIM VQE (n_qubits={n_qubits})",
                    xlabel="L (number of gates replaced)",
                    ylabel=ylabel,
                    save_path=os.path.join(out_dir, fname),
                )

    print("已保存图表到:", out_dir)


if __name__ == "__main__":
    main()