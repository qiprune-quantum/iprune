from __future__ import annotations

from typing import Any, Mapping


def validate_result_record(record: Mapping[str, Any]) -> None:
    """Reject result rows that mix candidate and actual-circuit semantics."""
    required = {
        "schema_version",
        "pruning_semantics",
        "L_definition",
        "L",
        "positions",
        "gate_count_before",
        "gate_count_after",
        "pruned_positions",
        "kept_positions",
        "actual_gate_prune_ratio",
        "baseline_circuit_signature",
        "selected_circuit_signature",
        "trace_dist_max",
        "rhs_raw",
        "dq_violations_over_replaced",
        "physical_violations_over_replaced",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"Result record is missing required fields: {missing}")

    if record["schema_version"] != 2:
        raise ValueError("V2 fixed results must use schema_version=2.")
    if record["pruning_semantics"] != "identity_gate_deletion":
        raise ValueError("V2 fixed results must describe actual identity-gate deletion.")
    if record["L_definition"] != "actual_deleted_forward_positions":
        raise ValueError("L must count actual deleted forward positions.")

    L = int(record["L"])
    before = int(record["gate_count_before"])
    after = int(record["gate_count_after"])
    positions = int(record["positions"])
    if L != int(record["pruned_positions"]):
        raise ValueError("L and pruned_positions disagree.")
    if before - after != L:
        raise ValueError("gate_count_before - gate_count_after must equal L.")
    if int(record["kept_positions"]) != after:
        raise ValueError("kept_positions must equal the compact circuit gate count.")
    if before != positions:
        raise ValueError("The baseline must contain exactly one gate per position.")

    expected_ratio = L / max(1, positions)
    if abs(float(record["actual_gate_prune_ratio"]) - expected_ratio) > 1e-12:
        raise ValueError("actual_gate_prune_ratio has the wrong denominator.")
    if not 0.0 <= float(record["trace_dist_max"]) <= 1.0 + 1e-12:
        raise ValueError("Pure-state trace distance must lie in [0, 1].")
    if float(record["rhs_raw"]) < 0.0:
        raise ValueError("The theoretical RHS cannot be negative.")
    if int(record["dq_violations_over_replaced"]) != 0:
        raise ValueError("A deleted gate violates the q-angle threshold.")
    if int(record["physical_violations_over_replaced"]) != 0:
        raise ValueError("A deleted gate violates the physical-angle threshold.")
    if float(record["trace_dist_max"]) > float(record["rhs_raw"]) + 1e-8:
        raise ValueError("Empirical trace distance exceeds the matched-circuit RHS.")
    if L == 0 and record["baseline_circuit_signature"] != record["selected_circuit_signature"]:
        raise ValueError("An unpruned circuit must keep the baseline signature.")
