from __future__ import annotations

import unittest

from iprune.result_validation import validate_result_record


def valid_record() -> dict:
    return {
        "schema_version": 2,
        "pruning_semantics": "identity_gate_deletion",
        "L_definition": "actual_deleted_forward_positions",
        "L": 2,
        "positions": 10,
        "gate_count_before": 10,
        "gate_count_after": 8,
        "pruned_positions": 2,
        "kept_positions": 8,
        "actual_gate_prune_ratio": 0.2,
        "baseline_circuit_signature": "baseline",
        "selected_circuit_signature": "selected",
        "trace_dist_max": 0.1,
        "rhs_raw": 0.2,
        "dq_violations_over_replaced": 0,
        "physical_violations_over_replaced": 0,
    }


class ResultValidationTests(unittest.TestCase):
    def test_accepts_consistent_record(self) -> None:
        validate_result_record(valid_record())

    def test_rejects_candidate_count_as_L(self) -> None:
        record = valid_record()
        record["L"] = 7
        with self.assertRaises(ValueError):
            validate_result_record(record)

    def test_rejects_wrong_ratio_denominator(self) -> None:
        record = valid_record()
        record["actual_gate_prune_ratio"] = 0.02
        with self.assertRaises(ValueError):
            validate_result_record(record)

    def test_rejects_drift_from_a_different_circuit_pair(self) -> None:
        record = valid_record()
        record["trace_dist_max"] = 0.3
        with self.assertRaises(ValueError):
            validate_result_record(record)


if __name__ == "__main__":
    unittest.main()
