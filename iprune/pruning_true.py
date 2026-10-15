from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from iprune.gates import GateOp, is_identity_rot
from iprune.pennylane_unitary import unitary_of_single_gate
from iprune.qgeometry import task_conditioned_output_angles


def _pure_state_trace_distances(
    left_states: np.ndarray, right_states: np.ndarray
) -> np.ndarray:
    """Return pure-state trace distances with explicit norm normalization.

    Repeated dense-unitary products can perturb a norm at machine precision.
    Normalizing the overlap prevents identical computational paths from being
    reported as a spurious nonzero trace distance when no gate is deleted.
    """
    left_states = np.asarray(left_states, dtype=np.complex128)
    right_states = np.asarray(right_states, dtype=np.complex128)
    if left_states.shape != right_states.shape or left_states.ndim != 2:
        raise ValueError("State batches must have the same two-dimensional shape.")
    overlaps = np.abs(np.sum(np.conj(left_states) * right_states, axis=1))
    denominators = np.linalg.norm(left_states, axis=1) * np.linalg.norm(right_states, axis=1)
    if np.any(denominators <= 0.0):
        raise ValueError("Trace distance is undefined for a zero-norm state.")
    normalized_overlaps = np.clip(overlaps / denominators, 0.0, 1.0)
    # Treat departures from one below double-precision angular resolution as
    # zero distance.  Without this, identical paths can acquire a false
    # O(1e-8) distance after taking sqrt(1 - overlap**2).
    normalized_overlaps = np.where(normalized_overlaps >= 1.0 - 1e-14, 1.0, normalized_overlaps)
    return np.sqrt(np.maximum(0.0, 1.0 - normalized_overlaps**2))


@dataclass
class TruePruneResult:
    selected_circuit: List[GateOp]
    compact_circuit: List[GateOp]
    pruned_positions: int
    kept_positions: int
    gate_count_before: int
    gate_count_after: int
    dq_mean_over_replaced: float
    dq_max_over_replaced: float
    dq_violations_over_replaced: int
    physical_angle_max_over_replaced: float
    physical_trace_dist_max_over_replaced: float
    physical_violations_over_replaced: int


@dataclass
class FixedKPruneResult:
    """Result of one-shot fixed-budget ranking plus a realized-path audit."""

    selected_circuit: List[GateOp]
    compact_circuit: List[GateOp]
    requested_deletion_budget: int
    selected_positions: List[int]
    physical_feasible_positions: List[int]
    budget_shortfall: int
    gate_count_before: int
    gate_count_after: int
    q_scores_by_position: Dict[int, float]
    physical_scores_by_position: Dict[int, float]
    realized_q_angles_by_position: Dict[int, float]
    realized_physical_angles_by_position: Dict[int, float]
    realized_path_trace_sum_mean: float
    realized_path_trace_sum_max: float
    physical_violations_on_realized_path: int


@dataclass
class FixedDeletionPathAudit:
    """Audit a frozen deletion set on an independent state ensemble."""

    selected_positions: List[int]
    q_angle_max_by_position: Dict[int, float]
    physical_angle_max_by_position: Dict[int, float]
    path_trace_sum_mean: float
    path_trace_sum_max: float
    actual_trace_distance_mean: float
    actual_trace_distance_max: float
    physical_threshold_violations: int
    physical_state_position_violations: int
    path_bound_violations: int


@dataclass
class IterativePruneResult:
    """Sequential deletion result with a complete realized-path record."""

    selected_circuit: List[GateOp]
    compact_circuit: List[GateOp]
    accepted_positions: List[int]
    rejected_by_path_budget: List[int]
    gate_count_before: int
    gate_count_after: int
    q_angle_max_by_position: Dict[int, float]
    physical_angle_max_by_position: Dict[int, float]
    path_trace_sum_mean: float
    path_trace_sum_max: float
    actual_trace_distance_mean: float
    actual_trace_distance_max: float
    physical_threshold_violations: int
    path_budget: Optional[float]


def q_iPrune_fixed_k_identity(
    *,
    all_gates: List[GateOp],
    baseline_circuit: List[GateOp],
    n_qubits: int,
    device_name: str,
    task_states: np.ndarray,
    Gq_diag: np.ndarray,
    physical_epsilon: float,
    deletion_budget: int,
    require_full_budget: bool = True,
    tol_identity: float = 0.0,
) -> FixedKPruneResult:
    """Delete a fixed number of rotations using one-shot q ranking.

    Scores are computed on prefixes of the unchanged trained circuit.  The
    physical score defines a q-independent feasible set, and the requested
    number of feasible positions with the smallest q scores are deleted.
    After selection, the function separately audits the realized sequential
    deletion path on hybrid prefixes.  The path audit never changes the set.
    """
    if deletion_budget < 0:
        raise ValueError("deletion_budget must be non-negative.")
    if physical_epsilon < 0.0:
        raise ValueError("physical_epsilon must be non-negative.")

    states = np.asarray(task_states, dtype=np.complex128)
    if states.ndim != 2:
        raise ValueError(f"task_states must have shape (M, dim), got {states.shape}")
    if states.shape[1] != 2**n_qubits:
        raise ValueError(
            f"task_states width must be 2**n_qubits={2**n_qubits}, got {states.shape[1]}"
        )

    for index, gate in enumerate(baseline_circuit):
        if gate.pos_id != index:
            raise ValueError(
                "baseline_circuit must contain one gate per consecutive position; "
                f"found pos_id={gate.pos_id} at index={index}."
            )

    id_by_pos: Dict[int, GateOp] = {}
    for gate in all_gates:
        if is_identity_rot(gate, tol=tol_identity):
            id_by_pos[gate.pos_id] = gate

    unitary_cache: Dict[int, np.ndarray] = {}

    def unitary_of(gate: GateOp) -> np.ndarray:
        key = id(gate)
        if key not in unitary_cache:
            unitary_cache[key] = unitary_of_single_gate(
                gate.apply, n_qubits=n_qubits, device_name=device_name
            )
        return unitary_cache[key]

    # One-shot scores use the unchanged trained-circuit prefixes.
    baseline_prefix_states = states.copy()
    q_scores: Dict[int, float] = {}
    physical_scores: Dict[int, float] = {}
    for baseline_gate in baseline_circuit:
        baseline_unitary = unitary_of(baseline_gate)
        position = int(baseline_gate.pos_id)
        if baseline_gate.gate_type == "Rot" and position in id_by_pos:
            identity_unitary = unitary_of(id_by_pos[position])
            q_angles, physical_angles = task_conditioned_output_angles(
                baseline_unitary,
                identity_unitary,
                baseline_prefix_states,
                Gq_diag,
            )
            q_scores[position] = float(np.max(q_angles))
            physical_scores[position] = float(np.max(physical_angles))
        baseline_prefix_states = (baseline_unitary @ baseline_prefix_states.T).T

    feasible = sorted(
        position
        for position, score in physical_scores.items()
        if score <= physical_epsilon + 1e-12
    )
    if require_full_budget and len(feasible) < deletion_budget:
        raise ValueError(
            "The physical feasible set contains "
            f"{len(feasible)} positions, fewer than deletion_budget={deletion_budget}."
        )

    actual_budget = min(deletion_budget, len(feasible))
    ranked_feasible = sorted(feasible, key=lambda pos: (q_scores[pos], pos))
    selected_positions = sorted(ranked_feasible[:actual_budget])
    selected_set = set(selected_positions)

    selected_circuit = [
        id_by_pos[int(gate.pos_id)] if int(gate.pos_id) in selected_set else gate
        for gate in baseline_circuit
    ]
    compact = [
        gate for gate in selected_circuit if not is_identity_rot(gate, tol=tol_identity)
    ]

    # Realized-path audit uses hybrid prefixes created by earlier deletions.
    hybrid_states = states.copy()
    path_cost_per_state = np.zeros(states.shape[0], dtype=np.float64)
    realized_q: Dict[int, float] = {}
    realized_physical: Dict[int, float] = {}
    physical_violations = 0
    for baseline_gate, selected_gate in zip(baseline_circuit, selected_circuit):
        position = int(baseline_gate.pos_id)
        if position in selected_set:
            q_angles, physical_angles = task_conditioned_output_angles(
                unitary_of(baseline_gate),
                unitary_of(selected_gate),
                hybrid_states,
                Gq_diag,
            )
            realized_q[position] = float(np.max(q_angles))
            realized_physical[position] = float(np.max(physical_angles))
            path_cost_per_state += np.sin(physical_angles)
            if realized_physical[position] > physical_epsilon + 1e-12:
                physical_violations += 1
        hybrid_states = (unitary_of(selected_gate) @ hybrid_states.T).T

    return FixedKPruneResult(
        selected_circuit=selected_circuit,
        compact_circuit=compact,
        requested_deletion_budget=int(deletion_budget),
        selected_positions=selected_positions,
        physical_feasible_positions=feasible,
        budget_shortfall=int(deletion_budget - actual_budget),
        gate_count_before=len(baseline_circuit),
        gate_count_after=len(compact),
        q_scores_by_position=q_scores,
        physical_scores_by_position=physical_scores,
        realized_q_angles_by_position=realized_q,
        realized_physical_angles_by_position=realized_physical,
        realized_path_trace_sum_mean=float(np.mean(path_cost_per_state)),
        realized_path_trace_sum_max=float(np.max(path_cost_per_state)),
        physical_violations_on_realized_path=physical_violations,
    )


def audit_fixed_identity_deletion_path(
    *,
    all_gates: List[GateOp],
    baseline_circuit: List[GateOp],
    selected_positions: List[int],
    n_qubits: int,
    device_name: str,
    task_states: np.ndarray,
    Gq_diag: np.ndarray,
    physical_epsilon: float,
    tol_identity: float = 0.0,
) -> FixedDeletionPathAudit:
    """Audit a preselected identity-deletion path without changing its set."""
    states = np.asarray(task_states, dtype=np.complex128)
    if states.ndim != 2 or states.shape[1] != 2**n_qubits:
        raise ValueError("task_states must have shape (M, 2**n_qubits).")
    selected = {int(position) for position in selected_positions}
    id_by_pos = {
        int(gate.pos_id): gate
        for gate in all_gates
        if is_identity_rot(gate, tol=tol_identity)
    }
    missing = selected - set(id_by_pos)
    if missing:
        raise ValueError(f"Missing identity candidates for positions {sorted(missing)}.")

    unitary_cache: Dict[int, np.ndarray] = {}

    def unitary_of(gate: GateOp) -> np.ndarray:
        key = id(gate)
        if key not in unitary_cache:
            unitary_cache[key] = unitary_of_single_gate(
                gate.apply, n_qubits=n_qubits, device_name=device_name
            )
        return unitary_cache[key]

    baseline_states = states.copy()
    hybrid_states = states.copy()
    path_cost = np.zeros(states.shape[0], dtype=np.float64)
    q_max: Dict[int, float] = {}
    physical_max: Dict[int, float] = {}
    threshold_violations = 0
    state_position_violations = 0
    for index, baseline_gate in enumerate(baseline_circuit):
        if int(baseline_gate.pos_id) != index:
            raise ValueError("baseline_circuit positions must be consecutive.")
        position = int(baseline_gate.pos_id)
        baseline_unitary = unitary_of(baseline_gate)
        selected_gate = id_by_pos[position] if position in selected else baseline_gate
        selected_unitary = unitary_of(selected_gate)
        if position in selected:
            q_angles, physical_angles = task_conditioned_output_angles(
                baseline_unitary,
                selected_unitary,
                hybrid_states,
                Gq_diag,
            )
            q_max[position] = float(np.max(q_angles))
            physical_max[position] = float(np.max(physical_angles))
            path_cost += np.sin(physical_angles)
            threshold_violations += int(
                physical_max[position] > physical_epsilon + 1e-12
            )
            state_position_violations += int(
                np.sum(physical_angles > physical_epsilon + 1e-12)
            )
        baseline_states = (baseline_unitary @ baseline_states.T).T
        hybrid_states = (selected_unitary @ hybrid_states.T).T

    actual = _pure_state_trace_distances(baseline_states, hybrid_states)
    capped_bound = np.minimum(1.0, path_cost)
    bound_violations = int(np.sum(actual > capped_bound + 1e-10))
    return FixedDeletionPathAudit(
        selected_positions=sorted(selected),
        q_angle_max_by_position=q_max,
        physical_angle_max_by_position=physical_max,
        path_trace_sum_mean=float(np.mean(path_cost)),
        path_trace_sum_max=float(np.max(path_cost)),
        actual_trace_distance_mean=float(np.mean(actual)),
        actual_trace_distance_max=float(np.max(actual)),
        physical_threshold_violations=threshold_violations,
        physical_state_position_violations=state_position_violations,
        path_bound_violations=bound_violations,
    )


def q_iPrune_iterative_identity_audited(
    *,
    all_gates: List[GateOp],
    baseline_circuit: List[GateOp],
    n_qubits: int,
    device_name: str,
    task_states: np.ndarray,
    Gq_diag: np.ndarray,
    q_epsilon: float,
    physical_epsilon: float,
    path_budget: Optional[float] = None,
    tol_identity: float = 0.0,
) -> IterativePruneResult:
    """Run the legacy sequential policy and record its realized certificate.

    Rotations are visited in circuit order.  At each position, the candidate
    identity replacement is evaluated on the hybrid prefix generated by all
    earlier accepted deletions.  The q score selects whether a candidate is
    eligible; physical angles provide the certificate.  If ``path_budget`` is
    supplied, a candidate that would make the calibration-path trace sum exceed
    it is rejected.  With ``path_budget=None`` the selection policy is exactly
    the historical per-gate sequential threshold rule, now with an auditable
    cumulative path record.
    """
    if q_epsilon < 0.0 or physical_epsilon < 0.0:
        raise ValueError("Local thresholds must be non-negative.")
    if path_budget is not None and path_budget < 0.0:
        raise ValueError("path_budget must be non-negative when supplied.")
    states = np.asarray(task_states, dtype=np.complex128)
    if states.ndim != 2 or states.shape[1] != 2**n_qubits:
        raise ValueError("task_states must have shape (M, 2**n_qubits).")
    for index, gate in enumerate(baseline_circuit):
        if int(gate.pos_id) != index:
            raise ValueError("baseline_circuit positions must be consecutive.")

    id_by_pos = {
        int(gate.pos_id): gate
        for gate in all_gates
        if is_identity_rot(gate, tol=tol_identity)
    }
    unitary_cache: Dict[int, np.ndarray] = {}

    def unitary_of(gate: GateOp) -> np.ndarray:
        key = id(gate)
        if key not in unitary_cache:
            unitary_cache[key] = unitary_of_single_gate(
                gate.apply, n_qubits=n_qubits, device_name=device_name
            )
        return unitary_cache[key]

    baseline_states = states.copy()
    hybrid_states = states.copy()
    path_cost = np.zeros(states.shape[0], dtype=np.float64)
    selected: List[GateOp] = []
    accepted_positions: List[int] = []
    rejected_by_budget: List[int] = []
    q_maxima: Dict[int, float] = {}
    physical_maxima: Dict[int, float] = {}
    physical_violations = 0

    for baseline_gate in baseline_circuit:
        position = int(baseline_gate.pos_id)
        baseline_unitary = unitary_of(baseline_gate)
        selected_gate = baseline_gate
        if baseline_gate.gate_type == "Rot" and position in id_by_pos:
            identity_gate = id_by_pos[position]
            q_angles, physical_angles = task_conditioned_output_angles(
                baseline_unitary,
                unitary_of(identity_gate),
                hybrid_states,
                Gq_diag,
            )
            q_max = float(np.max(q_angles))
            physical_max = float(np.max(physical_angles))
            candidate_path = path_cost + np.sin(physical_angles)
            if q_max <= q_epsilon + 1e-12 and physical_max <= physical_epsilon + 1e-12:
                if path_budget is None or float(np.max(candidate_path)) <= path_budget + 1e-12:
                    selected_gate = identity_gate
                    accepted_positions.append(position)
                    q_maxima[position] = q_max
                    physical_maxima[position] = physical_max
                    path_cost = candidate_path
                else:
                    rejected_by_budget.append(position)

        if selected_gate is not baseline_gate and physical_maxima[position] > physical_epsilon + 1e-12:
            physical_violations += 1
        selected.append(selected_gate)
        baseline_states = (baseline_unitary @ baseline_states.T).T
        hybrid_states = (unitary_of(selected_gate) @ hybrid_states.T).T

    compact = [gate for gate in selected if not is_identity_rot(gate, tol=tol_identity)]
    actual = _pure_state_trace_distances(baseline_states, hybrid_states)
    capped_path_cost = np.minimum(1.0, path_cost)
    if np.any(actual > capped_path_cost + 1e-10):
        worst = int(np.argmax(actual - capped_path_cost))
        raise AssertionError(
            "Realized iterative path failed its trace-distance bound: "
            f"state_index={worst}, actual={actual[worst]:.16g}, "
            f"bound={capped_path_cost[worst]:.16g}, raw_path={path_cost[worst]:.16g}, "
            f"accepted_positions={accepted_positions}."
        )
    return IterativePruneResult(
        selected_circuit=selected,
        compact_circuit=compact,
        accepted_positions=accepted_positions,
        rejected_by_path_budget=rejected_by_budget,
        gate_count_before=len(baseline_circuit),
        gate_count_after=len(compact),
        q_angle_max_by_position=q_maxima,
        physical_angle_max_by_position=physical_maxima,
        path_trace_sum_mean=float(np.mean(path_cost)),
        path_trace_sum_max=float(np.max(path_cost)),
        actual_trace_distance_mean=float(np.mean(actual)),
        actual_trace_distance_max=float(np.max(actual)),
        physical_threshold_violations=physical_violations,
        path_budget=path_budget,
    )


def q_iPrune_true_gate_pruning_by_identity(
    *,
    all_gates: List[GateOp],
    baseline_circuit: List[GateOp],
    n_qubits: int,
    device_name: str,
    Gq_diag: np.ndarray,
    epsilon_q: float,
    prefix_states_list: Optional[List[np.ndarray]] = None,
    task_states: Optional[np.ndarray] = None,
    tol_identity: float = 0.0,
) -> TruePruneResult:
    """Sequentially delete actual Rot gates by replacing them with identity.

    Every decision is evaluated on the states produced by the already-selected
    prefix.  An accepted deletion must satisfy the threshold for every
    calibration state in both q-weighted and physical Fubini-Study geometry.

    ``prefix_states_list`` remains supported for older callers; only its first
    entry is used.  New callers should pass ``task_states`` directly.
    """
    if task_states is None:
        if not prefix_states_list:
            raise ValueError("Provide task_states or a non-empty prefix_states_list.")
        task_states = prefix_states_list[0]
    current_states = np.asarray(task_states, dtype=np.complex128).copy()
    if current_states.ndim != 2:
        raise ValueError(f"task_states must have shape (M, dim), got {current_states.shape}")

    for index, gate in enumerate(baseline_circuit):
        if gate.pos_id != index:
            raise ValueError(
                "baseline_circuit must contain one gate per consecutive position; "
                f"found pos_id={gate.pos_id} at index={index}."
            )

    id_by_pos: Dict[int, GateOp] = {}
    for gate in all_gates:
        if is_identity_rot(gate, tol=tol_identity):
            id_by_pos[gate.pos_id] = gate

    unitary_cache: Dict[int, np.ndarray] = {}

    def unitary_of(gate: GateOp) -> np.ndarray:
        key = id(gate)
        if key not in unitary_cache:
            unitary_cache[key] = unitary_of_single_gate(
                gate.apply, n_qubits=n_qubits, device_name=device_name
            )
        return unitary_cache[key]

    selected: List[GateOp] = []
    replaced_dq_means: List[float] = []
    replaced_dq_maxima: List[float] = []
    replaced_physical_maxima: List[float] = []

    for pos_id, baseline_gate in enumerate(baseline_circuit):
        baseline_unitary = unitary_of(baseline_gate)
        selected_gate = baseline_gate

        if baseline_gate.gate_type == "Rot" and pos_id in id_by_pos:
            identity_gate = id_by_pos[pos_id]
            identity_unitary = unitary_of(identity_gate)
            q_angles, physical_angles = task_conditioned_output_angles(
                baseline_unitary,
                identity_unitary,
                current_states,
                Gq_diag,
            )
            q_mean = float(np.mean(q_angles))
            q_max = float(np.max(q_angles))
            physical_max = float(np.max(physical_angles))

            if q_max <= epsilon_q and physical_max <= epsilon_q:
                selected_gate = identity_gate
                replaced_dq_means.append(q_mean)
                replaced_dq_maxima.append(q_max)
                replaced_physical_maxima.append(physical_max)

        selected.append(selected_gate)
        selected_unitary = unitary_of(selected_gate)
        current_states = (selected_unitary @ current_states.T).T

    compact = [gate for gate in selected if not is_identity_rot(gate, tol=tol_identity)]
    pruned_positions = len(selected) - len(compact)
    dq_mean = float(np.mean(replaced_dq_means)) if replaced_dq_means else 0.0
    dq_max = float(np.max(replaced_dq_maxima)) if replaced_dq_maxima else 0.0
    physical_angle_max = (
        float(np.max(replaced_physical_maxima)) if replaced_physical_maxima else 0.0
    )

    return TruePruneResult(
        selected_circuit=selected,
        compact_circuit=compact,
        pruned_positions=pruned_positions,
        kept_positions=len(selected) - pruned_positions,
        gate_count_before=len(baseline_circuit),
        gate_count_after=len(compact),
        dq_mean_over_replaced=dq_mean,
        dq_max_over_replaced=dq_max,
        dq_violations_over_replaced=int(
            sum(value > epsilon_q + 1e-12 for value in replaced_dq_maxima)
        ),
        physical_angle_max_over_replaced=physical_angle_max,
        physical_trace_dist_max_over_replaced=float(np.sin(physical_angle_max)),
        physical_violations_over_replaced=int(
            sum(value > epsilon_q + 1e-12 for value in replaced_physical_maxima)
        ),
    )
