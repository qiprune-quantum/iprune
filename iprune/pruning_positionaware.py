from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from iprune.gates import GateOp, select_first_candidate_per_position
from iprune.qgeometry import dq_task_conditioned
from iprune.pennylane_unitary import unitary_of_single_gate


@dataclass
class PositionAwarePruneResult:
    replaced_indices: List[int]   # indices in ALL-CANDIDATES list that got replaced (optional bookkeeping)
    replaced_count: int           # L in the paper
    group_stats: Dict[Tuple[int, str, Tuple[int, ...]], Dict[str, float]]
    pruned_candidates: List[GateOp]     # same length as input all_gates, but some entries replaced by refs
    selected_circuit: List[GateOp]      # realizable circuit after pruning (one gate per pos_id)
    selected_replacement_count: int     # executed baseline positions mapped to a different candidate

    # --- NEW: paper-critical stats for actually replaced gates ---
    dq_max_over_replaced: float = 0.0
    dq_mean_over_replaced: float = 0.0
    dq_violations_over_replaced: int = 0

    # --- NEW: optional global stats over all compared candidates (for logging) ---
    dq_max_global: float = 0.0
    dq_mean_global: float = 0.0


def q_iPrune_positionaware_candidate_pools(
    all_gates: List[GateOp],
    *,
    n_qubits: int,
    device_name: str,
    prefix_states_list: List[np.ndarray],   # length P+1, indexed by pos_id
    Gq_diag: np.ndarray,
    epsilon_q: float,
    baseline_circuit: Optional[List[GateOp]] = None,
) -> PositionAwarePruneResult:
    """
    Strict paper-aligned pruning for candidate pools:

    - Candidate pools are defined by (pos_id, gate_type, wires).
      (So only gates at the SAME circuit location compete.)
    - We pick a reference candidate per pool (first candidate).
    - For each other candidate in the pool:
        compute d_q(U_ref, U_cand) using position-specific states D_pos = prefix_states_list[pos_id]
      If d_q <= epsilon_q: replace candidate with reference (same position, replacement semantics).
    - Finally map the actually executed baseline candidate through the pool
      replacement.  Candidates that were never executed may be compressed, but
      they cannot silently change the selected forward circuit.
    """
    # Group candidates by same position
    groups: Dict[Tuple[int, str, Tuple[int, ...]], List[int]] = {}

    # --- NEW: store dq for replaced only and all comparisons ---
    replaced_dq_list: List[float] = []  # dq values for actually replaced candidates (must be <= epsilon_q)
    all_dq_list: List[float] = []       # dq values for all compared candidates (idxs[1:]) across all pools

    for i, g in enumerate(all_gates):
        key = (g.pos_id, g.gate_type, tuple(sorted(g.wires)))
        groups.setdefault(key, []).append(i)

    # Cache unitaries for candidates
    U_cache: Dict[int, np.ndarray] = {}

    def getU(idx: int) -> np.ndarray:
        if idx in U_cache:
            return U_cache[idx]
        U = unitary_of_single_gate(all_gates[idx].apply, n_qubits=n_qubits, device_name=device_name)
        U_cache[idx] = U
        return U

    pruned_candidates = list(all_gates)
    replaced_indices: List[int] = []
    L = 0
    stats: Dict[Tuple[int, str, Tuple[int, ...]], Dict[str, float]] = {}

    for key, idxs in groups.items():
        pos_id, gate_type, wires = key

        if len(idxs) == 1:
            stats[key] = {"group_size": 1.0, "replaced": 0.0, "dq_mean": 0.0, "dq_max": 0.0}
            continue

        ref_idx = idxs[0]
        U_ref = getU(ref_idx)

        dq_vals: List[float] = []
        replaced = 0

        # position-specific task ensemble
        if pos_id < 0 or pos_id >= len(prefix_states_list):
            raise ValueError(f"pos_id={pos_id} out of range for prefix_states_list length {len(prefix_states_list)}")
        D_pos = prefix_states_list[pos_id]

        for j in idxs[1:]:
            U_j = getU(j)
            d = float(dq_task_conditioned(U_ref, U_j, D_pos, Gq_diag))
            dq_vals.append(d)
            all_dq_list.append(d)

            if d <= float(epsilon_q):
                # replace candidate at j by the reference candidate (same position semantics)
                pruned_candidates[j] = all_gates[ref_idx]
                replaced_indices.append(j)
                replaced += 1

                # --- NEW: record dq for replaced gates (paper-critical) ---
                replaced_dq_list.append(d)

                L += 1

        stats[key] = {
            "group_size": float(len(idxs)),
            "replaced": float(replaced),
            "dq_mean": float(np.mean(dq_vals)) if dq_vals else 0.0,
            "dq_max": float(np.max(dq_vals)) if dq_vals else 0.0,
        }

    if baseline_circuit is None:
        # Compatibility fallback for historical callers.  Correct experiments
        # should always supply the circuit that produced prefix_states_list.
        selected_circuit = select_first_candidate_per_position(pruned_candidates)
        selected_replacement_count = 0
    else:
        candidate_index_by_identity = {id(gate): index for index, gate in enumerate(all_gates)}
        selected_circuit = []
        selected_replacement_count = 0
        for baseline_gate in baseline_circuit:
            candidate_index = candidate_index_by_identity.get(id(baseline_gate))
            if candidate_index is None:
                raise ValueError("Every baseline gate must be an object from all_gates.")
            selected_gate = pruned_candidates[candidate_index]
            selected_circuit.append(selected_gate)
            if selected_gate is not baseline_gate:
                selected_replacement_count += 1

    # sanity: selected_circuit pos_id should be consecutive 0..P-1
    for idx, g in enumerate(selected_circuit):
        if g.pos_id != idx:
            raise ValueError(
                f"Selected circuit must be ordered by pos_id consecutive. "
                f"Found gate with pos_id={g.pos_id} at index={idx}."
            )

    # --- NEW: compute paper-critical replaced-dq stats ---
    if len(replaced_dq_list) == 0:
        dq_max_over_replaced = 0.0
        dq_mean_over_replaced = 0.0
        dq_violations = 0
    else:
        dq_max_over_replaced = float(np.max(replaced_dq_list))
        dq_mean_over_replaced = float(np.mean(replaced_dq_list))
        # Should be 0 by construction, but we assert numerically with tiny tolerance:
        dq_violations = int(np.sum(np.array(replaced_dq_list) > float(epsilon_q) + 1e-12))

    # --- NEW: compute global stats across all compared candidates ---
    if len(all_dq_list) == 0:
        dq_max_global = 0.0
        dq_mean_global = 0.0
    else:
        dq_max_global = float(np.max(all_dq_list))
        dq_mean_global = float(np.mean(all_dq_list))

    return PositionAwarePruneResult(
        replaced_indices=replaced_indices,
        replaced_count=L,
        group_stats=stats,
        pruned_candidates=pruned_candidates,
        selected_circuit=selected_circuit,
        selected_replacement_count=selected_replacement_count,

        dq_max_over_replaced=dq_max_over_replaced,
        dq_mean_over_replaced=dq_mean_over_replaced,
        dq_violations_over_replaced=dq_violations,

        dq_max_global=dq_max_global,
        dq_mean_global=dq_mean_global,
    )
