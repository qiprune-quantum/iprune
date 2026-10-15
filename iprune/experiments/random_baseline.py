from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np


@dataclass
class RandomBaselineResult:
    selected_circuit: Any
    L: int
    replace_ratio_over_candidates: float


def build_random_pruned_circuit_matched_L(
    *,
    all_gates: List[Any],
    selected_circuit_paper: List[Any],
    L_target: int,
    seed: int,
) -> RandomBaselineResult:
    """
    RANDOM baseline (matched-L), consistent with paper candidate pools:

    - Candidate pools are defined EXACTLY as in pruning: key=(pos_id, gate_type, wires).
    - For each position in the realizable circuit (selected_circuit_paper), obtain its pool.
    - Randomly pick L_target eligible positions (pool size > 1),
      then replace that position's gate by a random *different* candidate from its pool.

    This yields a fair baseline:
      - Same candidate pools
      - Same replacement budget L
      - Only difference: not using dq threshold
    """
    rng = np.random.default_rng(seed)

    # Build pools keyed by (pos_id, gate_type, wires)
    pools_by_key: Dict[Tuple[int, str, Tuple[int, ...]], List[Any]] = _group_candidates_by_key(all_gates)

    # Build per-position pools aligned with the realizable circuit ordering (pos_id consecutive)
    pools: List[Optional[List[Any]]] = _build_pos_pools_from_selected(selected_circuit_paper, pools_by_key)

    # Clone selected circuit (list-like)
    pruned = list(selected_circuit_paper)

    eligible_pos = [i for i, pool in enumerate(pools) if pool is not None and len(pool) > 1]
    if not eligible_pos or L_target <= 0:
        return RandomBaselineResult(selected_circuit=pruned, L=0, replace_ratio_over_candidates=0.0)

    # Limit L to #eligible
    L = int(min(L_target, len(eligible_pos)))
    chosen_pos = rng.choice(eligible_pos, size=L, replace=False).tolist()

    for pos in chosen_pos:
        pool = pools[pos]
        assert pool is not None and len(pool) > 1
        cur_gate = pruned[pos]
        pruned[pos] = _random_alt_gate(rng, pool, cur_gate)

    total_alternatives = sum((len(pool) - 1) for pool in pools if pool is not None and len(pool) > 1)
    replace_ratio = float(L) / float(total_alternatives) if total_alternatives > 0 else 0.0

    return RandomBaselineResult(selected_circuit=pruned, L=L, replace_ratio_over_candidates=replace_ratio)


# -----------------------------
# Helpers
# -----------------------------
def _group_candidates_by_key(all_gates: List[Any]) -> Dict[Tuple[int, str, Tuple[int, ...]], List[Any]]:
    pools_by_key: Dict[Tuple[int, str, Tuple[int, ...]], List[Any]] = {}
    for g in all_gates:
        # GateOp expected fields: pos_id, gate_type, wires
        key = (int(g.pos_id), str(g.gate_type), tuple(sorted(tuple(g.wires))))
        pools_by_key.setdefault(key, []).append(g)
    return pools_by_key


def _build_pos_pools_from_selected(
    selected_circuit: List[Any],
    pools_by_key: Dict[Tuple[int, str, Tuple[int, ...]], List[Any]],
) -> List[Optional[List[Any]]]:
    """
    For each gate in the realizable circuit (ordered by pos_id),
    pick its corresponding candidate pool by key=(pos_id, gate_type, wires).

    If a pool is missing (shouldn't happen), we set None.
    """
    pools: List[Optional[List[Any]]] = [None] * len(selected_circuit)
    for i, g in enumerate(selected_circuit):
        key = (int(g.pos_id), str(g.gate_type), tuple(sorted(tuple(g.wires))))
        pools[i] = pools_by_key.get(key, None)
    return pools


def _random_alt_gate(rng: np.random.Generator, pool: List[Any], cur_gate: Any) -> Any:
    """
    Pick an alternative candidate from pool that differs from cur_gate.
    We try:
      1) object inequality
      2) repr inequality fallback
    If no alternative exists (shouldn't if len(pool)>1), return cur_gate.
    """
    alts = [g for g in pool if g != cur_gate]
    if not alts:
        cur_repr = repr(cur_gate)
        alts = [g for g in pool if repr(g) != cur_repr]
    if not alts:
        return cur_gate
    return alts[int(rng.integers(0, len(alts)))]
