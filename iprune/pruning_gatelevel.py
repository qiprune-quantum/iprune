from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .gates import GateOp
from .qgeometry import dq_task_conditioned
from .pennylane_unitary import unitary_of_single_gate


@dataclass
class GatePruneResult:
    kept: List[GateOp]
    pruned: List[GateOp]
    L: int
    group_stats: Dict[Tuple[int, ...], Dict[str, float]]


def _choose_reference_first(n: int) -> int:
    return 0


def _choose_reference_medoid(U_list, states, Gq_diag) -> int:
    n = len(U_list)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = dq_task_conditioned(U_list[i], U_list[j], states, Gq_diag)
            D[i, j] = D[j, i] = d
    return int(np.argmin(D.sum(axis=1)))


def q_iPrune_gates(
    gates: List[GateOp],
    *,
    n_qubits: int,
    device_name: str,
    states: np.ndarray,
    Gq_diag: np.ndarray,
    epsilon_q: float,
    reference: str = "first",
    max_pairs_per_group: Optional[int] = None,
    cache_unitaries: bool = True,
) -> GatePruneResult:
    """
    同一层内 gate-level structured pruning：
      - subgroup = wires tuple
      - reference gate per subgroup
      - compare all others to reference by d_q
    """
    # group by wires
    groups: Dict[Tuple[int, ...], List[GateOp]] = {}
    for g in gates:
        groups.setdefault(tuple(sorted(g.wires)), []).append(g)

    # cache unitaries (optional)
    U_cache: Dict[int, np.ndarray] = {}

    def getU(g: GateOp) -> np.ndarray:
        key = id(g)
        if cache_unitaries and key in U_cache:
            return U_cache[key]
        U = unitary_of_single_gate(g.apply, n_qubits=n_qubits, device_name=device_name)
        if cache_unitaries:
            U_cache[key] = U
        return U

    kept: List[GateOp] = []
    pruned: List[GateOp] = []
    L = 0
    stats: Dict[Tuple[int, ...], Dict[str, float]] = {}

    for wires, glist in groups.items():
        if len(glist) == 1:
            kept.append(glist[0])
            stats[wires] = {"group_size": 1.0, "kept": 1.0, "pruned": 0.0, "dq_mean": 0.0, "dq_max": 0.0}
            continue

        # build unitaries for this group
        U_list = [getU(g) for g in glist]

        if reference == "medoid" and len(glist) <= 80:
            ref_idx = _choose_reference_medoid(U_list, states, Gq_diag)
        else:
            ref_idx = _choose_reference_first(len(glist))

        g_ref = glist[ref_idx]
        U_ref = U_list[ref_idx]
        kept.append(g_ref)

        # ===== DEBUG: inspect two-qubit CNOT groups =====
        if len(wires) == 2:
            print("\n[DEBUG] Inspecting two-qubit group:", wires)
            print("[DEBUG] Group size:", len(glist))
            print("[DEBUG] Reference gate:", g_ref.name)
            print("[DEBUG] First 3 gates in group:")
            for gg in glist[:3]:
                print("   ", gg.name)


        dq_vals = []
        kept_cnt = 1
        pruned_cnt = 0
        compared = 0

        for i, (g, U) in enumerate(zip(glist, U_list)):
            if i == ref_idx:
                continue

            if max_pairs_per_group is not None and compared >= max_pairs_per_group:
                # conservative keep
                kept.append(g)
                kept_cnt += 1
                continue

            d = dq_task_conditioned(U_ref, U, states, Gq_diag)

            # ===== DEBUG: print d_q for two-qubit groups =====
            if len(wires) == 2 and compared <= 3:
                print(f"[DEBUG] d_q({g_ref.name} vs {g.name}) = {d:.6f}")


            dq_vals.append(d)
            compared += 1

            if d <= epsilon_q:
                pruned.append(g)
                pruned_cnt += 1
                L += 1
            else:
                kept.append(g)
                kept_cnt += 1

        # ===== DEBUG: print dq_vals length =====
        if len(wires) == 2:
            print(f"[DEBUG] dq_vals length for wires={wires}: {len(dq_vals)}")


        stats[wires] = {
            "group_size": float(len(glist)),
            "kept": float(kept_cnt),
            "pruned": float(pruned_cnt),
            "dq_mean": float(np.mean(dq_vals)) if dq_vals else 0.0,
            "dq_max": float(np.max(dq_vals)) if dq_vals else 0.0,
        }

    return GatePruneResult(kept=kept, pruned=pruned, L=L, group_stats=stats)
