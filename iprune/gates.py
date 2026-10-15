from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List, Dict
import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class GateOp:
    """
    Paper-aligned gate primitive with an explicit position id (pos_id).
    - pos_id identifies a unique "location" in the realizable circuit.
    - Candidate pool gates share the same pos_id (within same gate_type,wires).
    """
    name: str
    gate_type: str          # "Rot" or "CNOT"
    layer: int
    pos_id: int
    wires: Tuple[int, ...]
    params: Optional[Tuple[float, ...]]   # Rot angles (a,b,c); None for CNOT
    apply: Callable[[], None]

def is_identity_rot(g: GateOp, tol: float = 0.0) -> bool:
    if g.gate_type != "Rot" or g.params is None:
        return False
    a, b, c = g.params
    if tol <= 0:
        return (a == 0.0 and b == 0.0 and c == 0.0)
    return (abs(a) <= tol and abs(b) <= tol and abs(c) <= tol)

def build_hea_gates(
    n_qubits: int,
    depth: int,
    seed: int = 42,
    *,
    rot_pool_k: int = 4,
    rot_pool_sigma: float = 0.003,
    include_cnot: bool = True,
    include_identity_rot: bool = True,
) -> List[GateOp]:
    """
    Build a gate list with candidate pools, aligned with the paper protocol.

    Realizable circuit locations:
      - For each (layer d, qubit q): one Rot location -> pos_id increments by 1.
      - For each (layer d, edge q->q+1): one CNOT location -> pos_id increments by 1.

    Candidate pools:
      - For each Rot location, create K candidate Rot gates that share the same pos_id.
      - By default, CNOT locations have K=1 (no pool).

    rot_pool_sigma:
      - Std of Gaussian perturbation applied to the Rot angles (a,b,c) around a per-location center.
      - Larger sigma => more diverse candidates (harder to replace under dq threshold).
    """
    if rot_pool_k < 1:
        raise ValueError("rot_pool_k must be >= 1")
    if rot_pool_sigma < 0:
        raise ValueError("rot_pool_sigma must be >= 0")
    if n_qubits < 1 or depth < 1:
        raise ValueError("n_qubits and depth must be >= 1")

    rng = np.random.default_rng(seed)

    # Centers for each (layer, qubit) Rot location: shape (depth, n_qubits, 3)
    centers = rng.uniform(0.0, 2.0 * np.pi, size=(depth, n_qubits, 3))

    gates: List[GateOp] = []
    pos_counter = 0

    for d in range(depth):
        # ---- Rot positions with candidate pool ----
        for q in range(n_qubits):
            center = centers[d, q].copy()
            pos_id = pos_counter
            pos_counter += 1

            for k in range(rot_pool_k):
                cand = center + rng.normal(scale=rot_pool_sigma, size=(3,))
                # a, b, c = (float(cand[0]), float(cand[1]), float(cand[2]))
                a, b, c = map(float, cand.tolist())

                # bind params into closure defaults (avoid late-binding bug)

                def make_rot_gate(_a=a, _b=b, _c=c, _q=q, _d=d, _k=k, _pos=pos_id):
                    def apply():
                        qml.Rot(_a, _b, _c, wires=_q)

                    return GateOp(
                        name=f"Rot(pos={_pos},d={_d},q={_q},k={_k})",
                        gate_type="Rot",
                        layer=_d,
                        pos_id=_pos,
                        wires=(_q,),
                        params=(_a, _b, _c),
                        apply=apply,
                    )

                gates.append(make_rot_gate())

            # Identity candidate (true pruning target)
            if include_identity_rot:
                def make_id_rot(_q=q, _d=d, _pos=pos_id):
                    def apply():
                        # Rot(0,0,0) is identity on a single qubit
                        qml.Rot(0.0, 0.0, 0.0, wires=_q)

                    return GateOp(
                        name=f"IDRot(pos={_pos},d={_d},q={_q})",
                        gate_type="Rot",
                        layer=_d,
                        pos_id=_pos,
                        wires=(_q,),
                        params=(0.0, 0.0, 0.0),
                        apply=apply,
                    )

                gates.append(make_id_rot())

        # ---- CNOT positions (single candidate by default) ----
        if include_cnot:
            for q in range(n_qubits):
                ctrl, tgt = q, (q + 1) % n_qubits
                pos_id = pos_counter
                pos_counter += 1

                def make_cnot_gate(_c=ctrl, _t=tgt, _d=d, _pos=pos_id):
                    def apply():
                        qml.CNOT(wires=[_c, _t])
                    return GateOp(
                        name=f"CNOT(pos={_pos},d={_d},{_c}->{_t})",
                        gate_type="CNOT",
                        layer=_d,
                        pos_id=_pos,
                        wires=(_c, _t),
                        params=None,
                        apply=apply,
                    )
                gates.append(make_cnot_gate())

    return gates


def select_first_candidate_per_position(all_gates: List[GateOp]) -> List[GateOp]:
    """
    Construct a realizable circuit by picking the first candidate for each pos_id.

    Notes:
      - We sort by pos_id to make the realizable circuit order deterministic.
      - If you rely on original insertion order, remove sorting.
    """
    # stable deterministic order for reproducibility
    all_gates_sorted = sorted(all_gates, key=lambda g: g.pos_id)

    seen = set()
    selected: List[GateOp] = []
    for g in all_gates_sorted:
        if g.pos_id in seen:
            continue
        seen.add(g.pos_id)
        selected.append(g)
    return selected


def build_hea_gates_from_parameters(
    params: np.ndarray,
    *,
    include_cnot: bool = True,
    include_identity_rot: bool = True,
) -> Tuple[List[GateOp], List[GateOp]]:
    """Build candidates and the exact HEA circuit represented by ``params``.

    Unlike ``build_hea_gates``, this function does not sample a candidate pool.
    Each Rot position contains the supplied trained/optimized gate and,
    optionally, one identity deletion candidate.  It returns
    ``(all_gates, baseline_circuit)`` so callers cannot accidentally construct
    a baseline from a different parameter set.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 3 or params.shape[2] != 3:
        raise ValueError("params must have shape (depth, n_qubits, 3).")

    depth, n_qubits, _ = params.shape
    all_gates: List[GateOp] = []
    baseline: List[GateOp] = []
    pos_counter = 0

    for layer in range(depth):
        for wire in range(n_qubits):
            a, b, c = map(float, params[layer, wire])
            pos_id = pos_counter
            pos_counter += 1

            def make_rot(
                _a=a,
                _b=b,
                _c=c,
                _wire=wire,
                _layer=layer,
                _pos=pos_id,
            ):
                def apply():
                    qml.Rot(_a, _b, _c, wires=_wire)

                return GateOp(
                    name=f"OptimizedRot(pos={_pos},d={_layer},q={_wire})",
                    gate_type="Rot",
                    layer=_layer,
                    pos_id=_pos,
                    wires=(_wire,),
                    params=(_a, _b, _c),
                    apply=apply,
                )

            trained_gate = make_rot()
            all_gates.append(trained_gate)
            baseline.append(trained_gate)

            if include_identity_rot:
                def make_identity(_wire=wire, _layer=layer, _pos=pos_id):
                    def apply():
                        qml.Rot(0.0, 0.0, 0.0, wires=_wire)

                    return GateOp(
                        name=f"IDRot(pos={_pos},d={_layer},q={_wire})",
                        gate_type="Rot",
                        layer=_layer,
                        pos_id=_pos,
                        wires=(_wire,),
                        params=(0.0, 0.0, 0.0),
                        apply=apply,
                    )

                all_gates.append(make_identity())

        if include_cnot:
            for control in range(n_qubits):
                target = (control + 1) % n_qubits
                pos_id = pos_counter
                pos_counter += 1

                def make_cnot(
                    _control=control,
                    _target=target,
                    _layer=layer,
                    _pos=pos_id,
                ):
                    def apply():
                        qml.CNOT(wires=[_control, _target])

                    return GateOp(
                        name=f"CNOT(pos={_pos},d={_layer},{_control}->{_target})",
                        gate_type="CNOT",
                        layer=_layer,
                        pos_id=_pos,
                        wires=(_control, _target),
                        params=None,
                        apply=apply,
                    )

                entangler = make_cnot()
                all_gates.append(entangler)
                baseline.append(entangler)

    return all_gates, baseline
