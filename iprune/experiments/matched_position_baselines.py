from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pennylane as qml

from iprune.gates import GateOp


@dataclass(frozen=True)
class MatchedPositionBaseline:
    method: str
    selected_circuit: List[GateOp]
    replacement_count: int
    eligible_positions: int
    replaced_positions: List[int]
    selected_scores: List[float]


def _pools_by_position(all_gates: Sequence[GateOp]) -> Dict[int, List[GateOp]]:
    pools: Dict[int, List[GateOp]] = {}
    for gate in all_gates:
        pools.setdefault(int(gate.pos_id), []).append(gate)
    return pools


def _eligible_reference_replacements(
    all_gates: Sequence[GateOp], baseline_circuit: Sequence[GateOp]
) -> Tuple[Dict[int, List[GateOp]], List[int]]:
    pools = _pools_by_position(all_gates)
    eligible = []
    for position, baseline_gate in enumerate(baseline_circuit):
        pool = pools[position]
        if len(pool) > 1 and baseline_gate is not pool[0]:
            eligible.append(position)
    return pools, eligible


def _replace_with_references(
    *,
    method: str,
    baseline_circuit: Sequence[GateOp],
    pools: Dict[int, List[GateOp]],
    eligible: Sequence[int],
    chosen: Sequence[int],
    scores: Dict[int, float],
) -> MatchedPositionBaseline:
    selected = list(baseline_circuit)
    for position in chosen:
        selected[position] = pools[position][0]
    return MatchedPositionBaseline(
        method=method,
        selected_circuit=selected,
        replacement_count=len(chosen),
        eligible_positions=len(eligible),
        replaced_positions=[int(position) for position in chosen],
        selected_scores=[float(scores[position]) for position in chosen],
    )


def random_reference_matched_positions(
    *,
    all_gates: Sequence[GateOp],
    baseline_circuit: Sequence[GateOp],
    replacement_count: int,
    seed: int,
) -> MatchedPositionBaseline:
    pools, eligible = _eligible_reference_replacements(all_gates, baseline_circuit)
    if replacement_count > len(eligible):
        raise ValueError("replacement_count exceeds reference-replaceable positions")
    rng = np.random.default_rng(seed)
    chosen = sorted(
        int(position)
        for position in rng.choice(eligible, size=replacement_count, replace=False)
    )
    return _replace_with_references(
        method="random_reference_matched_selected_L",
        baseline_circuit=baseline_circuit,
        pools=pools,
        eligible=eligible,
        chosen=chosen,
        scores={position: 0.0 for position in eligible},
    )


def _rot_matrix(gate: GateOp) -> np.ndarray:
    if gate.gate_type != "Rot" or gate.params is None or len(gate.wires) != 1:
        raise ValueError("Matched score baselines currently require one-qubit Rot pools")
    return np.asarray(qml.Rot.compute_matrix(*gate.params), dtype=np.complex128)


def _apply_one_qubit(
    states: np.ndarray, matrix: np.ndarray, wire: int, n_qubits: int
) -> np.ndarray:
    states = np.asarray(states, dtype=np.complex128)
    shaped = states.reshape((states.shape[0],) + (2,) * n_qubits)
    moved = np.moveaxis(shaped, 1 + wire, -1)
    transformed = np.einsum("ab,...b->...a", matrix, moved)
    restored = np.moveaxis(transformed, -1, 1 + wire)
    return restored.reshape(states.shape)


def _weighted_output_angle(
    reference: GateOp,
    candidate: GateOp,
    states: np.ndarray,
    Gq_diag: np.ndarray,
    n_qubits: int,
) -> float:
    wire = int(reference.wires[0])
    out_reference = _apply_one_qubit(states, _rot_matrix(reference), wire, n_qubits)
    out_candidate = _apply_one_qubit(states, _rot_matrix(candidate), wire, n_qubits)
    weights = np.asarray(Gq_diag, dtype=np.float64).reshape(1, -1)
    numerator = np.abs(
        np.sum(np.conjugate(out_reference) * weights * out_candidate, axis=1)
    )
    norm_reference = np.real(
        np.sum(np.conjugate(out_reference) * weights * out_reference, axis=1)
    )
    norm_candidate = np.real(
        np.sum(np.conjugate(out_candidate) * weights * out_candidate, axis=1)
    )
    denominator = np.sqrt(np.maximum(norm_reference * norm_candidate, 0.0))
    if np.any(denominator <= 1e-12):
        raise ValueError("q-norm too small while scoring a matched baseline")
    angles = np.arccos(np.clip(numerator / denominator, 0.0, 1.0))
    return float(np.mean(angles))


def _unitary_overlap_angle(reference: GateOp, candidate: GateOp) -> float:
    reference_matrix = _rot_matrix(reference)
    candidate_matrix = _rot_matrix(candidate)
    overlap = np.abs(np.trace(reference_matrix.conj().T @ candidate_matrix)) / 2.0
    return float(np.arccos(np.clip(overlap, 0.0, 1.0)))


def _circular_parameter_distance(reference: GateOp, candidate: GateOp) -> float:
    reference_params = np.asarray(reference.params, dtype=np.float64)
    candidate_params = np.asarray(candidate.params, dtype=np.float64)
    difference = np.angle(np.exp(1j * (candidate_params - reference_params)))
    return float(np.linalg.norm(difference))


def _ranked_reference_baseline(
    *,
    method: str,
    all_gates: Sequence[GateOp],
    baseline_circuit: Sequence[GateOp],
    replacement_count: int,
    score: Callable[[int, GateOp, GateOp], float],
) -> MatchedPositionBaseline:
    pools, eligible = _eligible_reference_replacements(all_gates, baseline_circuit)
    if replacement_count > len(eligible):
        raise ValueError("replacement_count exceeds reference-replaceable positions")
    scores = {
        position: float(score(position, pools[position][0], baseline_circuit[position]))
        for position in eligible
    }
    chosen = sorted(eligible, key=lambda position: (scores[position], position))[
        :replacement_count
    ]
    return _replace_with_references(
        method=method,
        baseline_circuit=baseline_circuit,
        pools=pools,
        eligible=eligible,
        chosen=chosen,
        scores=scores,
    )


def build_matched_position_baselines(
    *,
    all_gates: Sequence[GateOp],
    baseline_circuit: Sequence[GateOp],
    prefix_states_list: Sequence[np.ndarray],
    task_states: np.ndarray,
    Gq_diag: np.ndarray,
    replacement_count: int,
    n_qubits: int,
    seed: int,
) -> List[MatchedPositionBaseline]:
    ones = np.ones(2**n_qubits, dtype=np.float64)
    return [
        random_reference_matched_positions(
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            replacement_count=replacement_count,
            seed=seed,
        ),
        _ranked_reference_baseline(
            method="q1_positionaware_matched_selected_L",
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            replacement_count=replacement_count,
            score=lambda position, reference, candidate: _weighted_output_angle(
                reference,
                candidate,
                prefix_states_list[position],
                ones,
                n_qubits,
            ),
        ),
        _ranked_reference_baseline(
            method="qiprune_no_prefix_matched_selected_L",
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            replacement_count=replacement_count,
            score=lambda _position, reference, candidate: _weighted_output_angle(
                reference, candidate, task_states, Gq_diag, n_qubits
            ),
        ),
        _ranked_reference_baseline(
            method="unitary_overlap_matched_selected_L",
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            replacement_count=replacement_count,
            score=lambda _position, reference, candidate: _unitary_overlap_angle(
                reference, candidate
            ),
        ),
        _ranked_reference_baseline(
            method="parameter_euclidean_matched_selected_L",
            all_gates=all_gates,
            baseline_circuit=baseline_circuit,
            replacement_count=replacement_count,
            score=lambda _position, reference, candidate: _circular_parameter_distance(
                reference, candidate
            ),
        ),
    ]
