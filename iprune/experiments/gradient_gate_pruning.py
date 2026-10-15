from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from iprune.experiments.tasks_vqe import tfim_hamiltonian
from iprune.gates import GateOp


@dataclass(frozen=True)
class GradientGatePruneResult:
    """Matched-budget deletion selected by real VQE gate-mask gradients."""

    compact_circuit: List[GateOp]
    deleted_pos_ids: List[int]
    rot_pos_ids: List[int]
    integrated_gradient_scores: np.ndarray
    endpoint_gradient_scores: np.ndarray
    scoring_time_ms: float


def vqe_integrated_gate_gradient_scores(
    *,
    circuit: Sequence[GateOp],
    n_qubits: int,
    J: float,
    h: float,
    quadrature_points: int = 5,
    device: str = "default.qubit",
) -> tuple[List[int], np.ndarray, np.ndarray, float]:
    """Score Rot gates with energy gradients along each gate-to-identity path.

    A scalar mask ``m`` multiplies all three Euler angles of one Rot gate.  The
    score is the trapezoidal integral of ``|dE/dm|`` from identity (m=0) to the
    trained gate (m=1), with every other gate fixed.  This keeps the original
    notebook's gradient-pruning idea but replaces its random mock gradients
    with task-loss gradients.  The endpoint score is also returned for audit.
    """
    if quadrature_points < 2:
        raise ValueError("quadrature_points must be at least 2")

    template = list(circuit)
    rot_pos_ids = [gate.pos_id for gate in template if gate.gate_type == "Rot"]
    if len(rot_pos_ids) != len(set(rot_pos_ids)):
        raise ValueError("Rot position ids must be unique in a realizable circuit")
    if not rot_pos_ids:
        return [], np.zeros(0), np.zeros(0), 0.0

    dev = qml.device(device, wires=n_qubits)
    hamiltonian = tfim_hamiltonian(n_qubits, J=J, h=h)

    @qml.qnode(dev, interface="autograd")
    def masked_energy(mask_vector):
        rot_index = 0
        for gate in template:
            if gate.gate_type == "Rot":
                if gate.params is None:
                    raise ValueError("Rot gate is missing parameters")
                angles = pnp.asarray(gate.params) * mask_vector[rot_index]
                qml.Rot(*angles, wires=int(gate.wires[0]))
                rot_index += 1
            elif gate.gate_type == "CNOT":
                qml.CNOT(wires=list(gate.wires))
            else:
                raise ValueError(f"Unsupported gate type: {gate.gate_type}")
        return qml.expval(hamiltonian)

    grid = np.linspace(0.0, 1.0, quadrature_points, dtype=np.float64)
    n_rot = len(rot_pos_ids)
    integrated = np.zeros(n_rot, dtype=np.float64)
    endpoint = np.zeros(n_rot, dtype=np.float64)
    started = perf_counter()

    # Only one gate moves along the deletion path at a time.  Gradients for all
    # masks are produced by autograd, and the relevant component is retained.
    for rot_index in range(n_rot):
        magnitudes = []
        for mask_value in grid:
            masks = np.ones(n_rot, dtype=np.float64)
            masks[rot_index] = mask_value
            differentiable_masks = pnp.array(masks, requires_grad=True)
            gradient = qml.grad(masked_energy)(differentiable_masks)
            magnitudes.append(abs(float(gradient[rot_index])))
        integrated[rot_index] = float(np.trapz(magnitudes, grid))
        endpoint[rot_index] = float(magnitudes[-1])

    elapsed_ms = (perf_counter() - started) * 1000.0
    return rot_pos_ids, integrated, endpoint, elapsed_ms


def gradient_prune_vqe_matched_count(
    *,
    circuit: Sequence[GateOp],
    delete_count: int,
    n_qubits: int,
    J: float,
    h: float,
    quadrature_points: int = 5,
    device: str = "default.qubit",
) -> GradientGatePruneResult:
    """Delete exactly ``delete_count`` least-salient Rot gates."""
    rot_pos_ids, integrated, endpoint, scoring_time_ms = (
        vqe_integrated_gate_gradient_scores(
            circuit=circuit,
            n_qubits=n_qubits,
            J=J,
            h=h,
            quadrature_points=quadrature_points,
            device=device,
        )
    )
    if delete_count < 0 or delete_count > len(rot_pos_ids):
        raise ValueError(
            f"delete_count must be in [0, {len(rot_pos_ids)}], got {delete_count}"
        )

    # Stable lexicographic tie-breaking makes every result reproducible.
    ranked = sorted(
        range(len(rot_pos_ids)), key=lambda index: (integrated[index], rot_pos_ids[index])
    )
    deleted_pos_ids = sorted(rot_pos_ids[index] for index in ranked[:delete_count])
    deleted = set(deleted_pos_ids)
    compact = [gate for gate in circuit if gate.pos_id not in deleted]
    return GradientGatePruneResult(
        compact_circuit=compact,
        deleted_pos_ids=deleted_pos_ids,
        rot_pos_ids=rot_pos_ids,
        integrated_gradient_scores=integrated,
        endpoint_gradient_scores=endpoint,
        scoring_time_ms=scoring_time_ms,
    )
