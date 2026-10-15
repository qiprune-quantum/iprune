from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from iprune.experiments.tasks_vqe import tfim_hamiltonian
from iprune.gates import GateOp


@dataclass(frozen=True)
class VQEFineTuneResult:
    circuit: List[GateOp]
    energy_history: np.ndarray
    final_state: np.ndarray
    trainable_rotations: int


def _rot_params(circuit: Sequence[GateOp]) -> np.ndarray:
    params = [gate.params for gate in circuit if gate.gate_type == "Rot"]
    if not params:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(params, dtype=np.float64).reshape(-1, 3)


def rebuild_circuit_with_rot_params(
    circuit: Sequence[GateOp], params: np.ndarray
) -> List[GateOp]:
    """Return the same circuit structure with updated Rot parameters."""
    params = np.asarray(params, dtype=np.float64).reshape(-1, 3)
    expected = sum(gate.gate_type == "Rot" for gate in circuit)
    if len(params) != expected:
        raise ValueError(f"Expected {expected} Rot parameter triples, got {len(params)}.")

    rebuilt: List[GateOp] = []
    rot_index = 0
    for gate in circuit:
        if gate.gate_type != "Rot":
            rebuilt.append(gate)
            continue
        a, b, c = map(float, params[rot_index])
        rot_index += 1
        wire = int(gate.wires[0])

        def make_rot(
            template: GateOp = gate,
            _a: float = a,
            _b: float = b,
            _c: float = c,
            _wire: int = wire,
        ) -> GateOp:
            def apply() -> None:
                qml.Rot(_a, _b, _c, wires=_wire)

            return GateOp(
                name=f"FineTuned({template.name})",
                gate_type="Rot",
                layer=template.layer,
                pos_id=template.pos_id,
                wires=template.wires,
                params=(_a, _b, _c),
                apply=apply,
            )

        rebuilt.append(make_rot())
    return rebuilt


def fine_tune_vqe_circuit(
    *,
    circuit: Sequence[GateOp],
    n_qubits: int,
    J: float,
    h: float,
    steps: int,
    lr: float,
    device: str = "default.qubit",
) -> VQEFineTuneResult:
    """Optimize only the surviving Rot gates while keeping structure fixed."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if lr <= 0:
        raise ValueError("lr must be positive")

    template = list(circuit)
    initial = _rot_params(template)
    dev = qml.device(device, wires=n_qubits)
    hamiltonian = tfim_hamiltonian(n_qubits, J=J, h=h)

    def apply_template(current_params) -> None:
        rot_index = 0
        for gate in template:
            if gate.gate_type == "Rot":
                qml.Rot(*current_params[rot_index], wires=int(gate.wires[0]))
                rot_index += 1
            elif gate.gate_type == "CNOT":
                qml.CNOT(wires=list(gate.wires))
            else:
                raise ValueError(f"Unsupported gate type for VQE fine-tuning: {gate.gate_type}")

    if len(initial) == 0:
        @qml.qnode(dev)
        def fixed_energy():
            for gate in template:
                if gate.gate_type == "CNOT":
                    qml.CNOT(wires=list(gate.wires))
                else:
                    raise ValueError(f"Unsupported fixed gate type: {gate.gate_type}")
            return qml.expval(hamiltonian)

        @qml.qnode(dev)
        def fixed_state():
            for gate in template:
                if gate.gate_type == "CNOT":
                    qml.CNOT(wires=list(gate.wires))
                else:
                    raise ValueError(f"Unsupported fixed gate type: {gate.gate_type}")
            return qml.state()

        energy = float(fixed_energy())
        return VQEFineTuneResult(
            circuit=template,
            energy_history=np.full(steps + 1, energy, dtype=np.float64),
            final_state=np.asarray(fixed_state(), dtype=np.complex128),
            trainable_rotations=0,
        )

    params = pnp.array(initial, requires_grad=True)

    @qml.qnode(dev, interface="autograd")
    def energy_qnode(current_params):
        apply_template(current_params)
        return qml.expval(hamiltonian)

    @qml.qnode(dev, interface="autograd")
    def state_qnode(current_params):
        apply_template(current_params)
        return qml.state()

    optimizer = qml.GradientDescentOptimizer(stepsize=lr)
    history = [float(energy_qnode(params))]
    for _ in range(steps):
        params = optimizer.step(energy_qnode, params)
        history.append(float(energy_qnode(params)))

    final_params = np.asarray(params, dtype=np.float64)
    return VQEFineTuneResult(
        circuit=rebuild_circuit_with_rot_params(template, final_params),
        energy_history=np.asarray(history, dtype=np.float64),
        final_state=np.asarray(state_qnode(params), dtype=np.complex128),
        trainable_rotations=len(initial),
    )
