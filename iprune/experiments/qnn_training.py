from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from iprune.experiments.vqe_finetune import rebuild_circuit_with_rot_params
from iprune.gates import GateOp


@dataclass(frozen=True)
class QNNTrainingResult:
    circuit: List[GateOp]
    readout_w: np.ndarray
    readout_b: float
    loss_history: np.ndarray


@dataclass(frozen=True)
class QNNConvergedTrainingResult:
    """QNN training result with a deterministic fixed-monitor convergence audit."""

    circuit: List[GateOp]
    readout_w: np.ndarray
    readout_b: float
    loss_history: np.ndarray
    monitor_loss_history: np.ndarray
    steps_run: int
    converged: bool
    final_monitor_window_range: float


def stratified_subsample(
    states: np.ndarray, labels: np.ndarray, n: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    if n >= len(labels):
        return np.asarray(states), labels
    rng = np.random.default_rng(seed)
    chosen = []
    classes = sorted(np.unique(labels).tolist())
    base = n // len(classes)
    remaining = n - base * len(classes)
    for idx, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        take = min(len(candidates), base + (1 if idx < remaining else 0))
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    chosen = np.asarray(chosen, dtype=int)
    rng.shuffle(chosen)
    return np.asarray(states)[chosen], labels[chosen]


def stratified_calibration_test_split(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    n_calibration: int,
    n_test: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    classes = sorted(np.unique(labels).tolist())
    if n_calibration + n_test > len(labels):
        raise ValueError("Not enough validation samples for disjoint calibration/test split")
    per_class = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        rng.shuffle(candidates)
        per_class.append(candidates.tolist())
    interleaved = []
    while any(per_class):
        for candidates in per_class:
            if candidates:
                interleaved.append(candidates.pop())
    cal_indices = np.asarray(interleaved[:n_calibration], dtype=int)
    test_indices = np.asarray(interleaved[n_calibration:n_calibration + n_test], dtype=int)
    rng.shuffle(cal_indices)
    rng.shuffle(test_indices)
    return (
        np.asarray(states)[cal_indices],
        labels[cal_indices],
        np.asarray(states)[test_indices],
        labels[test_indices],
    )


def _rot_params(circuit: Sequence[GateOp]) -> np.ndarray:
    params = [gate.params for gate in circuit if gate.gate_type == "Rot"]
    return np.asarray(params, dtype=np.float64).reshape(-1, 3)


def _make_feature_qnode(circuit: Sequence[GateOp], n_qubits: int, device: str):
    template = list(circuit)
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def feature_qnode(input_states, current_params):
        qml.StatePrep(input_states, wires=range(n_qubits), normalize=False)
        rot_index = 0
        for gate in template:
            if gate.gate_type == "Rot":
                qml.Rot(*current_params[rot_index], wires=int(gate.wires[0]))
                rot_index += 1
            elif gate.gate_type == "CNOT":
                qml.CNOT(wires=list(gate.wires))
            else:
                raise ValueError(f"Unsupported QNN gate type: {gate.gate_type}")
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    return feature_qnode


def _features(qnode, states: np.ndarray, params) -> pnp.ndarray:
    outputs = qnode(states, params)
    return pnp.stack(outputs, axis=1)


def train_qnn_circuit(
    *,
    circuit: Sequence[GateOp],
    train_states: np.ndarray,
    train_labels: np.ndarray,
    n_qubits: int,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: str = "default.qubit",
    initial_readout: Optional[Tuple[np.ndarray, float]] = None,
) -> QNNTrainingResult:
    if steps < 0 or batch_size < 1:
        raise ValueError("steps must be non-negative and batch_size positive")
    rng = np.random.default_rng(seed)
    qnode = _make_feature_qnode(circuit, n_qubits, device)
    params = pnp.array(_rot_params(circuit), requires_grad=True)
    if initial_readout is None:
        w = pnp.array(rng.normal(scale=0.05, size=n_qubits), requires_grad=True)
        b = pnp.array(0.0, requires_grad=True)
    else:
        w = pnp.array(np.asarray(initial_readout[0], dtype=np.float64), requires_grad=True)
        b = pnp.array(float(initial_readout[1]), requires_grad=True)

    train_states = np.asarray(train_states, dtype=np.complex128)
    train_labels = np.asarray(train_labels, dtype=np.float64)

    def loss_for_batch(current_params, current_w, current_b, batch_states, batch_labels):
        feats = _features(qnode, batch_states, current_params)
        logits = feats @ current_w + current_b
        probs = 1.0 / (1.0 + pnp.exp(-logits))
        eps = 1e-8
        bce = -pnp.mean(
            batch_labels * pnp.log(probs + eps)
            + (1.0 - batch_labels) * pnp.log(1.0 - probs + eps)
        )
        return bce + 1e-4 * pnp.sum(current_w**2)

    optimizer = qml.AdamOptimizer(stepsize=lr)
    history = []
    effective_batch = min(batch_size, len(train_labels))
    for _ in range(steps):
        indices = rng.choice(len(train_labels), size=effective_batch, replace=False)
        batch_states = train_states[indices]
        batch_labels = pnp.array(train_labels[indices], requires_grad=False)

        def objective(current_params, current_w, current_b):
            return loss_for_batch(
                current_params, current_w, current_b, batch_states, batch_labels
            )

        params, w, b = optimizer.step(objective, params, w, b)
        history.append(float(objective(params, w, b)))

    return QNNTrainingResult(
        circuit=rebuild_circuit_with_rot_params(circuit, np.asarray(params, dtype=np.float64)),
        readout_w=np.asarray(w, dtype=np.float64),
        readout_b=float(b),
        loss_history=np.asarray(history, dtype=np.float64),
    )


def train_qnn_circuit_with_diagnostics(
    *,
    circuit: Sequence[GateOp],
    train_states: np.ndarray,
    train_labels: np.ndarray,
    monitor_states: np.ndarray,
    monitor_labels: np.ndarray,
    n_qubits: int,
    max_steps: int,
    min_steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    monitor_interval: int = 10,
    monitor_window: int = 5,
    convergence_tol: float = 1e-3,
    device: str = "default.qubit",
    initial_readout: Optional[Tuple[np.ndarray, float]] = None,
) -> QNNConvergedTrainingResult:
    """Train until a fixed monitor-set loss has stabilized or the budget ends.

    The monitor subset is fixed before training and must be disjoint from the
    pruning calibration and held-out test ensembles.  Its deterministic loss
    window avoids treating noisy minibatch losses as convergence evidence.
    """
    if max_steps < 1 or min_steps < 0 or min_steps > max_steps:
        raise ValueError("Require 0 <= min_steps <= max_steps with max_steps >= 1.")
    if batch_size < 1 or monitor_interval < 1 or monitor_window < 2:
        raise ValueError("batch_size and monitor_interval must be positive; monitor_window >= 2.")
    if convergence_tol < 0.0:
        raise ValueError("convergence_tol must be non-negative.")

    rng = np.random.default_rng(seed)
    qnode = _make_feature_qnode(circuit, n_qubits, device)
    params = pnp.array(_rot_params(circuit), requires_grad=True)
    if initial_readout is None:
        w = pnp.array(rng.normal(scale=0.05, size=n_qubits), requires_grad=True)
        b = pnp.array(0.0, requires_grad=True)
    else:
        w = pnp.array(np.asarray(initial_readout[0], dtype=np.float64), requires_grad=True)
        b = pnp.array(float(initial_readout[1]), requires_grad=True)

    train_states = np.asarray(train_states, dtype=np.complex128)
    train_labels = np.asarray(train_labels, dtype=np.float64)
    monitor_states = np.asarray(monitor_states, dtype=np.complex128)
    monitor_labels = np.asarray(monitor_labels, dtype=np.float64)
    if len(train_states) != len(train_labels) or len(monitor_states) != len(monitor_labels):
        raise ValueError("States and labels must have matching lengths.")
    if len(monitor_labels) == 0:
        raise ValueError("The convergence monitor must contain at least one state.")

    def loss_for_batch(current_params, current_w, current_b, batch_states, batch_labels):
        feats = _features(qnode, batch_states, current_params)
        logits = feats @ current_w + current_b
        probs = 1.0 / (1.0 + pnp.exp(-logits))
        eps = 1e-8
        bce = -pnp.mean(
            batch_labels * pnp.log(probs + eps)
            + (1.0 - batch_labels) * pnp.log(1.0 - probs + eps)
        )
        return bce + 1e-4 * pnp.sum(current_w**2)

    optimizer = qml.AdamOptimizer(stepsize=lr)
    history: List[float] = []
    monitor_history: List[float] = []
    effective_batch = min(batch_size, len(train_labels))
    converged = False
    final_range = float("inf")
    for step in range(max_steps):
        indices = rng.choice(len(train_labels), size=effective_batch, replace=False)
        batch_states = train_states[indices]
        batch_labels = pnp.array(train_labels[indices], requires_grad=False)

        def objective(current_params, current_w, current_b):
            return loss_for_batch(
                current_params, current_w, current_b, batch_states, batch_labels
            )

        params, w, b = optimizer.step(objective, params, w, b)
        history.append(float(objective(params, w, b)))
        if (step + 1) % monitor_interval != 0:
            continue
        monitor_value = float(
            loss_for_batch(
                params,
                w,
                b,
                monitor_states,
                pnp.array(monitor_labels, requires_grad=False),
            )
        )
        monitor_history.append(monitor_value)
        if step + 1 >= min_steps and len(monitor_history) >= monitor_window:
            final_range = float(np.ptp(monitor_history[-monitor_window:]))
            if final_range <= convergence_tol:
                converged = True
                break

    if len(monitor_history) >= monitor_window:
        final_range = float(np.ptp(monitor_history[-monitor_window:]))
    return QNNConvergedTrainingResult(
        circuit=rebuild_circuit_with_rot_params(circuit, np.asarray(params, dtype=np.float64)),
        readout_w=np.asarray(w, dtype=np.float64),
        readout_b=float(b),
        loss_history=np.asarray(history, dtype=np.float64),
        monitor_loss_history=np.asarray(monitor_history, dtype=np.float64),
        steps_run=len(history),
        converged=converged,
        final_monitor_window_range=final_range,
    )


def evaluate_qnn(
    *,
    circuit: Sequence[GateOp],
    states: np.ndarray,
    labels: np.ndarray,
    readout_w: np.ndarray,
    readout_b: float,
    n_qubits: int,
    device: str = "default.qubit",
    batch_size: int = 128,
) -> float:
    qnode = _make_feature_qnode(circuit, n_qubits, device)
    params = pnp.array(_rot_params(circuit), requires_grad=False)
    predictions = []
    states = np.asarray(states, dtype=np.complex128)
    for start in range(0, len(states), batch_size):
        feats = np.asarray(_features(qnode, states[start:start + batch_size], params), dtype=np.float64)
        logits = feats @ np.asarray(readout_w, dtype=np.float64) + float(readout_b)
        predictions.extend((logits >= 0.0).astype(int).tolist())
    return float(np.mean(np.asarray(predictions, dtype=int) == np.asarray(labels, dtype=int)))
