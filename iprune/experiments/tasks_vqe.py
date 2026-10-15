from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


@dataclass
class VQEConfig:
    n_qubits: int = 4
    depth: int = 6
    steps: int = 40
    lr: float = 0.1
    seed: int = 42
    # TFIM params
    J: float = 1.0
    h: float = 1.0
    min_steps: int = 40
    convergence_window: int = 20
    convergence_tol: float = 1e-5


@dataclass
class VQEOptimizationResult:
    params: np.ndarray
    energy_history: np.ndarray
    converged: bool
    steps_run: int
    final_window_range: float
    stop_reason: str


VQETwitterConfig = VQEConfig  # Backward-compatible alias for older scripts.


def tfim_hamiltonian(n_qubits: int, J: float, h: float):
    """H = -J Σ Z_i Z_{i+1} - h Σ X_i  (ring)"""
    coeffs = []
    ops = []
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


def hea_ansatz(params: np.ndarray, n_qubits: int, depth: int):
    """Simple HEA: Rot layers + ring CNOT"""
    assert params.shape == (depth, n_qubits, 3)
    for d in range(depth):
        for q in range(n_qubits):
            a, b, c = params[d, q]
            qml.Rot(a, b, c, wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])


def _window_converged(
    history: List[float], *, min_steps: int, window: int, tolerance: float
) -> tuple[bool, float]:
    steps_run = len(history) - 1
    if steps_run < min_steps or len(history) < window:
        return False, float("inf")
    final_window_range = float(np.ptp(np.asarray(history[-window:], dtype=np.float64)))
    return final_window_range <= tolerance, final_window_range


def optimize_vqe_parameters_with_diagnostics(
    cfg: VQEConfig,
    device: str = "default.qubit",
) -> VQEOptimizationResult:
    """Optimize the TFIM HEA with an auditable plateau stopping rule."""
    if cfg.steps < 1:
        raise ValueError("steps must be positive.")
    if not 1 <= cfg.min_steps <= cfg.steps:
        raise ValueError("min_steps must lie in [1, steps].")
    if cfg.convergence_window < 2:
        raise ValueError("convergence_window must be at least 2.")
    if cfg.convergence_tol <= 0.0:
        raise ValueError("convergence_tol must be positive.")
    rng = np.random.default_rng(cfg.seed)
    dev = qml.device(device, wires=cfg.n_qubits)
    hamiltonian = tfim_hamiltonian(cfg.n_qubits, cfg.J, cfg.h)
    initial = rng.uniform(0, 2 * np.pi, size=(cfg.depth, cfg.n_qubits, 3))
    params = pnp.array(initial, requires_grad=True)

    @qml.qnode(dev, interface="autograd")
    def energy_qnode(current_params):
        hea_ansatz(current_params, cfg.n_qubits, cfg.depth)
        return qml.expval(hamiltonian)

    optimizer = qml.GradientDescentOptimizer(stepsize=cfg.lr)
    history: List[float] = [float(energy_qnode(params))]
    converged = False
    final_window_range = float("inf")
    for _ in range(cfg.steps):
        params = optimizer.step(energy_qnode, params)
        history.append(float(energy_qnode(params)))
        converged, final_window_range = _window_converged(
            history,
            min_steps=cfg.min_steps,
            window=cfg.convergence_window,
            tolerance=cfg.convergence_tol,
        )
        if converged:
            break

    history_array = np.asarray(history, dtype=np.float64)
    if not np.isfinite(final_window_range):
        final_window_range = float(np.ptp(history_array[-cfg.convergence_window :]))
    return VQEOptimizationResult(
        params=np.asarray(params, dtype=np.float64),
        energy_history=history_array,
        converged=bool(converged),
        steps_run=len(history) - 1,
        final_window_range=final_window_range,
        stop_reason="plateau_tolerance" if converged else "max_steps",
    )


def optimize_vqe_parameters(
    cfg: VQEConfig,
    device: str = "default.qubit",
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible tuple interface for VQE optimization."""
    result = optimize_vqe_parameters_with_diagnostics(cfg, device=device)

    return result.params, result.energy_history


def sample_intermediate_states_from_vqe(cfg: VQEConfig, M: int, device: str = "default.qubit") -> np.ndarray:
    """
    Run a small VQE and sample intermediate states during optimization to form D.
    Returns states (M, 2^n).
    """
    rng = np.random.default_rng(cfg.seed)
    dim = 2 ** cfg.n_qubits
    dev = qml.device(device, wires=cfg.n_qubits)
    H = tfim_hamiltonian(cfg.n_qubits, cfg.J, cfg.h)

    # init params
    params = rng.uniform(0, 2 * np.pi, size=(cfg.depth, cfg.n_qubits, 3)).astype(np.float64)

    @qml.qnode(dev)
    def energy_qnode(p):
        hea_ansatz(p, cfg.n_qubits, cfg.depth)
        return qml.expval(H)

    @qml.qnode(dev)
    def state_qnode(p, upto_layer: int):
        # prepare |0..0>
        # apply ansatz up to certain layer to get intermediate states
        for d in range(upto_layer):
            for q in range(cfg.n_qubits):
                a, b, c = p[d, q]
                qml.Rot(a, b, c, wires=q)
            for q in range(cfg.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % cfg.n_qubits])
        return qml.state()

    opt = qml.GradientDescentOptimizer(stepsize=cfg.lr)

    collected: List[np.ndarray] = []
    # collect states along the trajectory
    for t in range(cfg.steps):
        params = opt.step(energy_qnode, params)

        # sample one intermediate layer uniformly each step
        upto = int(rng.integers(1, cfg.depth + 1))
        s = np.asarray(state_qnode(params, upto), dtype=np.complex128)
        s = s / np.linalg.norm(s)
        collected.append(s)

    # choose M uniformly from collected
    if len(collected) < M:
        # pad by repeating if too few
        reps = (M + len(collected) - 1) // len(collected)
        collected = (collected * reps)[:M]
    else:
        idx = rng.choice(len(collected), size=M, replace=False)
        collected = [collected[i] for i in idx]

    states = np.stack(collected, axis=0).reshape(M, dim)
    return states
