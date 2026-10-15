from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


def lambda_from_gamma(gamma: float, alpha: float) -> float:
    lam = 1.0 - alpha * gamma
    return float(np.clip(lam, 0.0, 1.0))


def q_from_lambda(lam: float, beta: float) -> float:
    """q(lambda)=exp(beta*(1-lambda)), so q approaches 1 as lambda approaches 1."""
    return float(np.exp(beta * (1.0 - lam)))


def popcount(x: int) -> int:
    return int(bin(x).count("1"))


def build_Gq_diag_popcount(q: float, n_qubits: int) -> Tuple[np.ndarray, float]:
    """
    Build G_q=diag(q**popcount(b)) and normalize its largest eigenvalue to 1.

    The normalization is numerically convenient.  Consequently Mq is 1 and
    must not be interpreted as an independently noise-varying threshold term.
    """
    if q <= 0.0:
        raise ValueError("q must be positive.")
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")

    dim = 2**n_qubits
    weights = np.empty(dim, dtype=np.float64)
    for basis_index in range(dim):
        weights[basis_index] = q ** popcount(basis_index)
    weights /= np.max(weights)
    return weights, float(np.max(weights))


def q_inner_product_diag(psi: np.ndarray, phi: np.ndarray, Gq_diag: np.ndarray) -> complex:
    return np.vdot(psi, Gq_diag * phi)


def q_norm2_diag(psi: np.ndarray, Gq_diag: np.ndarray) -> float:
    return float(np.real(q_inner_product_diag(psi, psi, Gq_diag)))


def task_conditioned_output_angles(
    U: np.ndarray,
    V: np.ndarray,
    states: np.ndarray,
    Gq_diag: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-state q-weighted and physical Fubini-Study angles.

    The q-weighted overlap compares the output states U|psi> and V|psi> and
    normalizes both output q-norms.  This matters because a physical unitary is
    not generally unitary in the G_q inner product.
    """
    U = np.asarray(U, dtype=np.complex128)
    V = np.asarray(V, dtype=np.complex128)
    states = np.asarray(states, dtype=np.complex128)
    Gq_diag = np.asarray(Gq_diag, dtype=np.float64).reshape(-1)

    if states.ndim != 2:
        raise ValueError(f"states must have shape (M, dim), got {states.shape}")
    dim = states.shape[1]
    if U.shape != (dim, dim) or V.shape != (dim, dim):
        raise ValueError(f"U and V must both have shape {(dim, dim)}")
    if Gq_diag.shape != (dim,) or np.any(Gq_diag <= 0.0):
        raise ValueError("Gq_diag must be a positive vector of length dim.")

    q_angles = np.empty(states.shape[0], dtype=np.float64)
    physical_angles = np.empty(states.shape[0], dtype=np.float64)

    for index, psi in enumerate(states):
        out_u = U @ psi
        out_v = V @ psi

        q_norm_u = q_norm2_diag(out_u, Gq_diag)
        q_norm_v = q_norm2_diag(out_v, Gq_diag)
        q_denom = float(np.sqrt(max(q_norm_u * q_norm_v, 0.0)))
        if q_denom < eps:
            raise ValueError("q-norm too small; check G_q and task states.")
        q_overlap = np.abs(q_inner_product_diag(out_u, out_v, Gq_diag)) / q_denom
        q_angles[index] = np.arccos(float(np.clip(q_overlap, 0.0, 1.0)))

        norm_u = float(np.linalg.norm(out_u))
        norm_v = float(np.linalg.norm(out_v))
        physical_denom = norm_u * norm_v
        if physical_denom < eps:
            raise ValueError("Output-state norm too small; check U, V and task states.")
        physical_overlap = np.abs(np.vdot(out_u, out_v)) / physical_denom
        physical_angles[index] = np.arccos(float(np.clip(physical_overlap, 0.0, 1.0)))

    return q_angles, physical_angles


def dq_task_conditioned(
    U: np.ndarray,
    V: np.ndarray,
    states: np.ndarray,
    Gq_diag: np.ndarray,
    eps: float = 1e-12,
    reduction: Literal["mean", "max"] = "mean",
) -> float:
    """Aggregate the task-conditioned q-weighted output-state angle."""
    q_angles, _ = task_conditioned_output_angles(U, V, states, Gq_diag, eps=eps)
    if reduction == "mean":
        return float(np.mean(q_angles))
    if reduction == "max":
        return float(np.max(q_angles))
    raise ValueError(f"Unsupported reduction: {reduction}")


def physical_fs_task_conditioned(
    U: np.ndarray,
    V: np.ndarray,
    states: np.ndarray,
    reduction: Literal["mean", "max"] = "mean",
) -> float:
    """Aggregate the task-conditioned physical Fubini-Study angle."""
    dim = np.asarray(states).shape[1]
    _, physical_angles = task_conditioned_output_angles(
        U, V, states, np.ones(dim, dtype=np.float64)
    )
    if reduction == "mean":
        return float(np.mean(physical_angles))
    if reduction == "max":
        return float(np.max(physical_angles))
    raise ValueError(f"Unsupported reduction: {reduction}")


def epsilon_from_delta(delta: float, Mq: float) -> float:
    """Paper-compatible per-replacement threshold calibration."""
    t = delta * Mq / 2.0
    return float(np.arcsin(np.clip(t, 0.0, 1.0)))
