from __future__ import annotations
import numpy as np


def precompute_z_table(n_qubits: int) -> np.ndarray:
    """
    Returns Zvals[dim, n_qubits] where Zvals[b, q] in {+1,-1}
    corresponding to eigenvalue of PauliZ on qubit q for basis |b>.
    Convention: computational basis bit 0 -> +1, bit 1 -> -1.
    """
    dim = 2 ** n_qubits
    Z = np.empty((dim, n_qubits), dtype=np.float64)
    for b in range(dim):
        for q in range(n_qubits):
            bit = (b >> (n_qubits - 1 - q)) & 1
            Z[b, q] = 1.0 if bit == 0 else -1.0
    return Z


def z_expectations_from_statevectors(states: np.ndarray, n_qubits: int, Ztable: np.ndarray | None = None) -> np.ndarray:
    """
    states: (N, dim) complex statevectors
    returns features: (N, n_qubits) where features[i,q] = <Z_q>
    """
    if states.ndim != 2:
        raise ValueError("states must be (N, dim)")
    N, dim = states.shape
    if dim != 2 ** n_qubits:
        raise ValueError(f"dim mismatch: got {dim}, expected {2**n_qubits}")

    if Ztable is None:
        Ztable = precompute_z_table(n_qubits)

    probs = (states.real ** 2 + states.imag ** 2)  # (N, dim)
    feats = probs @ Ztable  # (N, n_qubits)
    return feats


def train_logistic_readout(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.2,
    steps: int = 400,
    l2: float = 1e-3,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Simple binary logistic regression with L2 regularization.
    X: (N, d)
    y: (N,) in {0,1}
    Returns (w, b).
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape
    y = y.astype(np.float64)

    # standardize for stability
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sd

    w = rng.normal(scale=0.01, size=(d,))
    b = 0.0

    for _ in range(steps):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        # gradients
        gw = (Xn.T @ (p - y)) / N + l2 * w
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb

    # pack normalization into w,b for inference on raw X
    # z = ((X-mu)/sd)@w + b = X@(w/sd) + (b - (mu/sd)@w)
    w_eff = w / sd.reshape(-1)
    b_eff = b - float((mu.reshape(-1) / sd.reshape(-1)) @ w)
    return w_eff.astype(np.float64), float(b_eff)


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    p = predict_proba(X, w, b)
    yhat = (p >= 0.5).astype(int)
    return float(np.mean(yhat == y.astype(int)))
