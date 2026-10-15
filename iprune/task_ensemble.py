from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


def _normalize_state(psi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128)
    nrm = np.linalg.norm(psi)
    if nrm < eps:
        raise ValueError("State norm too small.")
    return psi / nrm


def random_ket(dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return _normalize_state(v)


@dataclass
class TaskEnsemble:
    states: np.ndarray  # (M, dim), complex128

    @property
    def M(self) -> int:
        return self.states.shape[0]

    @property
    def dim(self) -> int:
        return self.states.shape[1]

    @staticmethod
    def from_random(dim: int, M: int, seed: int = 42) -> "TaskEnsemble":
        rng = np.random.default_rng(seed)
        states = np.stack([random_ket(dim, rng) for _ in range(M)], axis=0)
        return TaskEnsemble(states=states)

    @staticmethod
    def from_encoder(
        X_val: np.ndarray,
        encoder: Callable[[np.ndarray], np.ndarray],
        *,
        seed: int = 42,
        M: Optional[int] = None,
    ) -> "TaskEnsemble":
        rng = np.random.default_rng(seed)
        idx = np.arange(len(X_val))
        rng.shuffle(idx)
        if M is None:
            M = len(X_val)
        idx = idx[:M]
        states = []
        for i in idx:
            psi = encoder(X_val[i])
            states.append(_normalize_state(psi))
        return TaskEnsemble(states=np.stack(states, axis=0))
