from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class PruningConfig:
    delta: float = 0.02
    reference: Literal["first", "medoid"] = "first"
    max_pairs_per_group: Optional[int] = None  # None -> compare all vs reference only


@dataclass(frozen=True)
class ExperimentConfig:
    n_qubits: int = 4
    depth: int = 12
    M: int = 50
    seed: int = 42

    gamma: float = 0.05  # hardware proxy (RB) or simulated
    alpha: float = 0.6   # lambda = 1 - alpha*gamma
    qbeta: float = 0.0   # Physical default: q(lambda)=1; q deformation is opt-in.

    pruning: PruningConfig = PruningConfig()

    # backend options: "statevector" recommended for theory-aligned overlaps
    device: Literal["default.qubit"] = "default.qubit"
