from __future__ import annotations
import numpy as np
import pennylane as qml

from iprune.experiments.readout import precompute_z_table, z_expectations_from_statevectors, train_logistic_readout, accuracy
from iprune.experiments.tasks_vqe import tfim_hamiltonian


def expectation_from_state(state: np.ndarray, A: np.ndarray) -> float:
    """<psi|A|psi> for statevector psi."""
    return float(np.real(np.vdot(state, A @ state)))


def evaluate_classification_accuracy_drop(
    *,
    n_qubits: int,
    train_out_states_base: np.ndarray,
    train_labels: np.ndarray,
    val_out_states_base: np.ndarray,
    val_out_states_prun: np.ndarray,
    val_labels: np.ndarray,
    seed: int,
) -> dict:
    """
    Train a fixed readout on BASELINE circuit outputs (train set),
    then evaluate accuracy on validation for baseline vs pruned (same readout).
    """
    Ztable = precompute_z_table(n_qubits)

    Xtr = z_expectations_from_statevectors(train_out_states_base, n_qubits, Ztable)
    Xva_base = z_expectations_from_statevectors(val_out_states_base, n_qubits, Ztable)
    Xva_prun = z_expectations_from_statevectors(val_out_states_prun, n_qubits, Ztable)

    w, b = train_logistic_readout(Xtr, train_labels, seed=seed)

    acc_base = accuracy(Xva_base, val_labels, w, b)
    acc_prun = accuracy(Xva_prun, val_labels, w, b)
    return {
        "acc_base": acc_base,
        "acc_prun": acc_prun,
        "acc_drop": float(acc_base - acc_prun),
    }


def evaluate_vqe_energy_deviation(
    *,
    n_qubits: int,
    states_base: np.ndarray,
    states_prun: np.ndarray,
    J: float = 1.0,
    h: float = 1.0,
) -> dict:
    """
    Compute energy E = <psi|H|psi> on the FINAL states of baseline/pruned circuits,
    report mean/max |ΔE|.
    """
    H = tfim_hamiltonian(n_qubits, J=J, h=h)
    Hmat = qml.matrix(H, wire_order=list(range(n_qubits))).astype(np.complex128)

    Eb = np.array([expectation_from_state(states_base[i], Hmat) for i in range(states_base.shape[0])], dtype=np.float64)
    Ep = np.array([expectation_from_state(states_prun[i], Hmat) for i in range(states_prun.shape[0])], dtype=np.float64)

    dE = np.abs(Eb - Ep)
    return {
        "energy_base_mean": float(np.mean(Eb)),
        "energy_prun_mean": float(np.mean(Ep)),
        "dE_mean": float(np.mean(dE)),
        "dE_max": float(np.max(dE)),
    }
