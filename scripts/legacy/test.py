import numpy as np
from iprune.gates import build_hea_gates

if __name__ == '__main__':
    all_g = build_hea_gates(n_qubits=8, depth=12, seed=42, rot_pool_k=4, rot_pool_sigma=0.001, include_cnot=True)
    pool = [g for g in all_g if g.gate_type == "Rot" and g.pos_id == 0]  # pos0 的 Rot pool
    P = np.array([g.params for g in pool], dtype=float)
    print("sigma=0.001 std:", P.std(axis=0))

    all_g = build_hea_gates(n_qubits=8, depth=12, seed=42, rot_pool_k=4, rot_pool_sigma=0.003, include_cnot=True)
    pool = [g for g in all_g if g.gate_type == "Rot" and g.pos_id == 0]
    P = np.array([g.params for g in pool], dtype=float)
    print("sigma=0.01 std:", P.std(axis=0))
