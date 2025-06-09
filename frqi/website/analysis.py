import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, state_fidelity

def balanced_weighted_mae(original, retrieved, epsilon=0.01):
    # Normalize to [0,1] if not already
    orig = np.asarray(original, dtype=np.float32) / 255.0
    retr = np.asarray(retrieved, dtype=np.float32) / 255.0
    weights = epsilon + (1 - np.abs(orig - 0.5))
    numerator = np.sum(weights * np.abs(orig - retr))
    denominator = np.sum(weights)
    return 1.0 - (numerator / denominator)  # Higher is better, like fidelity

def compute_fidelity(original, retrieved):
    original_flat = original.flatten()
    original_norm = original_flat / np.linalg.norm(original_flat)
    retrieved_flat = retrieved.flatten()
    retrieved_norm = retrieved_flat / np.linalg.norm(retrieved_flat)
    return state_fidelity(Statevector(original_norm), Statevector(retrieved_norm))

def fidelity_vs_shots(qc, original_image, simulator):
    shot_counts = np.linspace(100, 5000, 50, dtype=int)
    fidelities = []
    orig_flat = original_image.flatten()
    orig_norm = orig_flat / np.linalg.norm(orig_flat)
    ideal_state = Statevector(orig_norm)
    for shots in shot_counts:
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        retrieved = np.zeros_like(orig_flat, dtype=float)
        for idx in range(64):
            key = '1' + format(idx, '06b')
            freq = counts.get(key, 0)
            retrieved[idx] = np.sqrt(freq / shots) if freq else 0.0
        retrieved = (retrieved * 8.0 * 255.0).astype(int)
        retr_norm = retrieved / np.linalg.norm(retrieved)
        fidelities.append(state_fidelity(ideal_state, Statevector(retr_norm)))
 