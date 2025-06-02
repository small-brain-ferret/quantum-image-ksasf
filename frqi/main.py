
import sys
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Checking module imports...")
try:
    from preprocess import load_and_process_image
    print("✅ preprocess.py loaded")
except ImportError as e:
    print(f"❌ preprocess.py failed: {e}")

try:
    from build_circuit import build_circuit
    print("✅ build_circuit.py loaded")
except ImportError as e:
    print(f"❌ build_circuit.py failed: {e}")

try:
    from simulate import simulate_and_decode
    print("✅ simulate.py loaded")
except ImportError as e:
    print(f"❌ simulate.py failed: {e}")

try:
    from analysis import compute_fidelity
    print("✅ analysis.py loaded")
except ImportError as e:
    print(f"❌ analysis.py failed: {e}")

# Configuration: adjust index and shots here
selected_index = int(input("Enter the index of the image to retrieve (0-41999): "))
num_shots = int(input("Enter the number of shots for simulation (e.g., 1000): "))

images, angles = load_and_process_image(selected_index)

qc = build_circuit(angles)

retrieve_image, simplified_counts = simulate_and_decode(qc, num_shots)

# Prepare a large figure to include all visuals
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: Original Image
axs[0, 0].imshow(images[selected_index, :, :], cmap='gray')
axs[0, 0].set_title(f"Original Image (Index = {selected_index})")
axs[0, 0].axis('off')

# Top-right: Retrieved Image
axs[0, 1].imshow(retrieve_image, cmap='gray', vmin=0, vmax=255)
axs[0, 1].set_title(f"Retrieved Image")
axs[0, 1].axis('off')

# Bottom-left: Measurement Histogram
plot_histogram(simplified_counts, title="Measurement Outcomes", bar_labels=False, ax=axs[1, 0])
axs[1, 0].set_title("Measurement Outcomes")

# Bottom-right: Fidelity
fidelity = compute_fidelity(images[selected_index], retrieve_image)
axs[1, 1].axis('off')
axs[1, 1].text(0.1, 0.5, f"Fidelity: {fidelity:.4f}", fontsize=14)

plt.tight_layout()
plt.show()

simulator = AerSimulator()
t_qc = transpile(qc, simulator)

shot_counts = np.linspace(100, 5000, 50, dtype=int)
fidelities = []

for shots in shot_counts:
    result = simulator.run(t_qc, shots=shots).result()
    counts = result.get_counts()
    retrieved = np.zeros((64,), dtype=float)
    for idx in range(64):
        key = '1' + format(idx, '06b')
        freq = counts.get(key, 0)
        retrieved[idx] = np.sqrt(freq / shots) if freq else 0.0
    retrieved = (retrieved * 8.0 * 255.0).astype(int).reshape((8, 8))
    fidelities.append(compute_fidelity(images[selected_index], retrieved))

csv_path = os.path.join(os.path.dirname(__file__), 'fidelity_vs_shots.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Shots', 'Fidelity'])
    for s, f in zip(shot_counts, fidelities):
        writer.writerow([s, f])
print(f"\nCSV saved to: {csv_path}")

plt.figure(figsize=(8, 5))
plt.plot(shot_counts, fidelities, '-', lw=2)
plt.xscale('log')
plt.xlabel('Number of Shots')
plt.ylabel('Fidelity')
plt.title('FRQI Retrieval Fidelity vs. Shots')
plt.grid(True)
plt.show()
