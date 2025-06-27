import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import matplotlib.pyplot as plt
from frqi.website.preprocess import load_and_process_image
from frqi.website.build_circuit import build_circuit
from qiskit import transpile
from qiskit_aer import AerSimulator

# Usage: python frqi_compare_shots.py <image_index> <shots>
if len(sys.argv) != 3:
    print("Usage: python frqi_compare_shots.py <image_index> <shots>")
    sys.exit(1)

image_index = int(sys.argv[1])
shots1 = int(sys.argv[2])

# Load image and angles using pre-existing function
images, _ = load_and_process_image(0)
image = images[image_index]

# Helper to run FRQI encode/decode for a given number of shots
def frqi_output_img(image_index, shots):
    images, angles = load_and_process_image(image_index)
    qc = build_circuit(angles)
    simulator = AerSimulator()
    t_qc = transpile(qc, simulator, optimization_level=0)
    result = simulator.run(t_qc, shots=shots).result()
    counts = result.get_counts()
    retrieved = np.zeros((64,), dtype=float)
    for idx in range(64):
        key = '1' + format(idx, '06b')
        freq = counts.get(key, 0)
        retrieved[idx] = np.sqrt(freq / shots) if freq else 0.0
    retrieved_img = (retrieved * 8.0 * 255.0).astype(int).reshape((8, 8))
    return retrieved_img

# Generate output image for the given shot count
img1 = frqi_output_img(image_index, shots1)

# Compute difference image (absolute difference, normalized for color display)
diff_img = np.abs(image.astype(float) - img1.astype(float))

# Plot all images
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(img1, cmap='gray', vmin=0, vmax=255)
axs[1].set_title(f'FRQI Output ({shots1} shots)')
axs[1].axis('off')
axs[2].imshow(diff_img, cmap='gray', vmin=0, vmax=255)
axs[2].set_title('Difference Image')
axs[2].axis('off')
plt.tight_layout()
plt.show()
