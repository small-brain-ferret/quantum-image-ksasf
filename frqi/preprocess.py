
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_image(selected_index):
    dataset = pd.read_csv('mnist_dataset.csv')
    images = dataset.to_numpy()[:, 1:].reshape(42000, 8, 8)
    pixel_values = images.reshape(42000, 64)
    plt.imshow(images[selected_index, :], cmap='gray')
    normalized_pixels = pixel_values / 255.0
    angles = np.arcsin(normalized_pixels[selected_index, :])
    plt.title("Original Image")
    return images, angles

# build_circuit.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from frqi_utils import frqi

def build_circuit(angles):
    qr = QuantumRegister(7, 'q')
    cr = ClassicalRegister(7, 'c')
    qc = QuantumCircuit(qr, cr)
    frqi(qc, [0, 1, 2, 3, 4, 5], 6, angles)
    qc.measure([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6])
    return qc