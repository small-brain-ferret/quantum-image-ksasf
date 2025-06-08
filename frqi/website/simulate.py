from qiskit import transpile
from qiskit_aer import AerSimulator
import numpy as np

simulator = AerSimulator()

def simulate_and_decode(qc, num_shots=1000):
    t_qc = transpile(qc, simulator)
    result = simulator.run(t_qc, shots=num_shots).result()
    counts = result.get_counts()
    simplified_counts = {key.split()[0]: value for key, value in counts.items()}
    retrieve_image = np.array([])
    for i in range(64):
        s = format(i, '06b')
        new_s = '1' + s
        try:
            value = np.sqrt(simplified_counts[new_s] / num_shots)
        except KeyError:
            value = 0.0
        retrieve_image = np.append(retrieve_image, value)
    retrieve_image *= 8.0 * 255.0
    retrieve_image = retrieve_image.astype('int')
    retrieve_image = retrieve_image.reshape((8, 8))
    return retrieve_image, simplified_counts