from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from .frqi_utils import frqi

def build_circuit(angles):
    qr = QuantumRegister(7, 'q')
    cr = ClassicalRegister(7, 'c')
    qc = QuantumCircuit(qr, cr)
    frqi(qc, [0, 1, 2, 3, 4, 5], 6, angles)
    qc.measure([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6])
    return qc