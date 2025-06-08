import numpy as np
from qiskit.circuit.library import RYGate

def hadamard(circ, n):
    for i in n:
        circ.h(i)

def change(state, new_state):
    n = len(state)
    c = np.array([])
    for i in range(n):
        if state[i] != new_state[i]:
            c = np.append(c, int(i))
    return c.astype(int) if len(c) > 0 else c

def binary(circ, state, new_state):
    c = change(state, new_state)
    if len(c) > 0:
        circ.x(np.abs(c - 5))
    else:
        pass

def cnri(circ, n, t, theta):
    controls = len(n)
    cry = RYGate(2 * theta).control(controls)
    aux = np.append(n, t).tolist()
    circ.append(cry, aux)

def frqi(circ, n, t, angles):
    hadamard(circ, n)
    j = 0
    for i in angles:
        state = '{0:06b}'.format(j - 1)
        new_state = '{0:06b}'.format(j)
        if j == 0:
            cnri(circ, n, t, i)
        else:
            binary(circ, state, new_state)
            cnri(circ, n, t, i)
        j += 1