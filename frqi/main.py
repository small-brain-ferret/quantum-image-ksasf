# main.py
import sys
import os
import matplotlib.pyplot as plt

# Ensure current folder is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Diagnostic test: check file imports
print("Checking module imports...")
try:
    from preprocess import load_and_process_image
    print("preprocess.py loaded")
except ImportError as e:
    print(f"preprocess.py failed: {e}")

try:
    from build_circuit import build_circuit
    print("build_circuit.py loaded")
except ImportError as e:
    print(f"build_circuit.py failed: {e}")

try:
    from simulate import simulate_and_decode
    print("simulate.py loaded")
except ImportError as e:
    print(f"simulate.py failed: {e}")

try:
    from analysis import compute_fidelity
    print("analysis.py loaded")
except ImportError as e:
    print(f"analysis.py failed: {e}")



# PARAMETERS
selected_index = 0   # Change this to load a different image
num_shots = 1000     # Change this to use different number of simulation shots

# STEP 1: LOAD IMAGE AND CONVERT TO ANGLES
images, angles = load_and_process_image(selected_index)

# STEP 2: BUILD QUANTUM CIRCUIT
qc = build_circuit(angles)

# STEP 3: RUN SIMULATION AND DECODE IMAGE
retrieve_image, simplified_counts = simulate_and_decode(qc, num_shots)

# STEP 4: DISPLAY RETRIEVED VS ORIGINAL IMAGE
plt.imshow(retrieve_image, cmap='gray', vmin=0, vmax=255)
plt.title(f"Retrieved Image (Index = {selected_index})")
plt.show()

plt.imshow(images[selected_index, :], cmap='gray')
plt.title(f"Original Image (Index = {selected_index})")
plt.show()

# STEP 5: COMPUTE FIDELITY
fidelity = compute_fidelity(images[selected_index], retrieve_image)
print(f"Fidelity between the original and retrieved images: {fidelity}")