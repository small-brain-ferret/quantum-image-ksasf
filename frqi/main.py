
import matplotlib.pyplot as plt
from preprocess import load_and_process_image
from build_circuit import build_circuit
from simulate import simulate_and_decode
from analysis import compute_fidelity

# PARAMETERS
selected_index = 0   # Change this to load a different image
num_shots = 1000     # Change this to use different number of simulation shots

# STEP 1: LOAD IMAGE AND CONVERT TO ANGLES
from preprocess import load_and_process_image
images, angles = load_and_process_image(selected_index)

# STEP 2: BUILD QUANTUM CIRCUIT
from build_circuit import build_circuit
qc = build_circuit(angles)

# STEP 3: RUN SIMULATION AND DECODE IMAGE
from simulate import simulate_and_decode
retrieve_image, simplified_counts = simulate_and_decode(qc, num_shots)

# STEP 4: DISPLAY RETRIEVED VS ORIGINAL IMAGE
import matplotlib.pyplot as plt
plt.imshow(retrieve_image, cmap='gray', vmin=0, vmax=255)
plt.title(f"Retrieved Image (Index = {selected_index})")
plt.show()

plt.imshow(images[selected_index, :], cmap='gray')
plt.title(f"Original Image (Index = {selected_index})")
plt.show()

# STEP 5: COMPUTE FIDELITY
from analysis import compute_fidelity
fidelity = compute_fidelity(images[selected_index], retrieve_image)
print(f"Fidelity between the original and retrieved images: {fidelity}")