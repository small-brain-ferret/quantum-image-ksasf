import multiprocessing
multiprocessing.set_start_method('fork', force=True)  # Fix for macOS multiprocessing

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib

from flask import Flask, request, render_template_string
from qiskit import transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from website.preprocess import load_and_process_image
from website.build_circuit import build_circuit

app = Flask(__name__)
simulator = AerSimulator()

def array_to_base64_img(arr):
    """Convert a numpy array to a base64-encoded image."""
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def demo():
    if request.method == 'POST':
        try:
            # Get inputs from the form
            shots = int(request.form['shots'])
            index = int(request.form['index'])

            # Load and process the image
            images, angles = load_and_process_image(index)
            original_image = images[index]

            # Build and simulate the circuit
            qc = build_circuit(angles)
            t_qc = transpile(qc, simulator)
            result = simulator.run(t_qc, shots=shots).result()
            counts = result.get_counts()

            # Decode the retrieved image
            retrieved = np.zeros((64,), dtype=float)
            for idx in range(64):
                key = '1' + format(idx, '06b')
                freq = counts.get(key, 0)
                retrieved[idx] = np.sqrt(freq / shots) if freq else 0.0
            retrieved_img = (retrieved * 8.0 * 255.0).astype(int).reshape((8, 8))

            # Convert images to base64 for rendering
            original_b64 = array_to_base64_img(original_image)
            retrieved_b64 = array_to_base64_img(retrieved_img)

            # Render the result
            return render_template_string(f'''
                <h2>FRQI Image Reconstruction</h2>
                <p>Shots: {shots}</p>
                <p>Image Index: {index}</p>
                <h3>Original Image</h3>
                <img src="data:image/png;base64,{original_b64}" alt="Original Image"/>
                <h3>Reconstructed Image</h3>
                <img src="data:image/png;base64,{retrieved_b64}" alt="Reconstructed Image"/>
                <p><a href="/">Try Again</a></p>
            ''')
        except Exception as e:
            return f"<p>Error: {e}</p><p><a href='/'>Try Again</a></p>"

    # Render the input form
    return '''
        <h2>FRQI Image Reconstruction Demo</h2>
        <form method="POST">
            <label for="index">Image Index:</label>
            <input type="number" name="index" required><br>
            <label for="shots">Number of Shots:</label>
            <input type="number" name="shots" required><br>
            <button type="submit">Reconstruct Image</button>
        </form>
    '''

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, use_reloader=False)  # Disable debug mode