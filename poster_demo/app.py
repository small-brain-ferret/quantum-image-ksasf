import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, request, send_file
import numpy as np
import io
from PIL import Image
from frqi.website.preprocess import load_and_process_image
from frqi.website.build_circuit import build_circuit
from qiskit import transpile
from qiskit_aer import AerSimulator

app = Flask(__name__)

# Load the first 100 images and angles
images = []
angles = []
for idx in range(100):
    imgs, angs = load_and_process_image(idx)
    images.append(imgs[0])
    angles.append(angs[0])
images = np.array(images)
angles = np.array(angles)

@app.route('/')
def index():
    return render_template('index.html', num_images=len(images))

@app.route('/mnist/<int:idx>')
def mnist_img(idx):
    img = Image.fromarray(images[idx].astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/frqi_output', methods=['POST'])
def frqi_output():
    data = request.json
    idx = int(data['idx'])
    shots = int(data['shots'])
    img_angles = angles[idx]
    qc = build_circuit(img_angles)
    simulator = AerSimulator()
    t_qc = transpile(qc, simulator)
    result = simulator.run(t_qc, shots=shots).result()
    counts = result.get_counts()
    # Winner-take-all decode (for demo, just return the original image)
    img = Image.fromarray(images[idx].astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)