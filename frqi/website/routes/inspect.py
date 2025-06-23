from flask import Blueprint, request, render_template_string
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from website.preprocess import load_and_process_image
from website.build_circuit import build_circuit
from website.analysis import SSIM, balanced_weighted_mae
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

inspect_bp = Blueprint('inspect', __name__)

def array_to_base64_img(arr):
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@inspect_bp.route('/inspect_image')
def inspect_image():
    index = int(request.args.get('image'))
    shots = int(request.args.get('shots'))
    start = int(request.args.get('start', 0))
    metric = request.args.get('metric', 'ssim').lower()

    images, angles = load_and_process_image(index)
    qc = build_circuit(angles)
    simulator = AerSimulator()
    t_qc = transpile(qc, simulator)
    result = simulator.run(t_qc, shots=shots).result()
    counts = result.get_counts()

    retrieved = np.zeros((64,), dtype=float)
    for idx in range(64):
        key = '1' + format(idx, '06b')
        freq = counts.get(key, 0)
        retrieved[idx] = np.sqrt(freq / shots) if freq else 0.0
    retrieved_img = (retrieved * 8.0 * 255.0).astype(int).reshape((8, 8))

    original = images[index]
    if metric == 'mae':
        value = balanced_weighted_mae(original, retrieved_img)
        metric_name = 'MAE'
    else:
        value = SSIM(original, retrieved_img)
        metric_name = 'SSIM'

    orig_img_b64 = array_to_base64_img(original)
    retr_img_b64 = array_to_base64_img(retrieved_img)

    html = f'''
    <link rel="stylesheet" href="/static/style.css">
    <h2>Inspect Image {index}</h2>
    <p>Shots: {shots}</p>
    <p>{metric_name}: {value:.4f}</p>
    <h3>Original Image</h3>
    <img src="data:image/png;base64,{orig_img_b64}" alt="Original Image"/>
    <h3>Retrieved Image</h3>
    <img src="data:image/png;base64,{retr_img_b64}" alt="Retrieved Image"/>
    <p><a href="/result?start={start}&size=1000&metric={metric}">Back to Results</a></p>
    '''
    return render_template_string(html)