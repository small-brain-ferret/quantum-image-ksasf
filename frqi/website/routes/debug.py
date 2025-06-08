from flask import Blueprint, render_template_string
import random
import base64
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from qiskit import transpile
from qiskit_aer import AerSimulator
from website.preprocess import load_and_process_image
from website.build_circuit import build_circuit
from website.analysis import compute_fidelity

debug_bp = Blueprint('debug', __name__)

@debug_bp.route('/debug_run')
def debug_run():
    total_images = 42000  # Adjust if your dataset size is different
    random_indices = sorted(random.sample(range(total_images), 10))
    shot_counts = np.arange(100, 2100, 100)
    all_rows = [('ImageIndex', 'Shots', 'Fidelity')]
    avg_fidelity = np.zeros_like(shot_counts, dtype=float)
    simulator = AerSimulator()
    images, _ = load_and_process_image(0)

    for i in random_indices:
        try:
            _, angles = load_and_process_image(i)
            qc = build_circuit(angles)
            t_qc = transpile(qc, simulator, optimization_level=0)
            fidelity_sums = np.zeros_like(shot_counts, dtype=float)
            for j, shots in enumerate(shot_counts):
                result = simulator.run(t_qc, shots=shots).result()
                counts = result.get_counts()
                retrieved = np.zeros((64,), dtype=float)
                for idx in range(64):
                    key = '1' + format(idx, '06b')
                    freq = counts.get(key, 0)
                    retrieved[idx] = np.sqrt(freq / shots) if freq else 0.0
                retrieved = (retrieved * 8.0 * 255.0).astype(int).reshape((8, 8))
                fidelity = compute_fidelity(images[i], retrieved)
                all_rows.append((i, shots, fidelity))
                fidelity_sums[j] = fidelity
            avg_fidelity += fidelity_sums
        except Exception as e:
            print(f"Skipping image {i} due to error: {e}")
            continue

    avg_fidelity /= len(random_indices)

    # Save CSV
    filename = 'debug_results.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    # Save plot
    fig, ax = plt.subplots()
    ax.plot(shot_counts, avg_fidelity, marker='o')
    ax.set_title('Average Fidelity for 10 Random Images')
    ax.set_xlabel('Shots')
    ax.set_ylabel('Average Fidelity')
    ax.grid(True)
    plot_filename = 'debug_plot.png'
    plt.savefig(plot_filename, format='png')
    plt.close(fig)

    # Embed plot in HTML
    with open(plot_filename, 'rb') as f:
        plot_data = base64.b64encode(f.read()).decode('utf-8')

    html = f'''
    <h2>Debug Run: 10 Random Images</h2>
    <p><a href="/download_csv?start=debug">Download Debug Results CSV</a></p>
    <h3>Average Fidelity vs Shots</h3>
    <img src="data:image/png;base64,{plot_data}" alt="Fidelity Plot"/>
    <p><a href="/">Back to Home</a></p>
    '''
    return render_template_string(html)