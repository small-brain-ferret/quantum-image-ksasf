from flask import Blueprint, render_template_string, request
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
from website.analysis import SSIM, balanced_weighted_mae

debug_bp = Blueprint('debug', __name__)

@debug_bp.route('/debug_run')
def debug_run():
    weighted_fidelity = bool(int(request.args.get('weighted_fidelity', 0)))
    total_images = 42000  # Adjust if your dataset size is different
    random_indices = sorted(random.sample(range(total_images), 10))
    shot_counts = np.arange(100, 2100, 100)
    all_rows = [('ImageIndex', 'Shots', 'WeightedFidelity' if weighted_fidelity else 'Fidelity')]
    avg_fidelity = np.zeros_like(shot_counts, dtype=float)
    simulator = AerSimulator()
    images, _ = load_and_process_image(0)

    print(f"DEBUG RUN: Using {'Weighted Fidelity' if weighted_fidelity else 'Fidelity'} for 10 images.")

    for img_idx, i in enumerate(random_indices):
        try:
            print(f"Processing image {i} ({img_idx+1}/10)...")
            _, angles = load_and_process_image(i)
            qc = build_circuit(angles)
            t_qc = transpile(qc, simulator, optimization_level=0)
            fidelity_sums = np.zeros_like(shot_counts, dtype=float)
            for j, shots in enumerate(shot_counts):
                print(f"  Running with {shots} shots...")
                result = simulator.run(t_qc, shots=shots).result()
                counts = result.get_counts()
                retrieved = np.zeros((64,), dtype=float)
                for idx in range(64):
                    key = '1' + format(idx, '06b')
                    freq = counts.get(key, 0)
                    retrieved[idx] = np.sqrt(freq / shots) if freq else 0.0
                retrieved_img = (retrieved * 8.0 * 255.0).astype(int).reshape((8, 8))
                if weighted_fidelity:
                    metric = balanced_weighted_mae(images[i], retrieved_img)
                else:
                    metric = SSIM(images[i], retrieved_img)
                print(f"    Metric for image {i}, shots {shots}: {metric:.4f}")
                all_rows.append((i, shots, metric))
                fidelity_sums[j] = metric
            avg_fidelity += fidelity_sums
        except Exception as e:
            print(f"Skipping image {i} due to error: {e}")
            continue

    avg_fidelity /= len(random_indices)

    # Save CSV
    filename = f'debug_results{"_weighted" if weighted_fidelity else ""}.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    # Save plot
    fig, ax = plt.subplots()
    ax.plot(shot_counts, avg_fidelity, marker='o')
    metric_name = "Weighted Fidelity" if weighted_fidelity else "Fidelity"
    ax.set_title(f'Average {metric_name} for 10 Random Images')
    ax.set_xlabel('Shots')
    ax.set_ylabel(f'Average {metric_name}')
    ax.grid(True)
    plot_filename = f'debug_plot{"_weighted" if weighted_fidelity else ""}.png'
    plt.savefig(plot_filename, format='png')
    plt.close(fig)

    # Embed plot in HTML
    with open(plot_filename, 'rb') as f:
        plot_data = base64.b64encode(f.read()).decode('utf-8')

    html = f'''
    <h2>Debug Run: 10 Random Images</h2>
    <p><a href="/download_csv?start=debug&weighted_fidelity={int(weighted_fidelity)}">Download Debug Results CSV</a></p>
    <h3>Average {metric_name} vs Shots</h3>
    <img src="data:image/png;base64,{plot_data}" alt="{metric_name} Plot"/>
    <p><a href="/">Back to Home</a></p>
    '''
    return render_template_string(html)