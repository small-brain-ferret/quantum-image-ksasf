from flask import Blueprint, render_template_string, request, send_file
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
from website.plot import plot_metrics

debug_bp = Blueprint('debug', __name__)

@debug_bp.route('/debug_run')
def debug_run():
    metric = request.args.get('metric', 'ssim').lower()
    metric_name = 'SSIM' if metric == 'ssim' else 'MAE'
    total_images = 42000  # Adjust if your dataset size is different
    random_indices = sorted(random.sample(range(total_images), 10))
    shot_counts = np.arange(100, 5001, 200)  # Every 100 shots, up to 5000
    all_rows = [('ImageIndex', 'Shots', metric_name)]
    avg_metric = np.zeros_like(shot_counts, dtype=float)
    metric_matrix = np.zeros((len(shot_counts), len(random_indices)))
    simulator = AerSimulator()
    images, _ = load_and_process_image(0)

    print(f"DEBUG RUN: Using {metric_name} for 10 images.")

    for img_idx, i in enumerate(random_indices):
        try:
            print(f"Processing image {i} ({img_idx+1}/10)...")
            _, angles = load_and_process_image(i)
            qc = build_circuit(angles)
            t_qc = transpile(qc, simulator, optimization_level=0)
            metric_sums = np.zeros_like(shot_counts, dtype=float)
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
                if metric == 'mae':
                    value = balanced_weighted_mae(images[i], retrieved_img)
                else:
                    value = SSIM(images[i], retrieved_img)
                print(f"    {metric_name} for image {i}, shots {shots}: {value:.4f}")
                all_rows.append((i, shots, value))
                metric_sums[j] = value
                metric_matrix[j, img_idx] = value
            avg_metric += metric_sums
        except Exception as e:
            print(f"Skipping image {i} due to error: {e}")
            continue

    avg_metric = np.mean(metric_matrix, axis=1)
    std_metric = np.std(metric_matrix, axis=1)
    print("Standard deviations for each shot count:", std_metric)

    # Save CSV
    filename = f'debug_results_{metric_name.lower()}.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    plot_metrics(shot_counts, avg_metric, metric_name, std_metric, prefix='debug')

    # Embed plot in HTML
    with open(f'debug_plot_{metric_name.lower()}.png', 'rb') as f:
        plot_data = base64.b64encode(f.read()).decode('utf-8')

    html = f'''
    <h2>Debug Run: 10 Random Images</h2>
    <p><a href="/download_csv?start=debug&metric={metric}">Download Debug Results CSV</a></p>
    <h3>Average {metric_name} vs Shots</h3>
    <img src="data:image/png;base64,{plot_data}" alt="{metric_name} Plot"/>
    <p><a href="/">Back to Home</a></p>
    '''
    return render_template_string(html)

@debug_bp.route('/download_csv')
def download_csv():
    metric = request.args.get('metric', 'ssim').lower()
    filename = f'debug_results_{metric}.csv'
    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        return "CSV not found.", 404
    return send_file(path, as_attachment=True)