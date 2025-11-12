import concurrent.futures
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from website.preprocess import load_and_process_image
from website.build_circuit import build_circuit
from website.analysis import SSIM, balanced_weighted_mae
from qiskit import transpile
from website.plot import plot_metrics

def process_image(i, images, shot_counts, simulator, metric):
    try:
        rows = []
        metric_sums = np.zeros_like(shot_counts, dtype=float)
        _, angles = load_and_process_image(i)
        qc = build_circuit(angles)
        t_qc = transpile(qc, simulator, optimization_level=0)

        for j, shots in enumerate(shot_counts):
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
            rows.append((i, shots, value))
            metric_sums[j] = value
        return rows, metric_sums
    except Exception as e:
        print(f"Skipping image {i} due to error: {e}")
        return [], np.zeros_like(shot_counts, dtype=float)

def run_batch(start, size, simulator, progress, metric):
    end = start + size
    progress['done'] = 0
    progress['total'] = size
    progress['status'] = 'running'

    images, _ = load_and_process_image(0)
    shot_counts = np.concatenate([
    np.arange(20, 201, 20),
    np.arange(200, 3001, 200)
    ])
    metric_name = 'SSIM' if metric == 'ssim' else 'MAE'
    all_rows = [('ImageIndex', 'Shots', metric_name)]
    avg_metric = np.zeros_like(shot_counts, dtype=float)

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_image, i, images, shot_counts, simulator, metric): i
            for i in range(start, end)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                rows, metric_sums = future.result()
            except Exception as e:
                print(f"Error processing image {futures[future]}: {e}")
                continue
            all_rows.extend(rows)
            avg_metric += metric_sums
            progress['done'] += 1

    avg_metric /= size

    filename = f'batch_{start}_results_{metric_name.lower()}.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    plot_metrics(shot_counts, avg_metric, metric_name, prefix=f'batch_{start}')
    progress['status'] = 'done'