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

def process_image(i, images, shot_counts, simulator, weighted_fidelity):
    try:
        rows = []
        fidelity_sums = np.zeros_like(shot_counts, dtype=float)
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
            if weighted_fidelity:
                metric = balanced_weighted_mae(images[i], retrieved_img)
            else:
                metric = SSIM(images[i], retrieved_img)
            rows.append((i, shots, metric))
            fidelity_sums[j] = metric
        return rows, fidelity_sums
    except Exception as e:
        print(f"Skipping image {i} due to error: {e}")
        return [], np.zeros_like(shot_counts, dtype=float)

def run_batch(start, size, simulator, progress, weighted_fidelity):
    end = start + size
    progress['done'] = 0
    progress['total'] = size
    progress['status'] = 'running'

    images, _ = load_and_process_image(0)
    shot_counts = np.arange(100, 2100, 100)
    all_rows = [('ImageIndex', 'Shots', 'WeightedFidelity' if weighted_fidelity else 'Fidelity')]
    avg_fidelity = np.zeros_like(shot_counts, dtype=float)

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_image, i, images, shot_counts, simulator, weighted_fidelity): i
            for i in range(start, end)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                rows, fidelity_sums = future.result()
            except Exception as e:
                print(f"Error processing image {futures[future]}: {e}")
                continue
            all_rows.extend(rows)
            avg_fidelity += fidelity_sums
            progress['done'] += 1

    avg_fidelity /= size

    filename = f'batch_{start}_results{"_weighted" if weighted_fidelity else ""}.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    fig, ax = plt.subplots()
    ax.plot(shot_counts, avg_fidelity, marker='o')
    metric_name = "Weighted Fidelity" if weighted_fidelity else "Fidelity"
    ax.set_title(f'Average {metric_name} for Batch {start}-{end-1}')
    ax.set_xlabel('Shots')
    ax.set_ylabel(f'Average {metric_name}')
    ax.grid(True)
    plot_filename = f'batch_{start}_plot{"_weighted" if weighted_fidelity else ""}.png'
    plt.savefig(plot_filename, format='png')
    plt.close(fig)

    progress['status'] = 'done'