import concurrent.futures
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..preprocess import load_and_process_image

def process_image(i, images, shot_counts, simulator):
    try:
        from frqi.website.preprocess import load_and_process_image
        from build_circuit import build_circuit
        from analysis import compute_fidelity
        from qiskit import transpile
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
            retrieved = (retrieved * 8.0 * 255.0).astype(int).reshape((8, 8))
            fidelity = compute_fidelity(images[i], retrieved)
            rows.append((i, shots, fidelity))
            fidelity_sums[j] = fidelity
        return rows, fidelity_sums
    except Exception as e:
        print(f"Skipping image {i} due to error: {e}")
        return [], np.zeros_like(shot_counts, dtype=float)

def run_batch(start, size, simulator, progress):
    end = start + size
    progress['done'] = 0
    progress['total'] = size
    progress['status'] = 'running'

    images, _ = load_and_process_image(0)
    shot_counts = np.arange(100, 2100, 100)
    all_rows = [('ImageIndex', 'Shots', 'Fidelity')]
    avg_fidelity = np.zeros_like(shot_counts, dtype=float)

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_image, i, images, shot_counts, simulator): i
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

    filename = f'batch_{start}_results.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    fig, ax = plt.subplots()
    ax.plot(shot_counts, avg_fidelity, marker='o')
    ax.set_title(f'Average Fidelity for Batch {start}-{end-1}')
    ax.set_xlabel('Shots')
    ax.set_ylabel('Average Fidelity')
    ax.grid(True)
    plot_filename = f'batch_{start}_plot.png'
    plt.savefig(plot_filename, format='png')
    plt.close(fig)

    progress['status'] = 'done'