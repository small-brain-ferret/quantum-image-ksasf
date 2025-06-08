from flask import Flask, request, send_file, jsonify
import os
import csv
import threading
import numpy as np
from io import StringIO, BytesIO
from qiskit import transpile
from qiskit_aer import AerSimulator
from preprocess import load_and_process_image
from build_circuit import build_circuit
from simulate import simulate_and_decode
from analysis import compute_fidelity

app = Flask(__name__)
simulator = AerSimulator()
progress = {'done': 0, 'total': 1000, 'status': 'idle'}

@app.route('/start_batch')
def start_batch():
    start = int(request.args.get('start', 0))
    size = int(request.args.get('size', 1000))
    threading.Thread(target=run_batch, args=(start, size), daemon=True).start()
    return '', 202

@app.route('/progress')
def get_progress():
    return jsonify(progress)

@app.route('/download_csv')
def download_csv():
    start = int(request.args.get('start', 0))
    filename = f'batch_{start}_results.csv'
    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        return 'File not found. Please process the batch first.', 404
    return send_file(path, as_attachment=True, download_name=filename, mimetype='text/csv')

@app.route('/inspect_image')
def inspect_image():
    index = int(request.args.get('image'))
    shots = int(request.args.get('shots'))

    images, angles = load_and_process_image(index)
    qc = build_circuit(angles)
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
    fidelity = compute_fidelity(original, retrieved_img)

    # Return results as JSON
    return jsonify({
        'image_index': index,
        'shots': shots,
        'fidelity': float(fidelity),
        'retrieved_image': retrieved_img.tolist(),
        'original_image': original.tolist(),
        'counts': counts
    })

def run_batch(start, size):
    global progress
    import concurrent.futures
    end = start + size
    progress = {'done': 0, 'total': size, 'status': 'running'}

    images, _ = load_and_process_image(0)
    shot_counts = np.arange(100, 2100, 100)
    all_rows = [('ImageIndex', 'Shots', 'Fidelity')]
    avg_fidelity = np.zeros_like(shot_counts, dtype=float)

    def process_image(i):
        _, angles = load_and_process_image(i)
        qc = build_circuit(angles)
        t_qc = transpile(qc, simulator, optimization_level=0)
        rows = []
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
            rows.append((i, shots, fidelity))
            fidelity_sums[j] = fidelity
        return rows, fidelity_sums

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_image, i): i for i in range(start, end)}
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

    # Write results to CSV
    filename = f'batch_{start}_results.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    progress['status'] = 'done'

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5050, use_reloader=False)