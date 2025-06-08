from flask import Flask, request, render_template_string, send_file, jsonify
import os
import csv
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import StringIO, BytesIO
from qiskit import transpile
from qiskit_aer import AerSimulator
from preprocess import load_and_process_image
from build_circuit import build_circuit
from simulate import simulate_and_decode
from analysis import compute_fidelity

app = Flask(__name__)
simulator = AerSimulator()
progress = {'done': 0, 'total': 10, 'status': 'idle'}

@app.route('/')
def index():
    return '''
    <h1>Select Test Batch</h1>
    <form id="batchForm">
        <label for="batch">Select Batch Number (0–9):</label>
        <input type="range" id="batchSlider" name="batch_number" min="0" max="9" value="0" oninput="batchLabel.innerText='Batch ' + (parseInt(this.value)+1) + ' (' + (this.value * 10) + '–' + ((this.value * 10)+9) + ')'">
        <p id="batchLabel">Batch 1 (0–9)</p>
        <input type="hidden" name="start_index" id="start_index">
        <input type="submit" value="Start Test Batch">
    </form>
    <div id="progress"></div>
    <script>
    const form = document.getElementById('batchForm');
    form.onsubmit = e => {
        e.preventDefault();
        const batch_number = form.batch_number.value;
        const start_index = batch_number * 10;
        document.getElementById('start_index').value = start_index;
        fetch(`/start_batch?start=${start_index}&size=10`);
        const bar = document.createElement('progress');
        bar.max = 10;
        bar.value = 0;
        bar.id = 'bar';
        document.getElementById('progress').innerHTML = '<p id="progressText"></p>';
        document.getElementById('progress').appendChild(bar);

        let startTime = Date.now();
        const interval = setInterval(async () => {
            const res = await fetch('/progress');
            const data = await res.json();
            bar.value = data.done;

            const elapsed = (Date.now() - startTime) / 1000;
            const rate = data.done / elapsed;
            const remaining = rate > 0 ? (data.total - data.done) / rate : 0;
            document.getElementById('progressText').innerText =
                `Processed ${data.done} of ${data.total} images...\nEstimated time remaining: ${remaining.toFixed(1)}s`;

            if (data.done >= data.total) {
                clearInterval(interval);
                window.location.href = `/result?start=${start_index}&size=10`;
            }
        }, 1000);
    }
    </script>
    '''

@app.route('/start_batch')
def start_batch():
    start = int(request.args.get('start', 0))
    size = int(request.args.get('size', 10))
    threading.Thread(target=run_batch, args=(start, size), daemon=True).start()
    return '', 202

@app.route('/progress')
def get_progress():
    return jsonify(progress)

@app.route('/result')
def result():
    import time
    start = int(request.args.get('start', 0))
    size = int(request.args.get('size', 10))
    plot_path = f'batch_{start}_plot.png'
    timeout = 10
    while not os.path.exists(plot_path) and timeout > 0:
        time.sleep(1)
        timeout -= 1

    if not os.path.exists(plot_path):
        return "Plot not found yet. Please refresh this page in a few seconds.", 503

    with open(plot_path, 'rb') as f:
        plot_data = base64.b64encode(f.read()).decode('utf-8')

    html = f'''
    <h2>Batch {start}–{start+size-1} Processed</h2>
    <p><a href="/download_csv?start={start}">Download Results CSV</a></p>
    <h3>Average Fidelity vs Shots</h3>
    <img src="data:image/png;base64,{plot_data}" alt="Fidelity Plot"/>
    <h3>Inspect Specific Image</h3>
    <form action="/inspect_image" method="get">
        <input type="hidden" name="start" value="{start}" />
        <label for="image">Image Index in Batch:</label>
        <input type="number" name="image" min="{start}" max="{start+size-1}" required />
        <label for="shots">Shot Count:</label>
        <input type="number" name="shots" required />
        <button type="submit">View Details</button>
    </form>
    '''
    return render_template_string(html)

@app.route('/download_csv')
def download_csv():
    start = int(request.args.get('start', 0))
    filename = f'batch_{start}_results.csv'
    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        return 'File not found. Please process the batch first.', 404
    return send_file(path, as_attachment=True, download_name=filename, mimetype='text/csv')

def run_batch(start, size):
    global progress
    end = start + size
    progress = {'done': 0, 'total': size, 'status': 'running'}

    images, _ = load_and_process_image(0)
    shot_counts = np.arange(100, 2100, 100)
    all_rows = [('ImageIndex', 'Shots', 'Fidelity')]
    avg_fidelity = np.zeros_like(shot_counts, dtype=float)

    for i in range(start, end):
        _, angles = load_and_process_image(i)
        qc = build_circuit(angles)
        t_qc = transpile(qc, simulator)

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
            avg_fidelity[j] += fidelity
        progress['done'] += 1

    avg_fidelity /= size

    csv_io = StringIO()
    writer = csv.writer(csv_io)
    writer.writerows(all_rows)
    with open(f'batch_{start}_results.csv', 'w', newline='') as f:
        f.write(csv_io.getvalue())

    fig, ax = plt.subplots()
    ax.plot(shot_counts, avg_fidelity, marker='o')
    ax.set_title(f'Average Fidelity for Batch {start}-{end-1}')
    ax.set_xlabel('Shots')
    ax.set_ylabel('Average Fidelity')
    ax.grid(True)
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    with open(f'batch_{start}_plot.png', 'wb') as f:
        f.write(img_buf.getvalue())

    progress['status'] = 'done'

@app.route('/inspect_image')
def inspect_image():
    start = int(request.args.get('start'))
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

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(retrieved_img, cmap='gray')
    axs[1].set_title(f'Reconstructed (shots={shots})')
    axs[1].axis('off')
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')

    # Plot counts as a bar chart
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    sorted_keys = sorted(counts.keys())
    values = [counts[k] for k in sorted_keys]
    ax2.bar(sorted_keys, values)
    ax2.set_title('Measurement Counts')
    ax2.set_xlabel('Measurement Outcome')
    ax2.set_ylabel('Frequency')
    ax2.tick_params(axis='x', labelrotation=90)
    count_buf = BytesIO()
    plt.tight_layout()
    plt.savefig(count_buf, format='png')
    plt.close(fig2)
    count_buf.seek(0)
    counts_plot = base64.b64encode(count_buf.read()).decode('utf-8')

    html = f'''
    <h2>Image {index} Detail (Shots: {shots})</h2>
    <p>Fidelity: {fidelity:.4f}</p>
    <img src="data:image/png;base64,{img_data}" alt="Image Comparison"/>
    <h3>Measurement Counts</h3>
    <img src="data:image/png;base64,{counts_plot}" alt="Measurement Counts Bar Chart"/>
    <img src="data:image/png;base64,{img_data}" alt="Image Comparison"/>
    <p><a href="/result?start={start}&size=10">Back to Batch Result</a></p>
    '''
    return render_template_string(html)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5051, use_reloader=False)
