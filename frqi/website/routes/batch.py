from flask import Blueprint, request, render_template_string, send_file, jsonify, current_app
import threading
import os
import base64
from website.logic.batch_processing import run_batch

batch_bp = Blueprint('batch', __name__)

@batch_bp.route('/start_batch')
def start_batch():
    start = int(request.args.get('start', 0))
    size = int(request.args.get('size', 1000))
    weighted_fidelity = bool(int(request.args.get('weighted_fidelity', 0)))
    threading.Thread(target=run_batch, args=(start, size, current_app.simulator, current_app.progress, weighted_fidelity), daemon=True).start()
    return '', 202

@batch_bp.route('/progress')
def get_progress():
    return jsonify(current_app.progress)

@batch_bp.route('/result')
def result():
    start = int(request.args.get('start', 0))
    size = int(request.args.get('size', 1000))
    weighted_fidelity = bool(int(request.args.get('weighted_fidelity', 0)))
    plot_filename = f'batch_{start}_plot{"_weighted" if weighted_fidelity else ""}.png'
    if not os.path.exists(plot_filename):
        return "Plot not found. Please process the batch first.", 503
    with open(plot_filename, 'rb') as f:
        plot_data = base64.b64encode(f.read()).decode('utf-8')

    metric_name = "Weighted Fidelity" if weighted_fidelity else "Fidelity"
    html = f'''
    <h2>Batch {start}–{start+size-1} Processed</h2>
    <p><a href="/download_csv?start={start}&weighted_fidelity={int(weighted_fidelity)}">Download Results CSV</a></p>
    <h3>Average {metric_name} vs Shots</h3>
    <img src="data:image/png;base64,{plot_data}" alt="{metric_name} Plot"/>
    <h3>Inspect Specific Image</h3>
    <form action="/inspect_image" method="get">
        <input type="hidden" name="start" value="{start}" />
        <input type="hidden" name="weighted_fidelity" value="{int(weighted_fidelity)}" />
        <label for="image">Image Index in Batch:</label>
        <input type="number" name="image" min="{start}" max="{start+size-1}" required />
        <label for="shots">Shot Count:</label>
        <input type="number" name="shots" required />
        <button type="submit">View Details</button>
    </form>
    '''
    return render_template_string(html)

@batch_bp.route('/download_csv')
def download_csv():
    start = request.args.get('start', '0')
    weighted_fidelity = bool(int(request.args.get('weighted_fidelity', 0)))
    if start == 'debug':
        filename = f'debug_results{"_weighted" if weighted_fidelity else ""}.csv'
    else:
        filename = f'batch_{start}_results{"_weighted" if weighted_fidelity else ""}.csv'
    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        return 'File not found. Please process the batch first.', 404
    return send_file(path, as_attachment=True, download_name=filename, mimetype='text/csv')