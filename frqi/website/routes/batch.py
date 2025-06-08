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
    threading.Thread(target=run_batch, args=(start, size, current_app.simulator, current_app.progress), daemon=True).start()
    return '', 202

@batch_bp.route('/progress')
def get_progress():
    return jsonify(current_app.progress)

@batch_bp.route('/result')
def result():
    start = int(request.args.get('start', 0))
    size = int(request.args.get('size', 1000))
    plot_filename = f'batch_{start}_plot.png'
    if not os.path.exists(plot_filename):
        return "Plot not found. Please process the batch first.", 503
    with open(plot_filename, 'rb') as f:
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

@batch_bp.route('/download_csv')
def download_csv():
    start = request.args.get('start', '0')
    if start == 'debug':
        filename = 'debug_results.csv'
    else:
        filename = f'batch_{start}_results.csv'
    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        return 'File not found. Please process the batch first.', 404
    return send_file(path, as_attachment=True, download_name=filename, mimetype='text/csv')