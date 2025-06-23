from flask import Blueprint, render_template_string

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return '''
    <head>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <h1>Select Batch</h1>
    <form id="batchForm">
        <label for="batch">Select Batch Number (0–41):</label>
        <input type="range" id="batchSlider" name="batch_number" min="0" max="41" value="0" oninput="batchLabel.innerText='Batch ' + (parseInt(this.value)+1) + ' (' + (this.value * 1000) + '–' + ((this.value * 1000)+999) + ')'">
        <p id="batchLabel">Batch 1 (0–999)</p>
        <input type="hidden" name="start_index" id="start_index">
        <label for="metric">Metric:</label>
        <select id="metric" name="metric">
            <option value="ssim">SSIM</option>
            <option value="mae">MAE</option>
        </select>
        <input type="submit" value="Start Processing">
    </form>
    <button onclick="window.location.href='/debug_run?metric=' + document.getElementById('metric').value">Debug: Run 10 Random Images</button>
    <div id="progress"></div>
    <script>
    const form = document.getElementById('batchForm');
    form.onsubmit = e => {
        e.preventDefault();
        document.getElementById('batchSlider').disabled = true;
        document.querySelector('input[type="submit"]').disabled = true;
        const batch_number = document.getElementById('batchSlider').value;
        const start_index = batch_number * 1000;
        document.getElementById('start_index').value = start_index;
        const metric = document.getElementById('metric').value;
        fetch(`/start_batch?start=${start_index}&size=1000&metric=${metric}`);

        const bar = document.createElement('progress');
        bar.max = 1000;
        bar.value = 0;
        bar.id = 'bar';
        const timer = document.createElement('p');
        timer.id = 'timer';
        timer.innerText = 'Elapsed Time: 0.0s';
        document.getElementById('progress').innerHTML = '<p id="progressText"></p>';
        document.getElementById('progress').appendChild(bar);
        document.getElementById('progress').appendChild(timer);

        let timeTicker = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            timer.innerText = `Elapsed Time: ${elapsed.toFixed(1)}s`;
        }, 500);

        let startTime = Date.now();
        let interval = setInterval(async () => {
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
                clearInterval(timeTicker);
                window.location.href = `/result?start=${start_index}&size=1000&weighted_fidelity=${weighted}`;
            }
        }, 1000);
    };
    </script>
    '''