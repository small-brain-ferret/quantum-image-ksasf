import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, render_template_string
from qiskit import transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from website.preprocess import load_and_process_image
from website.build_circuit import build_circuit
from website.analysis import balanced_weighted_mae, SSIM, mae, quantum_state_fidelity

app = Flask(__name__)
simulator = AerSimulator()

def array_to_base64_img(arr, size=None):
    fig, ax = plt.subplots(figsize=(size, size) if size else None)
    ax.imshow(arr, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def demo():
    global THUMB_CACHE, IMAGES_CACHE
    if 'THUMB_CACHE' not in globals():
        THUMB_CACHE = None
    if 'IMAGES_CACHE' not in globals():
        IMAGES_CACHE = None
    if THUMB_CACHE is None or IMAGES_CACHE is None:
        print("Loading images and generating thumbnails (first time)...")
        images, _ = load_and_process_image(0)
        print(f"Total images loaded: {len(images)}")
        images = images[:100]
        IMAGES_CACHE = images
        image_thumbs = [array_to_base64_img(img) for img in images]
        THUMB_CACHE = image_thumbs
        print("All thumbnails generated and cached.")
    else:
        images = IMAGES_CACHE
        image_thumbs = THUMB_CACHE
    num_images = len(images)
    images_per_page = 20
    num_pages = (num_images + images_per_page - 1) // images_per_page

    print(f"Request method: {request.method}")
    print(f"Form: {request.form}")
    print(f"Args: {request.args}")

    if request.method == 'POST':
        try:
            shots = int(request.form['shots'])
            index = int(request.form['index'])
            page = int(request.form.get('page', 0))
            print(f"POST index: {index}, shots: {shots}, page: {page}")
            original_image = images[index]
            # Convert image to angles (1D array)
            pixel_values = original_image.flatten()
            normalized_pixels = pixel_values / 255.0
            angles = np.arcsin(normalized_pixels)
            qc = build_circuit(angles)
            print(f"qc type: {type(qc)}")
            t_qc = transpile(qc, simulator)
            result = simulator.run(t_qc, shots=shots).result()
            counts = result.get_counts()
            print(f"Counts: {counts}")
            retrieved = np.zeros((64,), dtype=float)
            for idx2 in range(64):
                key = '1' + format(idx2, '06b')
                freq = counts.get(key, 0)
                retrieved[idx2] = np.sqrt(freq / shots) if freq else 0.0
            print(f"Retrieved array: {retrieved}")
            if isinstance(retrieved, np.ndarray) and retrieved.shape == (64,):
                retrieved_img = np.clip(retrieved * 8.0 * 255.0, 0, 255).astype(np.uint8).reshape((8, 8))
            else:
                print(f"Warning: retrieved is not shape (64,): {retrieved}")
                retrieved_img = np.zeros((8,8), dtype=np.uint8)
            original_b64 = array_to_base64_img(original_image)
            retrieved_b64 = array_to_base64_img(retrieved_img)
            # Compute difference image
            try:
                diff_img = np.abs(original_image.astype(float) - retrieved_img.astype(float))
                diff_b64 = array_to_base64_img(diff_img, size=1.5)
            except Exception as e:
                print(f"Diff error: {e}")
                diff_b64 = None
            # Compute fidelities
            try:
                balanced_mae_fid = balanced_weighted_mae(original_image, retrieved_img)
            except Exception as e:
                print(f"Balanced MAE error: {e}")
                balanced_mae_fid = 'N/A'
            try:
                ssim_fid = SSIM(original_image, retrieved_img)
            except Exception as e:
                print(f"SSIM error: {e}")
                ssim_fid = 'N/A'
            try:
                mae_fid = mae(original_image, retrieved_img)
            except Exception as e:
                print(f"MAE error: {e}")
                mae_fid = 'N/A'
            try:
                state_fid = quantum_state_fidelity(original_image, retrieved_img)
            except Exception as e:
                print(f"Quantum State Fidelity error: {e}")
                state_fid = 'N/A'
            start = page * images_per_page
            end = min(start + images_per_page, num_images)
            grid_thumbs = image_thumbs[start:end]
            grid_html = ''.join([
                '<div class="thumb {}" id="thumb-{}" onclick="selectImage({})"><img src="data:image/png;base64,{}"></div>'.format(
                    'selected' if i+start==index else '', i+start, i+start, img
                ) for i, img in enumerate(grid_thumbs)
            ])
            # Page selector: flex, fits below grid
            page_selector = '<div class="page-selector-flex"><div class="page-selector" style="gap: 16px; display: flex; justify-content: center; flex-wrap: wrap;">' + ''.join([
                f'<button type="button" class="page-btn {'active' if p==page else ''}" style="margin: 0 10px 10px 0; padding: 10px 22px; font-size: 1.15em;" onclick="gotoPage({p})">{p+1}</button>'
                for p in range(num_pages)
            ]) + '</div></div>'
            return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>FRQI Image Reconstruction</title>
        <style>
        body { font-family: Arial, sans-serif; background: #f7f7f7; margin: 0; padding: 0; }
        .container { max-width: 1800px; margin: 40px auto; background: #fff; border-radius: 16px; box-shadow: 0 2px 16px #bbb; padding: 48px; display: flex; flex-direction: row; gap: 48px; margin-top: 0; }
        .left-panel { flex: 1 1 0; min-width: 400px; max-width: 600px; }
        .right-panel { flex: 2 1 0; }
        h2 { text-align: center; font-size: 2.2em; margin-bottom: 8px; margin-top: 18px; }
        .right-panel-flex { display: flex; flex-direction: column; height: 100%; min-height: 0; }
        .grid-selector-wrap { display: flex; flex-direction: column; align-items: stretch; justify-content: flex-start; flex: 1 1 0; min-height: 0; height: 100%; }
        .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 22px; margin-bottom: 0; }
        .page-selector-flex { display: flex; justify-content: center; align-items: center; margin-top: 18px; }
        .thumb { border: 3px solid #eee; border-radius: 10px; overflow: hidden; cursor: pointer; transition: border 0.2s; margin-bottom: 8px; aspect-ratio: 1 / 1; display: flex; align-items: center; justify-content: center; background: #f0f0f0; width: 100%; max-width: 140px; height: auto; justify-content: center; align-items: center; }
        .thumb.selected, .thumb:hover { border: 3px solid #007bff; }
        .thumb img { width: 100%; height: auto; object-fit: cover; aspect-ratio: 1 / 1; display: block; margin: auto; }
        .img-block { display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 36px; }
        .img-block-row { flex-direction: row; align-items: flex-start; gap: 32px; }
        .img-col { display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .diff-col { margin-left: 24px; }
        .img-block img { border-radius: 12px; box-shadow: 0 2px 12px #bbb; width: 100%; max-width: 180px; height: auto; object-fit: cover; aspect-ratio: 1 / 1; background: #f0f0f0; display: block; }
        .diff-img { max-width: 100px; width: 100px; height: auto; margin-top: 8px; border-radius: 8px; box-shadow: 0 1px 6px #bbb; }
        .diff-img { margin-top: 32px !important; }
        .form-section { text-align: center; margin-bottom: 40px; }
        label { margin-right: 16px; font-size: 1.3em; }
        .slider-input-wrap { display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 18px; }
        input[type=number] { width: 100px; padding: 8px; border-radius: 6px; border: 2px solid #ccc; font-size: 1.2em; }
        input[type=range] { width: 220px; }
        button { padding: 12px 32px; border-radius: 8px; border: none; background: #007bff; color: #fff; font-size: 1.3em; cursor: pointer; margin-top: 12px; }
        button:hover { background: #0056b3; }
        .page-selector { text-align: center; }
        .page-btn { margin: 0 3px; padding: 6px 16px; border-radius: 5px; border: 1px solid #bbb; background: #f7f7f7; color: #333; font-size: 1.08em; cursor: pointer; }
        .page-btn.active, .page-btn:hover { background: #007bff; color: #fff; border: 1px solid #007bff; }
        /* iPhone/Small mobile layout */
        @media (max-width: 700px) {
          body { background: #f7f7f7; }
          .container {
            box-shadow: none;
            border-radius: 0;
            margin: 0;
            max-width: 100vw;
            padding: 0;
            background: none;
            display: block;
          }
          .left-panel, .right-panel {
            max-width: 100vw;
            min-width: 0;
            background: none;
            box-shadow: none;
            padding: 0;
          }
          .left-panel {
            order: 2;
            width: 100vw;
            margin: 0 auto 0 auto;
          }
          .right-panel {
            order: 1;
            width: 100vw;
            margin: 0 auto 0 auto;
          }
          .img-block { margin-bottom: 18px; }
          .img-block:last-of-type { margin-bottom: 32px; }
          .img-block-row {
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: center;
            gap: 0;
          }
          .img-col {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
          }
          .diff-col {
            margin-left: 0 !important;
            margin-top: 0 !important;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
          }
          .img-block img {
            max-width: 80vw;
            width: 80vw;
            height: auto;
            box-shadow: none;
            border-radius: 8px;
            margin: 0 auto;
          }
          .diff-img {
            max-width: 60vw;
            width: 60vw;
            margin-top: 32px !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            display: block;
          }
          .slider-input-wrap { flex-direction: column; gap: 8px; }
          .form-section { margin-bottom: 18px; }
          input[type=number], input[type=range] { width: 90vw; max-width: 98vw; font-size: 1em; }
          button { width: 100%; font-size: 1.1em; padding: 10px 0; }
          .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 0 auto; justify-items: center; align-items: center; width: fit-content; max-width: 100vw; }
          .thumb { max-width: 44vw; min-width: 0; border-radius: 7px; margin-bottom: 4px; }
          .thumb img { border-radius: 7px; }
          .page-selector-flex {
            margin-top: 8px;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
          }
          .page-selector {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            width: 100%;
          }
          .page-btn {
            font-size: 1em;
            padding: 6px 10px;
            margin: 0 2px 2px 0;
            display: inline-block;
          }
          h2 { font-size: 1.3em; margin-top: 8px; margin-bottom: 4px; }
        }
        @media (max-width: 400px) {
          .img-block img { width: 98vw; max-width: 98vw; }
          .thumb { width: 40px; max-width: 40px; height: 40px; }
          .grid { grid-template-columns: repeat(2, 1fr); gap: 6px; }
        }
        </style>
<script>
function selectImage(idx) {
    document.getElementById('index').value = idx;
    let thumbs = document.getElementsByClassName('thumb');
    for (let i = 0; i < thumbs.length; i++) {
        thumbs[i].classList.remove('selected');
    }
    document.getElementById('thumb-' + idx).classList.add('selected');
}
function syncSlider(val) {
    document.getElementById('shots').value = val;
    document.getElementById('shots-range').value = val;
}
function syncNumber(val) {
    document.getElementById('shots').value = val;
    document.getElementById('shots-range').value = val;
}
function gotoPage(p) {
    document.getElementById('page').value = p;
    document.getElementById('mainform').submit();
}
</script>
</head>
<body>
<h2>FRQI Image Reconstruction Demo</h2>
<div class="container">
  <div class="left-panel">
    <div class="form-section">
      <form method="POST" id="mainform" autocomplete="off" action="" enctype="application/x-www-form-urlencoded">
        <div class="slider-input-wrap">
          <label for="shots">Number of Shots:</label>
          <input type="range" id="shots-range" min="100" max="5000" step="100" value="{{ shots }}" oninput="syncNumber(this.value)">
          <input type="number" name="shots" id="shots" min="100" max="5000" step="100" value="{{ shots }}" oninput="syncSlider(this.value)">
        </div>
        <input type="hidden" name="index" id="index" value="{{ index }}">
        <input type="hidden" name="page" id="page" value="{{ page }}">
        <button type="submit">Reconstruct Image</button>
      </form>
    </div>
    <div class="img-block">
        <h3 style="font-size:1.5em;">Original Image</h3>
        <img src="data:image/png;base64,{{ original_b64 }}" alt="Original Image"/>
    </div>
    <div class="img-block img-block-row">
        <div class="img-col">
            <h3 style="font-size:1.5em;">Reconstructed Image</h3>
            <img src="data:image/png;base64,{{ retrieved_b64 }}" alt="Reconstructed Image"/>
            <div style="margin-top: 10px; text-align: center; font-size: 1.1em;">
                <span>Balanced Weighted MAE Fidelity: {{ '{:.4f}'.format(balanced_mae_fid) if balanced_mae_fid != 'N/A' else 'N/A' }}</span><br>
                <span>SSIM Fidelity: {{ '{:.4f}'.format(ssim_fid) if ssim_fid != 'N/A' else 'N/A' }}</span><br>
                <span>MAE Fidelity: {{ '{:.4f}'.format(mae_fid) if mae_fid != 'N/A' else 'N/A' }}</span><br>
                <span>Quantum State Fidelity: {{ '{:.4f}'.format(state_fid) if state_fid != 'N/A' else 'N/A' }}</span>
            </div>
        </div>
        <div class="img-col diff-col">
            <h3 style="font-size:1.1em;">Difference</h3>
            {% if diff_b64 %}
            <img src="data:image/png;base64,{{ diff_b64 }}" alt="Difference Image" class="diff-img"/>
            {% else %}
            <div style="color: #888; font-size: 1em;">N/A</div>
            {% endif %}
        </div>
    </div>
  </div>
  <div class="right-panel right-panel-flex">
    <div class="grid-selector-wrap">
      <div class="grid">
        {{ grid_html|safe }}
      </div>
      {{ page_selector|safe }}
    </div>
  </div>
</div>
</body>
</html>
''', shots=shots, index=index, page=page, original_b64=original_b64, retrieved_b64=retrieved_b64, grid_html=grid_html, page_selector=page_selector, balanced_mae_fid=balanced_mae_fid, ssim_fid=ssim_fid, mae_fid=mae_fid, state_fid=state_fid, diff_b64=diff_b64)
        except Exception as e:
            print(f"POST error: {e}")
            return f"<p>Error: {e}</p>"

    # GET request: show default or selected image
    page = 0
    index = 0
    shots = 1000
    try:
        page = int(request.args.get('page', 0))
    except Exception:
        page = 0
    try:
        index = int(request.args.get('index', 0))
    except Exception:
        index = 0
    try:
        shots = int(request.args.get('shots', 1000))
    except Exception:
        shots = 1000
    try:
        original_img = images[index]
    except Exception:
        original_img = images[0]
        index = 0
    print(f"GET index: {index}, shots: {shots}, page: {page}")
    print(f"original_img type: {type(original_img)}, shape: {getattr(original_img, 'shape', None)}")
    original_b64 = array_to_base64_img(original_img)
    if isinstance(original_img, np.ndarray) and original_img.shape == (8, 8):
        retrieved_img = original_img
        retrieved_b64 = array_to_base64_img(original_img)
    else:
        print(f"Warning: original_img is not shape (8,8): {original_img}")
        retrieved_img = np.zeros((8,8), dtype=np.uint8)
        retrieved_b64 = array_to_base64_img(retrieved_img)
    # Compute difference image for GET
    try:
        diff_img = np.abs(original_img.astype(float) - retrieved_img.astype(float))
        diff_b64 = array_to_base64_img(diff_img, size=1.5)
    except Exception as e:
        print(f"Diff error: {e}")
        diff_b64 = None
    # Compute fidelities for GET (showing original as both)
    try:
        balanced_mae_fid = balanced_weighted_mae(original_img, retrieved_img)
    except Exception as e:
        print(f"MAE error: {e}")
        balanced_mae_fid = 'N/A'
    try:
        ssim_fid = SSIM(original_img, retrieved_img)
    except Exception as e:
        print(f"SSIM error: {e}")
        ssim_fid = 'N/A'
    try:
        mae_fid = mae(original_image, retrieved_img)
    except Exception as e:
        print(f"MAE error: {e}")
        mae_fid = 'N/A'
    try:
        state_fid = quantum_state_fidelity(original_image, retrieved_img)
    except Exception as e:
        print(f"Quantum State Fidelity error: {e}")
        state_fid = 'N/A'
    start = page * images_per_page
    end = min(start + images_per_page, num_images)
    grid_thumbs = image_thumbs[start:end]
    grid_html = ''.join([
        '<div class="thumb {}" id="thumb-{}" onclick="selectImage({})"><img src="data:image/png;base64,{}"></div>'.format(
            'selected' if i+start==index else '', i+start, i+start, img
        ) for i, img in enumerate(grid_thumbs)
    ])
    # Page selector: flex, fits below grid
    page_selector = '<div class="page-selector-flex"><div class="page-selector" style="gap: 16px; display: flex; justify-content: center; flex-wrap: wrap;">' + ''.join([
        f'<button type="button" class="page-btn {'active' if p==page else ''}" style="margin: 0 10px 10px 0; padding: 10px 22px; font-size: 1.15em;" onclick="gotoPage({p})">{p+1}</button>'
        for p in range(num_pages)
    ]) + '</div></div>'
    # Use the same HTML/CSS for both GET and POST
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>FRQI Image Reconstruction</title>
        <style>
        body { font-family: Arial, sans-serif; background: #f7f7f7; margin: 0; padding: 0; }
        .container { max-width: 1800px; margin: 40px auto; background: #fff; border-radius: 16px; box-shadow: 0 2px 16px #bbb; padding: 48px; display: flex; flex-direction: row; gap: 48px; margin-top: 0; }
        .left-panel { flex: 1 1 0; min-width: 400px; max-width: 600px; }
        .right-panel { flex: 2 1 0; }
        h2 { text-align: center; font-size: 2.2em; margin-bottom: 8px; margin-top: 18px; }
        .right-panel-flex { display: flex; flex-direction: column; height: 100%; min-height: 0; }
        .grid-selector-wrap { display: flex; flex-direction: column; align-items: stretch; justify-content: flex-start; flex: 1 1 0; min-height: 0; height: 100%; }
        .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 22px; margin-bottom: 0; }
        .page-selector-flex { display: flex; justify-content: center; align-items: center; margin-top: 18px; }
        .thumb { border: 3px solid #eee; border-radius: 10px; overflow: hidden; cursor: pointer; transition: border 0.2s; margin-bottom: 8px; aspect-ratio: 1 / 1; display: flex; align-items: center; justify-content: center; background: #f0f0f0; width: 100%; max-width: 140px; height: auto; justify-content: center; align-items: center; }
        .thumb.selected, .thumb:hover { border: 3px solid #007bff; }
        .thumb img { width: 100%; height: auto; object-fit: cover; aspect-ratio: 1 / 1; display: block; margin: auto; }
        .img-block { display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 36px; }
        .img-block-row { flex-direction: row; align-items: flex-start; gap: 32px; }
        .img-col { display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .diff-col { margin-left: 24px; }
        .img-block img { border-radius: 12px; box-shadow: 0 2px 12px #bbb; width: 100%; max-width: 180px; height: auto; object-fit: cover; aspect-ratio: 1 / 1; background: #f0f0f0; display: block; }
        .diff-img { max-width: 100px; width: 100px; height: auto; margin-top: 8px; border-radius: 8px; box-shadow: 0 1px 6px #bbb; }
        .diff-img { margin-top: 32px !important; }
        .form-section { text-align: center; margin-bottom: 40px; }
        label { margin-right: 16px; font-size: 1.3em; }
        .slider-input-wrap { display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 18px; }
        input[type=number] { width: 100px; padding: 8px; border-radius: 6px; border: 2px solid #ccc; font-size: 1.2em; }
        input[type=range] { width: 220px; }
        button { padding: 12px 32px; border-radius: 8px; border: none; background: #007bff; color: #fff; font-size: 1.3em; cursor: pointer; margin-top: 12px; }
        button:hover { background: #0056b3; }
        .page-selector { text-align: center; }
        .page-btn { margin: 0 3px; padding: 6px 16px; border-radius: 5px; border: 1px solid #bbb; background: #f7f7f7; color: #333; font-size: 1.08em; cursor: pointer; }
        .page-btn.active, .page-btn:hover { background: #007bff; color: #fff; border: 1px solid #007bff; }
        /* --- iPhone/mobile layout --- */
        @media (max-width: 700px) {
            .container {
                flex-direction: column;
                padding: 8px 2vw;
                gap: 18px;
                box-shadow: none;
                border-radius: 0;
                margin: 0;
                max-width: 100vw;
            }
            h2 {
                font-size: 1.3em;
                margin-top: 10px;
                margin-bottom: 4px;
            }
            .left-panel, .right-panel {
                min-width: 0;
                max-width: 100vw;
                padding: 0;
            }
            .left-panel {
                order: 2;
                width: 100vw;
                margin: 0 auto 0 auto;
            }
            .right-panel {
                order: 1;
                width: 100vw;
                margin: 0 auto 0 auto;
            }
            .img-block { margin-bottom: 18px; }
            .img-block:last-of-type { margin-bottom: 32px; }
          .img-block-row {
            flex-direction: row;
            gap: 0;
            justify-content: center;
            align-items: flex-start;
          }
          .img-col {
            flex-direction: row;
            gap: 16px;
            align-items: flex-start;
            justify-content: center;
          }
            .img-block img, .diff-img {
                max-width: 90vw;
                width: 90vw;
                height: auto;
                box-shadow: none;
                border-radius: 8px;
            }
            .diff-img { max-width: 60vw; width: 60vw; margin-top: 8px; }
            .slider-input-wrap { flex-direction: column; gap: 8px; }
            .form-section { margin-bottom: 18px; }
            input[type=number], input[type=range] { width: 90vw; max-width: 98vw; font-size: 1em; }
            button { width: 100%; font-size: 1.1em; padding: 10px 0; }
            .grid { grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 0 auto; justify-items: center; align-items: center; width: fit-content; max-width: 100vw; }
            .thumb { max-width: 28vw; min-width: 0; border-radius: 7px; margin-bottom: 4px; }
            .thumb img { border-radius: 7px; }
          .page-selector-flex { margin-top: 8px; width: 100%; }
          .page-selector {
            flex-direction: row !important;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            width: 100%;
            display: flex;
          }
          .page-btn {
            font-size: 1em;
            padding: 6px 10px;
            margin: 0 2px 2px 0;
            display: inline-block;
          }
            h2 { font-size: 1.3em; margin-top: 8px; margin-bottom: 4px; }
        }
        @media (max-width: 400px) {
            .grid { grid-template-columns: repeat(2, 1fr); }
            .thumb { max-width: 44vw; }
        }
        </style>
<script>
function selectImage(idx) {
    document.getElementById('index').value = idx;
    let thumbs = document.getElementsByClassName('thumb');
    for (let i = 0; i < thumbs.length; i++) {
        thumbs[i].classList.remove('selected');
    }
    document.getElementById('thumb-' + idx).classList.add('selected');
}
function syncSlider(val) {
    document.getElementById('shots').value = val;
    document.getElementById('shots-range').value = val;
}
function syncNumber(val) {
    document.getElementById('shots').value = val;
    document.getElementById('shots-range').value = val;
}
function gotoPage(p) {
    document.getElementById('page').value = p;
    document.getElementById('mainform').submit();
}
</script>
</head>
<body>
<h2>FRQI Image Reconstruction Demo</h2>
<div class="container">
  <div class="left-panel">
    <div class="form-section">
      <form method="POST" id="mainform" autocomplete="off" action="" enctype="application/x-www-form-urlencoded">
        <div class="slider-input-wrap">
          <label for="shots">Number of Shots:</label>
          <input type="range" id="shots-range" min="100" max="5000" step="100" value="{{ shots }}" oninput="syncNumber(this.value)">
          <input type="number" name="shots" id="shots" min="100" max="5000" step="100" value="{{ shots }}" oninput="syncSlider(this.value)">
        </div>
        <input type="hidden" name="index" id="index" value="{{ index }}">
        <input type="hidden" name="page" id="page" value="{{ page }}">
        <button type="submit">Reconstruct Image</button>
      </form>
    </div>
    <div class="img-block">
        <h3 style="font-size:1.5em;">Original Image</h3>
        <img src="data:image/png;base64,{{ original_b64 }}" alt="Original Image"/>
    </div>
    <div class="img-block img-block-row">
        <div class="img-col">
            <h3 style="font-size:1.5em;">Reconstructed Image</h3>
            <img src="data:image/png;base64,{{ retrieved_b64 }}" alt="Reconstructed Image"/>
            <div style="margin-top: 10px; text-align: center; font-size: 1.1em;">
                <span>Balanced Weighted MAE Fidelity: {{ '{:.4f}'.format(balanced_mae_fid) if balanced_mae_fid != 'N/A' else 'N/A' }}</span><br>
                <span>SSIM Fidelity: {{ '{:.4f}'.format(ssim_fid) if ssim_fid != 'N/A' else 'N/A' }}</span><br>
                <span>MAE Fidelity: {{ '{:.4f}'.format(mae_fid) if mae_fid != 'N/A' else 'N/A' }}</span><br>
                <span>Quantum State Fidelity: {{ '{:.4f}'.format(state_fid) if state_fid != 'N/A' else 'N/A' }}</span>
            </div>
        </div>
        <div class="img-col diff-col">
            <h3 style="font-size:1.1em;">Difference</h3>
            {% if diff_b64 %}
            <img src="data:image/png;base64,{{ diff_b64 }}" alt="Difference Image" class="diff-img"/>
            {% else %}
            <div style="color: #888; font-size: 1em;">N/A</div>
            {% endif %}
        </div>
    </div>
  </div>
  <div class="right-panel right-panel-flex">
    <div class="grid-selector-wrap">
      <div class="grid">
        {{ grid_html|safe }}
      </div>
      {{ page_selector|safe }}
    </div>
  </div>
</div>
<script>
// Show a warning if not HTTPS (for mobile browsers)
if (window.location.protocol !== 'https:') {
    var warn = document.getElementById('insecure-warning');
    if (warn) warn.style.display = 'block';
}
</script>
</body>
</html>
''', shots=shots, index=index, page=page, original_b64=original_b64, retrieved_b64=retrieved_b64, grid_html=grid_html, page_selector=page_selector, balanced_mae_fid=balanced_mae_fid, ssim_fid=ssim_fid, mae_fid=mae_fid, state_fid=state_fid, diff_b64=diff_b64)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, use_reloader=False)  # Disable debug mode