from flask import Flask, render_template, jsonify
import compare

app = Flask(__name__)

current_images = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    global current_images
    current_images = compare.generate_grayscale_images()
    return jsonify(current_images)

@app.route('/compare')
def compare_images():
    global current_images
    results = compare.compare_image_pairs(current_images)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True) 