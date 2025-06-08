from flask import Flask
from qiskit_aer import AerSimulator

simulator = AerSimulator()
progress = {'done': 0, 'total': 1000, 'status': 'idle'}

def create_app():
    app = Flask(__name__)
    app.simulator = simulator
    app.progress = progress

    from website.routes.main import main_bp
    from website.routes.batch import batch_bp
    from website.routes.inspect import inspect_bp
    from website.routes.debug import debug_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(batch_bp)
    app.register_blueprint(inspect_bp)
    app.register_blueprint(debug_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5050, use_reloader=False)