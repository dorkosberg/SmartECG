import os
import sys
import traceback

from flask import Flask, jsonify, render_template, request

# Allow imports from src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from inference import get_model, run_inference  # noqa: E402

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB uploads
DEFAULT_MODEL = os.environ.get(
    'MODEL_PATH',
    os.path.join(ROOT, 'output', 'general_model', 'best_model.pth'),
)


@app.after_request
def disable_cache(response):
    if response.content_type and 'text/html' in response.content_type:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'לא הועלה קובץ.'}), 400

    uploaded = request.files['file']
    if not uploaded.filename:
        return jsonify({'error': 'שם הקובץ ריק.'}), 400
    if not uploaded.filename.lower().endswith('.csv'):
        return jsonify({'error': 'יש להעלות קובץ CSV בלבד.'}), 400

    model_path = request.form.get('model_path', DEFAULT_MODEL)
    if not os.path.exists(model_path):
        return jsonify({'error': f'המודל לא נמצא: {model_path}'}), 500

    try:
        uploaded.seek(0)
        result = run_inference(uploaded, model_path)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 400


@app.route('/health')
def health():
    model_ok = os.path.exists(DEFAULT_MODEL)
    return jsonify({
        'status': 'ok' if model_ok else 'model_missing',
        'model_exists': model_ok,
        'model_path': DEFAULT_MODEL,
    })


def _preload_model():
    if os.path.exists(DEFAULT_MODEL):
        print('Loading model...')
        get_model(DEFAULT_MODEL)
        print('Model ready.')
    else:
        print(f'WARNING: model not found at {DEFAULT_MODEL}')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    _preload_model()
    print(f'\n  SmartECG Assist ready at: http://127.0.0.1:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
