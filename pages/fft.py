from flask import Blueprint, render_template, request
import numpy as np
import json

# Blueprint setup
fft_bp = Blueprint('fft', __name__, template_folder='templates')
MAX_FFT_SAMPLES = 16384

@fft_bp.route('/', methods=['GET', 'POST'])
def fft():
    result = None
    if request.method == 'POST':
        # Parse form inputs
        expr = request.form.get('expression', 'np.sin(2*np.pi*50*t)')
        fs = float(request.form.get('fs', 500.0))
        T = float(request.form.get('duration', 1.0))
        N = int(request.form.get('n_samples', 1024))
        if N > MAX_FFT_SAMPLES:
            return (f"n_samples exceeds limit ({MAX_FFT_SAMPLES}).", 400)

        # Time axis
        t = np.linspace(0, T, N, endpoint=False)
        # Evaluate signal safely
        x = eval(expr, {'np': np, 't': t, 'sqrt': np.sqrt})

        # Compute FFT
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(N, 1/fs)
        # Keep only non-negative frequencies
        mask = freqs >= 0
        freqs = freqs[mask]
        magnitudes = np.abs(X)[mask]
        phases = np.angle(X)[mask]

        # Prepare JSON-serializable result
        result = {
            'freqs': freqs.tolist(),
            'magnitudes': magnitudes.tolist(),
            'phases': phases.tolist(),
            'expr': expr,
            'fs': fs,
            'T': T,
            'N': N
        }
    return render_template('fft.html', result=result)