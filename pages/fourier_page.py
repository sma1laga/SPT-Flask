from flask import Blueprint, render_template, request, jsonify
import numpy as np
from scipy.fftpack import fftfreq, fftshift
from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

fourier_bp = Blueprint("fourier", __name__)

@fourier_bp.route("/", methods=["GET"])
def fourier():
    # Rendert nur die Seite, Plots werden per AJAX/Plotly erzeugt
    return render_template("fourier.html")

@fourier_bp.route("/update", methods=["POST"])
def update_fourier():
    data = request.get_json(force=True)
    func_str = data.get("func", "")
    try:
        phase_deg = float(data.get("phase", 0))
    except:
        phase_deg = 0.0

    result = compute_fourier(func_str, np.deg2rad(phase_deg))
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)

def compute_fourier(func_str, phase_rad):
    # 1. Zeitachse auf [-20,20]
    t = np.linspace(-20, 20, 4000)
    dt = t[1] - t[0]

    # 2. Sandbox für eval
    local_vars = {
        "t": t, "np": np,
        "rect": rect, "tri": tri, "step": step,
        "cos": cos, "sin": sin, "sign": sign,
        "delta": delta, "exp_iwt": exp_iwt,
        "inv_t": inv_t, "si": si, "exp": np.exp
    }

    try:
        y = eval(func_str, local_vars) * np.exp(1j * phase_rad)
    except Exception as e:
        return {"error": f"Error evaluating function: {e}"}

    # 3. FFT und Shift
    Yf = np.fft.fft(y) * dt
    Yf_shifted = fftshift(Yf)
    f = fftfreq(len(t), d=dt)
    f_shifted = fftshift(f)

    # 4. Betrag und Phase
    magnitude = np.abs(Yf_shifted)
    phase_raw = np.angle(Yf_shifted)
    phase_unwrapped = np.unwrap(phase_raw)
    phase_wrapped = (phase_unwrapped + np.pi) % (2 * np.pi) - np.pi

    # 5. Rauschunterdrückung
    threshold = 0.01 * np.max(magnitude) if np.max(magnitude) > 0 else 0
    phase_wrapped[magnitude < threshold] = 0

    # Phase in Einheiten von π darstellen
    phase_norm = phase_wrapped / np.pi

    # 6. Normierung
    if np.max(magnitude) > 0:
        magnitude /= np.max(magnitude)

    return {
        "t": t.tolist(),
        "y_real": y.real.tolist(),
        "y_imag": y.imag.tolist(),
        "f": f_shifted.tolist(),
        "magnitude": magnitude.tolist(),
        "phase": phase_norm.tolist(),
        "transformation_label": f"Phase Shift: {np.rad2deg(phase_rad):.2f}°"
    }
