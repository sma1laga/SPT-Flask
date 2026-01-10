from flask import Blueprint, render_template
import numpy as np
from scipy.fftpack import fftfreq, fftshift
from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

fourier_bp = Blueprint("fourier", __name__)

@fourier_bp.route("/", methods=["GET"])
def fourier():
    # Rendert nur die Seite, Plots werden per AJAX/Plotly erzeugt
    return render_template("fourier.html")

@fourier_bp.route("/update", methods=["GET", "POST"])
def update_fourier():
    # Deprecated: computation now happens client-side
    return render_template("fourier.html")

def compute_fourier(func_str, phase_rad):
    """Compute Fourier transform with dynamic centering."""
    # broad axis for centre estimation
    t_broad = np.linspace(-100, 100, 8000)

    sandbox = {
        "np": np, "rect": rect, "tri": tri, "step": step,
        "cos": cos, "sin": sin, "sign": sign, "delta": delta,
        "exp_iwt": exp_iwt, "inv_t": inv_t, "si": si, "exp": np.exp, "sqrt": np.sqrt
    }
    sandbox["t"] = t_broad

    try:
        y_broad = eval(func_str, sandbox) * np.exp(1j * phase_rad)
    except Exception as e:
        return {"error": f"Error evaluating function: {e}"}

    mag = np.abs(y_broad)
    if mag.sum() == 0:
        center = 0.0
    else:
        center = float(np.sum(t_broad * mag) / mag.sum())

    t = np.linspace(center - 20, center + 20, 4000)
    dt = t[1] - t[0]

    sandbox["t"] = t
    
    try:
        y = eval(func_str, sandbox) * np.exp(1j * phase_rad)
    except Exception as e:
        return {"error": f"Error evaluating function: {e}"}

    # 3. FFT und Shift
    Yf = np.fft.fft(y) * dt
    Yf_shifted = fftshift(Yf)
    f = fftfreq(len(t), d=dt)
    f_shifted = fftshift(f)

    omega = 2 * np.pi * f_shifted
    t0 = float(t[0])
    Yf_shifted = Yf_shifted * np.exp(-1j * omega * t0)

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
