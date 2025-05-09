# pages/convolution.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
from scipy.signal import convolve
from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

convolution_bp = Blueprint("convolution", __name__)

@convolution_bp.route("/", methods=["GET"])
def convolution():
    # Renders the convolution page; plots are generated client-side via AJAX + Plotly
    return render_template("convolution.html")

@convolution_bp.route("/update", methods=["POST"])
def convolution_update():
    data = request.get_json(force=True)
    func1_str = data.get("func1", "")
    func2_str = data.get("func2", "")
    result = compute_convolution(func1_str, func2_str)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)

def compute_convolution(func1_str, func2_str):
    # 1. Basis-Zeitachse
    t = np.linspace(-10, 10, 400)
    dt = t[1] - t[0]

    # 2. Safe-Eval-Kontext
    ctx = {
        "t": t, "np": np,
        "rect": rect, "tri": tri, "step": step,
        "cos": cos, "sin": sin, "sign": sign,
        "delta": delta, "exp_iwt": exp_iwt,
        "inv_t": inv_t, "si": si, "exp": np.exp
    }

    # 3. Auswertung der beiden Funktionen
    try:
        y1 = eval(func1_str, ctx) if func1_str.strip() else np.zeros_like(t)
    except Exception as e:
        return {"error": f"Error evaluating Function 1: {e}"}
    try:
        y2 = eval(func2_str, ctx) if func2_str.strip() else np.zeros_like(t)
    except Exception as e:
        return {"error": f"Error evaluating Function 2: {e}"}

    # 4. Erweiterte Achse, um Rand-Effekte zu vermeiden
    N = len(t)
    t_ext = np.linspace(2*t[0], 2*t[-1], 2*N - 1)
    dt_ext = t_ext[1] - t_ext[0]
    ctx_ext = ctx.copy(); ctx_ext["t"] = t_ext
    y1_ext = eval(func1_str, ctx_ext) if func1_str.strip() else np.zeros_like(t_ext)
    y2_ext = eval(func2_str, ctx_ext) if func2_str.strip() else np.zeros_like(t_ext)

    # 5. Faltung & Rückinterpolation
    y_conv_ext = convolve(y1_ext, y2_ext, mode="same") * dt_ext
    y_conv = np.interp(t, t_ext, y_conv_ext)

    # 6. JSON-Ausgabe für Plotly
    return {
        "t":       t.tolist(),
        "y1":      y1.tolist(),
        "y2":      y2.tolist(),
        "y_conv":  y_conv.tolist()
    }
