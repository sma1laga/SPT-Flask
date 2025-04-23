# pages/dynamic_convolution.py

from flask import Blueprint, render_template, request, jsonify
import numpy as np
from scipy.signal import convolve
from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

dynamic_convolution_bp = Blueprint("dynamic_convolution", __name__)

@dynamic_convolution_bp.route("/", methods=["GET"])
def dynamic_convolution():
    functions = [
        "rect(t)", "tri(t)", "step(t)", "sin(t)", "cos(t)",
        "sign(t)", "delta(t)", "exp_iwt(t)", "exp(t)", "inv_t(t)", "si(t)"
    ]
    return render_template("dynamic_convolution.html", functions=functions)

@dynamic_convolution_bp.route("/data", methods=["POST"])
def dynamic_data():
    """
    Expects JSON { func1, func2 }.
    Returns raw arrays t, y1, y2 (base), y_conv (full convolution).
    """
    data = request.get_json(force=True)
    f1_str = data.get("func1", "")
    f2_str = data.get("func2", "")

    # time axis
    t = np.linspace(-10, 10, 400)
    dt = t[1] - t[0]

    # safe eval context
    local = {
        "t": t, "np": np,
        "rect": rect, "tri": tri, "step": step,
        "cos": cos, "sin": sin, "sign": sign,
        "delta": delta, "exp_iwt": exp_iwt,
        "inv_t": inv_t, "si": si, "exp": np.exp
    }

    # eval f1, f2
    try:
        y1 = eval(f1_str, local) if f1_str else np.zeros_like(t)
    except Exception as e:
        return jsonify(error=f"Function 1 eval error: {e}"), 400
    try:
        y2 = eval(f2_str, local) if f2_str else np.zeros_like(t)
    except Exception as e:
        return jsonify(error=f"Function 2 eval error: {e}"), 400

    # full convolution
    y_conv = convolve(y1, y2, mode="same") * dt

    return jsonify({
        "t":     np.round(t, 4).tolist(),
        "y1":    np.round(y1, 4).tolist(),
        "y2":    np.round(y2, 4).tolist(),
        "y_conv":np.round(y_conv,4).tolist()
    })
