# pages/dynamic_convolution.py

from flask import Blueprint, render_template, request, jsonify
from functools import partial
import numpy as np
from scipy.signal import convolve
from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

dynamic_convolution_bp = Blueprint("dynamic_convolution", __name__)

@dynamic_convolution_bp.route("/", methods=["GET"])
def dynamic_convolution():
    functions = [
        ("rect(t)", "rect(t)"),
        ("tri(t)", "tri(t)"),
        ("step(t)", "step(t)"),
        ("sin(\u03c0t)", "sin(t)"),
        ("cos(\u03c0t)", "cos(t)"),
        ("sign(t)", "sign(t)"),
        ("delta(t)", "delta(t)"),
        ("exp(t)", "exp(t)"),
        ("inv_t(t)", "inv_t(t)"),
        ("si(\u03c0t)", "si(t)")
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

    # display grid limited to [-8, 8]
    t = np.linspace(-8, 8, 3200)
    # use a wider calculation grid so theoretically infinite signals appear flat
    t_calc = np.linspace(-16, 16, 6400)
    dt_calc = t_calc[1] - t_calc[0]
    
    # safe eval context on the wider grid
    local = {
        "t": t_calc, "np": np,
        "rect": rect, "tri": tri, "step": step,
        "cos": partial(cos, t_norm=np.pi), "sin": partial(sin, t_norm=np.pi), "sign": sign,
        "delta": delta, "exp_iwt": exp_iwt,
        "inv_t": inv_t, "si": si, "exp": np.exp
    }

    # eval f1, f2
    try:
        y1 = eval(f1_str, local) if f1_str else np.zeros_like(t_calc)
    except Exception as e:
        return jsonify(error=f"Function 1 eval error: {e}"), 400
    try:
        y2 = eval(f2_str, local) if f2_str else np.zeros_like(t_calc)
    except Exception as e:
        return jsonify(error=f"Function 2 eval error: {e}"), 400
    
    y1 = np.real(y1)
    y2 = np.real(y2)

    # full convolution
        # ——— Extended‐domain convolution to avoid edge truncation ———
    N = len(t_calc)
    t_ext = np.linspace(2*t_calc[0], 2*t_calc[-1], 2*N - 1)
    dt_ext = t_ext[1] - t_ext[0]

    local_ext = local.copy()
    local_ext['t'] = t_ext

    try:
        y1_ext = eval(f1_str, local_ext) if f1_str else np.zeros_like(t_ext)
        y2_ext = eval(f2_str, local_ext) if f2_str else np.zeros_like(t_ext)
    except Exception as e:
        return jsonify(error=f"Extended‐domain eval error: {e}"), 400

    y1_ext = np.real(y1_ext)
    y2_ext = np.real(y2_ext)

    y_conv_ext = convolve(y1_ext, y2_ext, mode="same") * dt_ext
    y_conv_ext = np.real(y_conv_ext)

    # sample back onto the calculation grid then the display grid
    y_conv_calc = np.interp(t_calc, t_ext, y_conv_ext)
    y_conv_calc = np.real(y_conv_calc)

    # resample original signals for display
    y1_disp = np.interp(t, t_calc, y1)
    y2_disp = np.interp(t, t_calc, y2)
    
    def _scale_delta(expr, arr):
        """Normalize delta(t) for display while leaving calculations unchanged."""
        if expr.strip() == "delta(t)":
            max_val = float(np.max(np.abs(arr)))
            if max_val != 0:
                return arr / max_val
        return arr

    y1_disp = _scale_delta(f1_str, y1_disp)
    y2_disp = _scale_delta(f2_str, y2_disp)
    y_conv = np.interp(t, t_calc, y_conv_calc)
    y_conv = np.real(y_conv)

    # ——————————————————————————————————————————————————————


    return jsonify({
        "t":     t.tolist(),
        "y1":    y1_disp.tolist(),
        "y2":    y2_disp.tolist(),
        "y_conv": y_conv.tolist()
    })
