from flask import jsonify
from functools import partial
import numpy as np
from scipy.signal import convolve
from utils.math_utils import rect, tri, step, cos, sin, delta, inv_t, si, delta_train
from utils.eval_helpers import safe_eval


functions = [
    ("rect(t)", "rect(t)"),
    ("tri(t)", "tri(t)"),
    ("step(t)", "step(t)"),
    ("sin(\u03c0t)\u22c5step(t)", "sin(t)*step(t)"),
    ("cos(\u03c0t)\u22c5step(t)", "cos(t)*step(t)"),
    ("delta(t)", "delta(t)"),
    ("delta(t-2)", "delta(t-2)"),
    ("delta_train(t)\u22c5step(t)", "delta_train(t)*step(t+0.1)"), # shift +0.1 to avoid half delta at t=0
    ("exp(t)\u22c5step(-t)", "exp(t)*step(-t)"),
    ("exp(-t)\u22c5step(t)", "exp(-t)*step(t)"),
    ("inv_t(t)", "inv_t(t)"),
    ("si(\u03c0t)", "si(pi*t)"),
    ("si^2(\u03c0t)", "si(pi*t)**2"),
]

def conv_json(data: dict):
    """
    Expects JSON { func1, func2 }.

    Returns raw arrays t, y1, y2 (base), y_conv (full convolution).
    """
    f1_str = data.get("func1", "")
    f2_str = data.get("func2", "")

    # safe eval context on the wider grid
    DISP_RANGE = 3.5
    SLIDER_RANGE = 3.5
    t_calc = np.linspace(-4*DISP_RANGE-0.1, 4*DISP_RANGE+0.1, 8001)
    dt = t_calc[1] - t_calc[0]
    ctx = {
        "t": t_calc, "pi": np.pi, "e": np.e,
        "rect": rect, "tri": tri, "step": step,
        "cos": partial(cos, t_norm=np.pi), "sin": partial(sin, t_norm=np.pi),
        "delta": delta, "delta_train": delta_train,
        "inv_t": inv_t, "si": si, "exp": np.exp
    }

    try:
        y1_calc = safe_eval(f1_str, ctx) if f1_str else np.zeros_like(t_calc)
        y2_calc = safe_eval(f2_str, ctx) if f2_str else np.zeros_like(t_calc)
    except Exception as e:
        return jsonify(error=f"Extended‐domain eval error: {e}"), 400

    y_conv_calc = convolve(y1_calc, y2_calc, mode="same") * dt

    # evaluate input signals for display
    t_disp = np.linspace(-DISP_RANGE-SLIDER_RANGE-0.1, DISP_RANGE+SLIDER_RANGE+0.1, 4001)
    ctx_disp = ctx.copy()
    ctx_disp["t"] = t_disp
    try:
        y1_disp = safe_eval(f1_str, ctx_disp) if f1_str else np.zeros_like(t_disp)
    except Exception as e:
        return jsonify(error=f"Function 1 eval error: {e}"), 400
    try:
        y2_disp = safe_eval(f2_str, ctx_disp) if f2_str else np.zeros_like(t_disp)
    except Exception as e:
        return jsonify(error=f"Function 2 eval error: {e}"), 400

    def _scale_delta(expr, arr):
        """Normalize delta-like signals for display while leaving calculations unchanged."""
        expr = expr.strip()
        if "delta" in expr:
            max_val = float(np.max(np.abs(arr)))
            if max_val != 0:
                return arr / max_val
        return arr

    y1_disp = _scale_delta(f1_str, y1_disp)
    y2_disp = _scale_delta(f2_str, y2_disp)
    y_conv_disp = np.interp(t_disp, t_calc, y_conv_calc)
    if "delta" in f1_str.strip() and "delta" in f2_str.strip():
        # convolution of two approximated deltas results in high output values
        y_conv_disp = _scale_delta("delta", y_conv_disp)

    # ——————————————————————————————————————————————————————


    return jsonify({
        "t":     t_disp.tolist(),
        "y1":    y1_disp.tolist(),
        "y2":    y2_disp.tolist(),
        "y_conv": y_conv_disp.tolist()
    })
