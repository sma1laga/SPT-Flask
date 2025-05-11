# pages/discrete_convolution.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
from utils.math_utils import (
    rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si
)

# ──────────────────────────────────────────────────────────────────────────────
# Blueprint
# ──────────────────────────────────────────────────────────────────────────────
discrete_convolution_bp = Blueprint("discrete_convolution", __name__,
                                    template_folder="templates")

# ------------------------------------------------------------------ page view
@discrete_convolution_bp.route("/", methods=["GET"])
def discrete_convolution():
    return render_template("discrete_convolution.html")

# ----------------------------------------------------------------- AJAX calc
@discrete_convolution_bp.route("/update", methods=["POST"])
def discrete_convolution_update():
    data       = request.get_json(force=True) or {}
    func1_str  = data.get("func1", "").strip()
    func2_str  = data.get("func2", "").strip()

    # optional sampling step  Δn
    try:
        ds = float(data.get("ds", 1.0))
        if ds <= 0:
            raise ValueError("Δn must be > 0")
    except Exception as e:
        return jsonify(error=f"Invalid Δn: {e}"), 400

    # base index axis  n = -20 … 20
    n = np.arange(-20, 20 + ds, ds)

    ctx = {
        "n": n, "np": np,
        "rect": rect, "tri": tri, "step": step,
        "cos": cos,   "sin": sin, "sign": sign,
        "delta": delta, "exp_iwt": exp_iwt,
        "inv_t": inv_t, "si": si, "exp": np.exp
    }

    # evaluate sequences
    try:
        y1 = eval(func1_str, ctx) if func1_str else np.zeros_like(n)
    except Exception as e:
        return jsonify(error=f"Error in sequence 1: {e}"), 400
    try:
        y2 = eval(func2_str, ctx) if func2_str else np.zeros_like(n)
    except Exception as e:
        return jsonify(error=f"Error in sequence 2: {e}"), 400

    # discrete convolution
    y_conv = np.convolve(y1, y2, mode="full")
    n_conv = np.arange(len(y_conv)) * ds + (n[0] + n[0])  # start at n_min+n_min

    return jsonify(
        n=n.tolist(),
        y1=y1.tolist(),
        y2=y2.tolist(),
        n_conv=n_conv.tolist(),
        y_conv=y_conv.tolist(),
        ds=ds
    )
