# pages/discrete_convolution.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
from utils.math_utils import (
    rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si
)

# ──────────────────────────────────────────────────────────────────────────────
# Blueprint
# ──────────────────────────────────────────────────────────────────────────────
discrete_convolution_bp = Blueprint(
    "discrete_convolution",
    __name__,
    template_folder="templates"
)

# ------------------------------------------------------------------ page view
@discrete_convolution_bp.route("/", methods=["GET"])
def discrete_convolution():
    return render_template("discrete_convolution.html")


def compute_discrete_convolution(func1_str, func2_str, ds=1.0):
    """Evaluate two sequences and their discrete convolution on an adaptive axis."""

    n_scan = np.arange(-100, 100 + ds, ds)

    ctx = {
        "n": n_scan, "np": np,
        "rect": rect, "tri": tri, "step": step,
        "cos": cos,   "sin": sin, "sign": sign,
        "delta": delta, "exp_iwt": exp_iwt,
        "inv_t": inv_t, "si": si, "exp": np.exp
    }

    try:
        y1_scan = eval(func1_str, ctx) if func1_str else np.zeros_like(n_scan)
    except Exception as e:
        raise ValueError(f"Error in sequence 1: {e}")
    try:
        y2_scan = eval(func2_str, ctx) if func2_str else np.zeros_like(n_scan)
    except Exception as e:
        raise ValueError(f"Error in sequence 2: {e}")

    amp1 = np.max(np.abs(y1_scan)) if y1_scan.size else 0.0
    amp2 = np.max(np.abs(y2_scan)) if y2_scan.size else 0.0

    def active_limits(y, amp):
        if amp <= 0:
            return None
        mask = np.abs(y) > 0.01 * amp
        if mask.any():
            i0 = np.argmax(mask)
            i1 = len(mask) - np.argmax(mask[::-1]) - 1
            return n_scan[i0], n_scan[i1]
        return None

    r1 = active_limits(y1_scan, amp1)
    r2 = active_limits(y2_scan, amp2)

    if r1 or r2:
        n1_min, n1_max = r1 if r1 else (0.0, 0.0)
        n2_min, n2_max = r2 if r2 else (0.0, 0.0)
        conv_min = n1_min + n2_min
        conv_max = n1_max + n2_max
        n_min = min(n1_min, n2_min, conv_min)
        n_max = max(n1_max, n2_max, conv_max)
    else:
        n_min, n_max = -10.0, 10.0

    margin = 2.0
    n_min -= margin
    n_max += margin

    n = np.arange(n_min, n_max + ds, ds)
    ctx_final = ctx.copy(); ctx_final["n"] = n

    y1 = eval(func1_str, ctx_final) if func1_str else np.zeros_like(n)
    y2 = eval(func2_str, ctx_final) if func2_str else np.zeros_like(n)

    y_conv = np.convolve(y1, y2, mode="full")
    n_conv = np.arange(len(y_conv)) * ds + (n[0] + n[0])

    return {
        "n": n.tolist(),
        "y1": y1.tolist(),
        "y2": y2.tolist(),
        "n_conv": n_conv.tolist(),
        "y_conv": y_conv.tolist(),
        "ds": ds,
    }

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

    try:
        result = compute_discrete_convolution(func1_str, func2_str, ds)
    except Exception as e:
        return jsonify(error=str(e)), 400

    return jsonify(result)

