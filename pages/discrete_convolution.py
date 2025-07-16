# pages/discrete_convolution.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
import re
from utils.math_utils import (
    rect_N, tri_N, step, cos, sin, sign, delta_n, exp_iwt, inv_t, si
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

def _rewrite_expr(expr: str) -> str:
    """Rewrite expression to match the discrete symbols/functions.

    Converts:
    - 'rect_4[...]' -> 'rect(..., 4)'
    - 'tri_3[...]' -> 'tri(..., 3)'
    """
    # e.g., replace "rect_5(anystr)" → "rect(anystr, 5)"
    # FIXME: fails for nested brackets, e.g., "rect_5[sin[k]])"
    pattern = r'rect_(\d+)\[([^]]+)\]'
    repl = r'rect(\g<2>, \g<1>)'
    expr_new = re.sub(pattern, repl, expr)
    # e.g., replace "tri_5(anystr)" → "tri(anystr, 5)"
    pattern = r'tri_(\d+)\[([^]]+)\]'
    repl = r'tri(\g<2>, \g<1>)'
    expr_new = re.sub(pattern, repl, expr_new)
    expr_new = expr_new.replace('[', '(').replace(']', ')')
    return expr_new

def compute_discrete_convolution(func1_str, func2_str, ds=1.0):
    """Evaluate two sequences and their discrete convolution on an adaptive axis."""
    func1_str = _rewrite_expr(func1_str)
    func2_str = _rewrite_expr(func2_str)

    k_scan = np.arange(-100, 100 + ds, ds)

    ctx = dict(
        n=k_scan, k=k_scan, np=np,
        pi=np.pi, e=np.e,
        rect=rect_N, tri=tri_N, step=step,
        cos=cos, sin=sin, sign=sign,
        delta=delta_n, exp=np.exp, exp_iwt=exp_iwt,
        si=si, inv_k=inv_t,
    )
        
    try:
        y1_scan = eval(func1_str, ctx) if func1_str else np.zeros_like(k_scan)
    except Exception as e:
        raise ValueError(f"Error in sequence 1: {e}")
    try:
        y2_scan = eval(func2_str, ctx) if func2_str else np.zeros_like(k_scan)
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
            return k_scan[i0], k_scan[i1]
        return None

    r1 = active_limits(y1_scan, amp1)
    r2 = active_limits(y2_scan, amp2)

    if r1 or r2:
        k1_min, k1_max = r1 if r1 else (0.0, 0.0)
        k2_min, k2_max = r2 if r2 else (0.0, 0.0)
        conv_min = k1_min + k2_min
        conv_max = k1_max + k2_max
        k_min = min(k1_min, k2_min, conv_min)
        k_max = max(k1_max, k2_max, conv_max)
    else:
        k_min, k_max = -10.0, 10.0

    margin = 2.0
    k_min -= margin
    k_max += margin

    k = np.arange(k_min, k_max + ds, ds)
    ctx_final = ctx.copy()
    ctx_final["k"] = k
    ctx_final["n"] = k

    y1 = eval(func1_str, ctx_final) if func1_str else np.zeros_like(k)
    y2 = eval(func2_str, ctx_final) if func2_str else np.zeros_like(k)

    y_conv = np.convolve(y1, y2, mode="full")
    k_conv = np.arange(len(y_conv)) * ds + (k[0] + k[0])

    return {
        "k": k.tolist(),
        "y1": y1.tolist(),
        "y2": y2.tolist(),
        "k_conv": k_conv.tolist(),
        "y_conv": y_conv.tolist(),
        "ds": ds,
    }

# ----------------------------------------------------------------- AJAX calc
@discrete_convolution_bp.route("/update", methods=["POST"])
def discrete_convolution_update():
    data       = request.get_json(force=True) or {}
    func1_str  = data.get("func1", "").strip()
    func2_str  = data.get("func2", "").strip()

    try:
        result = compute_discrete_convolution(func1_str, func2_str)
    except Exception as e:
        return jsonify(error=str(e)), 400

    return jsonify(result)

