# pages/discrete_plot_functions.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
import re
from functools import partial
from utils.math_utils import (
    rect_N, tri_N, step, cos, sin, sign, delta_n, si
)


discrete_plot_functions_bp = Blueprint(
    "discrete_plot_functions", __name__, template_folder="templates/discrete"
)

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

def _adjust_k(k: np.ndarray, shift: float, width: float) -> np.ndarray:
    """Adjust k values based on shift and width (1/width * k - shift)."""
    return k / width - shift

# -------------------------------------------------------------------- #
# page
# -------------------------------------------------------------------- #
@discrete_plot_functions_bp.route("/", methods=["GET"], endpoint="plot_functions")
def discrete_plot_functions():
    return render_template("discrete_plot_functions.html")


# -------------------------------------------------------------------- #
# AJAX update
# -------------------------------------------------------------------- #
@discrete_plot_functions_bp.route("/update", methods=["POST"],
                                   endpoint="plot_functions_update")
def discrete_plot_functions_update():
    data = request.get_json(force=True) or {}

    func1_str = _rewrite_expr(data.get("func1", ""))
    func2_str = _rewrite_expr(data.get("func2", ""))

    # sliders (per-function)
    s1  = float(data.get("shift1", 0)); a1 = float(data.get("amp1",   1)); w1 = float(data.get("width1", 1))
    s2  = float(data.get("shift2", 0)); a2 = float(data.get("amp2",   1)); w2 = float(data.get("width2", 1))

    _adjust_k1 = partial(_adjust_k, shift=s1, width=w1)
    _adjust_k2 = partial(_adjust_k, shift=s2, width=w2)

    MAX_N = 20. # plot xlim will be [(center - MAX_N), (center + MAX_N)]
    # broad grid for centre detection
    k_broad = np.linspace(-100, 101)

    # evaluation namespace with discrete helpers
    ctx_broad = dict(
        n=_adjust_k1(k_broad), k=_adjust_k1(k_broad), np=np,
        pi=np.pi, e=np.e,
        rect=rect_N, tri=tri_N,
        step=step, delta=delta_n,
        sin=sin, cos=cos,
        sign=sign, si=si,
        exp=np.exp,
    )

    # -------------- evaluate on broad grid -----------------------------------
    try:
        y1_broad = a1 * eval(func1_str, ctx_broad) if func1_str.strip() else np.zeros_like(k_broad)
    except Exception as e:
        return jsonify({"error": f"f₁ error: {e}"}), 400
    y2_broad = None
    if func2_str.strip():
        try:
            ctx_broad["n"] = _adjust_k2(k_broad)
            ctx_broad["k"] = _adjust_k2(k_broad)
            y2_broad = a2 * eval(func2_str, ctx_broad)
        except Exception as e:
            return jsonify({"error": f"f₂ error: {e}"}), 400

    def center_of_mass(x, y):
        if y is None:
            return None, 0
        mag = np.abs(y)
        tot = np.sum(mag)
        if tot == 0:
            return None, 0
        return float(np.sum(x * mag) / tot), tot

    c1, m1 = center_of_mass(k_broad, y1_broad)
    c2, m2 = center_of_mass(k_broad, y2_broad)

    if c1 is None and c2 is None:
        center = 0.0
    elif c2 is None:
        center = c1
    elif c1 is None:
        center = c2
    else:
        center = (c1 * m1 + c2 * m2) / (m1 + m2)

    # final grid centred around detected centre
    k_start = center - MAX_N
    k_end = center + MAX_N
    k = np.arange(int(round(k_start)), int(round(k_end)) + 1)

    ctx = dict(
        n=_adjust_k1(k), k=_adjust_k1(k), np=np,
        pi=np.pi, e=np.e,
        rect=rect_N, tri=tri_N, step=step, delta=delta_n,
        sin=sin, cos=cos, sign=sign, si=si,
    )

    try:
        y1 = a1 * eval(func1_str, ctx) if func1_str.strip() else np.zeros_like(k)
    except Exception as e:
        return jsonify({"error": f"f₁ error: {e}"}), 400

    y2 = None
    if func2_str.strip():
        try:
            ctx["n"] = _adjust_k2(k)
            ctx["k"] = _adjust_k2(k)
            y2 = a2 * eval(func2_str, ctx)
        except Exception as e:
            return jsonify({"error": f"f₂ error: {e}"}), 400

    return jsonify({
        "x1": k.tolist(), "y1": y1.tolist(),
        "x2": k.tolist() if y2 is not None else None,
        "y2": y2.tolist() if y2 is not None else None,
        "xrange": [k_start, k_end]
        })
