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
    return (k - shift) / width

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
    s1  = float(data.get("shift1", 0))
    a1 = float(data.get("amp1",   1))
    w1 = float(data.get("width1", 1))
    s2  = float(data.get("shift2", 0))
    a2 = float(data.get("amp2",   1))
    w2 = float(data.get("width2", 1))

    _adjust_k1 = partial(_adjust_k, shift=s1, width=w1)
    _adjust_k2 = partial(_adjust_k, shift=s2, width=w2)

    # calculate signals for t = [(center - MAX_K), (center + MAX_K)]
    # plot around t = [(center - MAX_K/2), (center + MAX_K/2)]
    MAX_K = 40
    center = 0
    k_start = center - MAX_K
    k_end = center + MAX_K
    k = np.arange(k_start, k_end + 1)

    ctx = dict(
        k=_adjust_k1(k), np=np,
        pi=np.pi, e=np.e,
        rect=rect_N, tri=tri_N,
        step=step, delta=delta_n,
        sin=sin, cos=cos,
        sign=sign, si=si,
        exp=np.exp,
    )

    try:
        y1 = a1 * eval(func1_str, ctx) if func1_str.strip() else np.zeros_like(k)
    except Exception as e:
        return jsonify({"error": f"f₁ error: {e}"}), 400

    y2 = None
    if func2_str.strip():
        try:
            ctx["k"] = _adjust_k2(k)
            y2 = a2 * eval(func2_str, ctx)
        except Exception as e:
            return jsonify({"error": f"f₂ error: {e}"}), 400

    return jsonify({
        "x1": k.tolist(),
        "y1": y1.tolist(),
        "x2": k.tolist() if y2 is not None else None,
        "y2": y2.tolist() if y2 is not None else None,
        "xrange": [center - MAX_K/2, center + MAX_K/2]
    })
