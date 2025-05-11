# pages/discrete_plot_functions.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np

from utils.math_utils import (
    rect, tri, step, delta, dsi       # discrete-time helpers you already have
)

discrete_plot_functions_bp = Blueprint(
    "discrete_plot_functions", __name__, template_folder="templates/discrete"
)

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

    func1_str, func2_str = data.get("func1", ""), data.get("func2", "")

    # sliders (per-function)
    s1  = float(data.get("shift1", 0)); a1 = float(data.get("amp1",   1)); w1 = float(data.get("width1", 1))
    s2  = float(data.get("shift2", 0)); a2 = float(data.get("amp2",   1)); w2 = float(data.get("width2", 1))

    # global sampling slider
    try:
        Δn = float(data.get("sampling", 1))
        if Δn <= 0: raise ValueError
    except ValueError:
        return jsonify({"error": "Δn must be positive."}), 400

    # integer index → sample locations
    MAX_N = 20.0                      # span −20 … +20  (matches continuous page)
    k_max = int(MAX_N / Δn)
    k     = np.arange(-k_max, k_max + 1)
    n     = k * Δn                    # base grid

    # evaluation namespace
    ctx = dict(
        n=n, k=k, np=np,
        rect=rect, tri=tri, step=step, delta=delta, sin=np.sin, cos=np.cos,
        sign=np.sign, si=dsi
    )

    # -------------- evaluate --------------------------------------------------
    try:
        y1 = eval(func1_str, ctx) if func1_str.strip() else np.zeros_like(n)
    except Exception as e:
        return jsonify({"error": f"f₁ error: {e}"}), 400

    y2 = None
    if func2_str.strip():
        try:
            y2 = eval(func2_str, ctx)
        except Exception as e:
            return jsonify({"error": f"f₂ error: {e}"}), 400

    # -------------- transform -------------------------------------------------
    x1 = n * w1 + s1
    y1 = y1 * a1

    if y2 is not None:
        x2 = n * w2 + s2
        y2 = y2 * a2
    else:
        x2 = None

    return jsonify({
        "x1": x1.tolist(), "y1": y1.tolist(),
        "x2": x2.tolist() if x2 is not None else None,
        "y2": y2.tolist() if y2 is not None else None,
        "Δn": Δn
    })
