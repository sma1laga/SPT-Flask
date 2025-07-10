# pages/discrete_plot_functions.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np

import utils.math_utils as m


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

    Δn = 1.0


    # broad grid for centre detection
    MAX_N = 20.0
    k_broad = np.arange(-100, 101)
    n_broad = k_broad

    # evaluation namespace with discrete helpers
    ctx_broad = dict(
        n=n_broad, k=k_broad, np=np,
        tri=m.tri_seq, step=m.step, delta=m.delta_n,
        sin=np.sin, cos=np.cos, sign=np.sign, si=m.dsi,
                sqrt=np.sqrt,

    )
    for N in range(1, 33):
        ctx_broad[f"rect_{N}"] = getattr(m, f"rect_{N}")

    # -------------- evaluate on broad grid -----------------------------------
    try:
        y1_broad = eval(func1_str, ctx_broad) if func1_str.strip() else np.zeros_like(n_broad)
    except Exception as e:
        return jsonify({"error": f"f₁ error: {e}"}), 400

    y2_broad = None
    if func2_str.strip():
        try:
            y2_broad = eval(func2_str, ctx_broad)
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

    c1, m1 = center_of_mass(n_broad, y1_broad)
    c2, m2 = center_of_mass(n_broad, y2_broad)

    if c1 is None and c2 is None:
        center = 0.0
    elif c2 is None:
        center = c1
    elif c1 is None:
        center = c2
    else:
        center = (c1 * m1 + c2 * m2) / (m1 + m2)

    # final grid centred around detected centre
    n_start = center - MAX_N
    n_end = center + MAX_N
    k = np.arange(int(round(n_start)), int(round(n_end)) + 1)
    n = k

    ctx = dict(
        n=n, k=k, np=np,
        tri=m.tri_seq, step=m.step, delta=m.delta_n,
        sin=np.sin, cos=np.cos, sign=np.sign, si=m.dsi,
                sqrt=np.sqrt,

    )
    for N in range(1, 33):
        ctx[f"rect_{N}"] = getattr(m, f"rect_{N}")

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
        
    if x2 is not None:
        x_min = float(min(x1.min(), x2.min()))
        x_max = float(max(x1.max(), x2.max()))
    else:
        x_min = float(x1.min())
        x_max = float(x1.max())

    return jsonify({
        "x1": x1.tolist(), "y1": y1.tolist(),
        "x2": x2.tolist() if x2 is not None else None,
        "y2": y2.tolist() if y2 is not None else None,
        "xrange": [x_min, x_max]
        })
