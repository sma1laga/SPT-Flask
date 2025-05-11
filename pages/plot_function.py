# pages/plot_function.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
from utils.math_utils import (
    rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si
)

plot_function_bp = Blueprint("plot_function", __name__,
                             template_folder="templates")


@plot_function_bp.route("/", methods=["GET"])
def plot_function():
    return render_template("plot_function.html")


@plot_function_bp.route("/update", methods=["POST"])
def plot_function_update():
    data = request.get_json(force=True) or {}

    func1_str, func2_str = data.get("func1", ""), data.get("func2", "")

    # f₁ sliders ----------------------------------------------------
    s1  = float(data.get("shift1", 0))
    a1  = float(data.get("amp1",   1))
    w1  = float(data.get("width1", 1))

    # f₂ sliders ----------------------------------------------------
    s2  = float(data.get("shift2", 0))
    a2  = float(data.get("amp2",   1))
    w2  = float(data.get("width2", 1))

    # base grid
    t = np.linspace(-20, 20, 4000)

    ns = dict(t=t, np=np, rect=rect, tri=tri, step=step, cos=cos, sin=sin,
              sign=sign, delta=delta, exp_iwt=exp_iwt, inv_t=inv_t, si=si, exp=np.exp)

    # evaluate, catching errors individually
    try:
        y1 = eval(func1_str, ns) if func1_str.strip() else np.zeros_like(t)
    except Exception as e:
        return jsonify({"error": f"Error in f₁(t): {e}"}), 400

    y2 = None
    if func2_str.strip():
        try:
            y2 = eval(func2_str, ns)
        except Exception as e:
            return jsonify({"error": f"Error in f₂(t): {e}"}), 400

    # apply separate transforms
    t1 = t * w1 + s1
    y1 = y1 * a1

    if y2 is not None:
        t2 = t * w2 + s2
        y2 = y2 * a2
    else:
        t2 = None

    return jsonify({
        "t1": t1.tolist(), "y1": y1.tolist(),
        "t2": t2.tolist() if t2 is not None else None,
        "y2": y2.tolist() if y2 is not None else None
    })
