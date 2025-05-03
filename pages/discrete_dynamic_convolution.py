from flask import Blueprint, render_template, request, jsonify
import numpy as np
from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

# Blueprint for Discrete Dynamic Convolution
# Implements extended-domain convolution to avoid edge truncation

discrete_dynamic_convolution_bp = Blueprint(
    "discrete_dynamic_convolution",
    __name__,
    template_folder="../templates/discrete"
)

@discrete_dynamic_convolution_bp.route("/", methods=["GET"])
def index():
    # Provide available discrete functions
    functions = [
        "rect(n)", "tri(n)", "step(n)", "sin(n)",
        "cos(n)", "sign(n)", "delta(n)", "exp_iwt(n)",
        "inv_t(n)", "si(n)"
    ]
    return render_template(
        "discrete_dynamic_convolution.html",
        functions=functions
    )

@discrete_dynamic_convolution_bp.route("/data", methods=["POST"])
def data():
    payload = request.get_json(force=True)
    f1_str = payload.get("func1", "").strip()
    f2_str = payload.get("func2", "").strip()

    # Original discrete index
    n = np.arange(-10, 11)
    # Safe evaluation context
    ctx = {"n": n, "np": np,
           "rect": rect, "tri": tri, "step": step,
           "cos": cos, "sin": sin, "sign": sign,
           "delta": delta, "exp_iwt": exp_iwt,
           "inv_t": inv_t, "si": si}

    # Evaluate f1 & f2 on original domain
    try:
        y1 = eval(f1_str, ctx) if f1_str else np.zeros_like(n, dtype=float)
    except Exception as e:
        return jsonify(error=f"Error evaluating Sequence 1: {e}"), 400
    try:
        y2 = eval(f2_str, ctx) if f2_str else np.zeros_like(n, dtype=float)
    except Exception as e:
        return jsonify(error=f"Error evaluating Sequence 2: {e}"), 400

    # Extended domain for convolution: double range
    n_ext = np.arange(2 * n.min(), 2 * n.max() + 1)
    ctx_ext = ctx.copy()
    ctx_ext["n"] = n_ext

    # Evaluate f1 & f2 on extended domain
    try:
        y1_ext = eval(f1_str, ctx_ext) if f1_str else np.zeros_like(n_ext, dtype=float)
        y2_ext = eval(f2_str, ctx_ext) if f2_str else np.zeros_like(n_ext, dtype=float)
    except Exception as e:
        return jsonify(error=f"Extended-domain eval error: {e}"), 400

    # Convolve on extended domain with 'same' to keep alignment
    y_conv_ext = np.convolve(y1_ext, y2_ext, mode="same")

    # Map back to original n for plotting
    # Find indices of original n within extended axis
    idx_map = [int(np.where(n_ext == val)[0]) for val in n]
    y_conv = y_conv_ext[idx_map]

    return jsonify({
        "n":      n.tolist(),
        "y1":     y1.tolist(),
        "y2":     y2.tolist(),
        "y_conv": y_conv.tolist()
    })