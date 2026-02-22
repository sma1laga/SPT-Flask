from flask import Blueprint, render_template, request, jsonify
from functools import partial
import numpy as np
from utils.math_utils import (
    rect_N, tri_N, step, cos, sin, sign, delta_n, exp_iwt, inv_t, si, delta_train_n
)
from utils.eval_helpers import safe_eval

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
    functions = [# put function with step before step
        ("rect_4[k]", "rect_4(k)"),
        ("tri_3[k]", "tri_3(k)"),
        ("sin[\u03c0/4\u22c5k]\u22c5step[k]", "sin(np.pi/4*k)*step(k)"),
        ("cos[\u03c0/4\u22c5k]\u22c5step[k]", "cos(np.pi/4*k)*step(k)"),
        ("delta[k]", "delta(k)"),
        ("delta[k-2]", "delta(k-2)"),
        ("delta_train_6[k]\u22c5step[k]", "delta_train(k)*step(k)"),
        ("step[k]", "step(k)"),
        ("sign[k]", "sign(k)"),
        ("exp[k]\u22c5step[-k]", "exp(k)*step(-k)"),
        ("exp[-k]\u22c5step[k]", "exp(-k)*step(k)"),
        ("0.5^k\u22c5step[k]", "0.5**k*step(k)"),
        ("(-0.5)^k\u22c5step[k]", "(-0.5)**k*step(k)"),
        ("inv_k[k]", "inv_k(k)"),
        ("si[\u03c0/2\u22c5k]", "si(np.pi*k/2)"),
        ("si²[\u03c0/2\u22c5k]", "si(np.pi*k/2)**2"),
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
    k = np.arange(-10, 11)
    # Safe evaluation context
    ctx = {
        "k": k, "n": k,
        "rect_4": partial(rect_N, N=4), "tri_3": partial(tri_N, N=3), "step": step,
        "cos": cos, "sin": sin,
        "sign": sign, "delta": delta_n, "delta_train": delta_train_n, "exp": np.exp,
        "inv_k": inv_t, "si": si,
    }

    # Evaluate f1 & f2 on original domain
    try:
        y1 = safe_eval(f1_str, ctx) if f1_str else np.zeros_like(k, dtype=float)
    except Exception as e:
        return jsonify(error=f"Error evaluating Sequence 1: {e}"), 400
    try:
        y2 = safe_eval(f2_str, ctx) if f2_str else np.zeros_like(k, dtype=float)
    except Exception as e:
        return jsonify(error=f"Error evaluating Sequence 2: {e}"), 400

    # Extended domain for convolution: triple range so shifted samples never
    # fall outside the lookup table used by the client. This keeps the
    # visible axis fixed while sequences behave as if they were infinite.
    k_ext = np.arange(3 * k.min(), 3 * k.max() + 1)
    ctx_ext = ctx.copy()
    ctx_ext["k"] = k_ext; ctx_ext["n"] = k_ext

    # Evaluate f1 & f2 on extended domain
    try:
        y1_ext = safe_eval(f1_str, ctx_ext) if f1_str else np.zeros_like(k_ext, dtype=float)
        y2_ext = safe_eval(f2_str, ctx_ext) if f2_str else np.zeros_like(k_ext, dtype=float)
    except Exception as e:
        return jsonify(error=f"Extended-domain eval error: {e}"), 400

    # Convolve on extended domain with 'same' to keep alignment
    y_conv_ext = np.convolve(y1_ext, y2_ext, mode="same")

    # Map back to original n for plotting
    # Find indices of original n within extended axis
    idx_map = [int(np.where(k_ext == val)[0]) for val in k]
    y_conv = y_conv_ext[idx_map]

    return jsonify({
        "k":       k.tolist(),       # plotting grid
        "k_ext":   k_ext.tolist(),   # lookup grid used client-side
        "y1":      y1.tolist(),
        "y2":      y2.tolist(),
        "y1_ext":  y1_ext.tolist(),
        "y2_ext":  y2_ext.tolist(),
        "y_conv":  y_conv.tolist()
    })