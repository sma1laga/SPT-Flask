# pages/convolution.py
from flask import Blueprint, render_template, request, jsonify
import io, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Import your math utility functions for safe eval:
from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

convolution_bp = Blueprint("convolution", __name__)

@convolution_bp.route("/", methods=["GET", "POST"])
def convolution():
    """
    GET: Renders a page with two input fields for function1, function2.
    POST: Computes the convolution and returns a Base64 plot with two subplots
          for the input functions and one large subplot for the convolution.
    """
    error = None
    plot_data = None
    func1_str = ""
    func2_str = ""

    if request.method == "POST":
        func1_str = request.form.get("func1", "")
        func2_str = request.form.get("func2", "")
        result = compute_convolution(func1_str, func2_str)
        if "error" in result:
            error = result["error"]
        else:
            plot_data = result["plot_data"]

    return render_template("convolution.html",
                           error=error,
                           plot_data=plot_data,
                           func1=func1_str,
                           func2=func2_str)

@convolution_bp.route("/update", methods=["POST"])
def convolution_update():
    """
    POST /update route for optional live updates (AJAX).
    Expects JSON with 'func1' and 'func2'.
    Returns JSON { plot_data, error? }.
    """
    data = request.get_json(force=True)
    func1_str = data.get("func1", "")
    func2_str = data.get("func2", "")

    result = compute_convolution(func1_str, func2_str)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)

def compute_convolution(func1_str, func2_str):
    """
    Evaluates func1, func2 in [-10,10], does discrete convolution,
    and returns a 2x2 subplot figure:
      - Top row: f1(t) in left subplot, f2(t) in right subplot
      - Bottom row: convolution in a subplot spanning both columns
    f1, f2 in different shades of blue, convolution in a darker blue.
    Returns { plot_data=<base64> } or { error=... } if fails.
    """
    t = np.linspace(-10, 10, 400)
    dt = t[1] - t[0]

    # Safe dictionaries for function eval
    local_eval = {
        "t": t,
        "np": np,
        "rect": rect, "tri": tri, "step": step,
        "cos": cos, "sin": sin, "sign": sign,
        "delta": delta, "exp_iwt": exp_iwt, "inv_t": inv_t,
        "si": si, "exp": np.exp
    }
    try:
        if func1_str.strip():
            y1 = eval(func1_str, local_eval)
        else:
            y1 = np.zeros_like(t)
    except Exception as e:
        return {"error": f"Error evaluating Function 1: {e}"}

    try:
        if func2_str.strip():
            y2 = eval(func2_str, local_eval)
        else:
            y2 = np.zeros_like(t)
    except Exception as e:
        return {"error": f"Error evaluating Function 2: {e}"}

    # Convolution
    y_conv = convolve(y1, y2, mode='same') * dt

    # Build a figure with a 2x2 layout, but the bottom row is a single subplot spanning columns
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1,1])

    ax_f1 = fig.add_subplot(gs[0, 0])
    ax_f2 = fig.add_subplot(gs[0, 1])
    ax_conv = fig.add_subplot(gs[1, :])  # merges across columns 0 and 1

    # Plot f1 (light blue)
    ax_f1.plot(t, y1, color="cornflowerblue", linewidth=2)
    ax_f1.axhline(0, color='k', ls='--', lw=0.8)
    ax_f1.axvline(0, color='k', ls='--', lw=0.8)
    ax_f1.set_title("Function 1")
    ax_f1.set_xlim(-10, 10)
    ax_f1.grid(ls='--', alpha=0.7)

    # Plot f2 (medium blue)
    ax_f2.plot(t, y2, color="blue", linewidth=2)
    ax_f2.axhline(0, color='k', ls='--', lw=0.8)
    ax_f2.axvline(0, color='k', ls='--', lw=0.8)
    ax_f2.set_title("Function 2")
    ax_f2.set_xlim(-10, 10)
    ax_f2.grid(ls='--', alpha=0.7)

    # Plot the convolution (dark blue)
    ax_conv.plot(t, y_conv, color="darkblue", linewidth=2)
    ax_conv.axhline(0, color='k', ls='--', lw=0.8)
    ax_conv.axvline(0, color='k', ls='--', lw=0.8)
    ax_conv.set_title("Convolution")
    ax_conv.set_xlim(-10, 10)
    ax_conv.grid(ls='--', alpha=0.7)

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    return {"plot_data": plot_data}
