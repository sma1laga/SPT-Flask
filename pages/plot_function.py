# pages/plot_function.py
from flask import Blueprint, render_template, request, jsonify
import io, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Added import for ticker
from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

plot_function_bp = Blueprint("plot_function", __name__)

@plot_function_bp.route("/", methods=["GET", "POST"])
def plot_function():
    error = None
    plot_data = None
    func1_str = request.form.get("func1", "") if request.method == "POST" else ""
    func2_str = request.form.get("func2", "") if request.method == "POST" else ""
    try:
        shift1 = float(request.form.get("shift1", 0))
    except:
        shift1 = 0.0
    try:
        stretch1 = float(request.form.get("stretch1", 1))
    except:
        stretch1 = 1.0
    try:
        hscale1 = float(request.form.get("hscale1", 1))
    except:
        hscale1 = 1.0
    try:
        shift2 = float(request.form.get("shift2", 0))
    except:
        shift2 = 0.0
    try:
        stretch2 = float(request.form.get("stretch2", 1))
    except:
        stretch2 = 1.0
    try:
        hscale2 = float(request.form.get("hscale2", 1))
    except:
        hscale2 = 1.0

    transformation_label = ""
    if request.method == "POST":
        result = compute_plot(func1_str, func2_str, shift1, stretch1, hscale1, shift2, stretch2, hscale2)
        if "error" in result:
            error = result["error"]
        else:
            plot_data = result["plot_data"]
            transformation_label = result["transformation_label"]
    return render_template(
        "plot_function.html",
        error=error,
        plot_data=plot_data,
        func1=func1_str,
        func2=func2_str,
        shift1=shift1,
        stretch1=stretch1,
        hscale1=hscale1,
        shift2=shift2,
        stretch2=stretch2,
        hscale2=hscale2,
        transformation_label=transformation_label,
    )

@plot_function_bp.route("/update", methods=["POST"])
def update_plot():
    data = request.get_json(force=True)
    func1_str = data.get("func1", "")
    func2_str = data.get("func2", "")
    try:
        shift1 = float(data.get("shift1", 0))
    except:
        shift1 = 0.0
    try:
        stretch1 = float(data.get("stretch1", 1))
    except:
        stretch1 = 1.0
    try:
        hscale1 = float(data.get("hscale1", 1))
    except:
        hscale1 = 1.0
    try:
        shift2 = float(data.get("shift2", 0))
    except:
        shift2 = 0.0
    try:
        stretch2 = float(data.get("stretch2", 1))
    except:
        stretch2 = 1.0
    try:
        hscale2 = float(data.get("hscale2", 1))
    except:
        hscale2 = 1.0

    result = compute_plot(func1_str, func2_str, shift1, stretch1, hscale1, shift2, stretch2, hscale2)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)

def compute_plot(func1_str, func2_str, shift1, stretch1, hscale1, shift2, stretch2, hscale2):
    t = np.linspace(-10, 10, 2000)
    # Local variables for function 1
    local_vars1 = {
        "t": (t - shift1) / hscale1,
        "np": np,
        "rect": rect,
        "tri": tri,
        "step": step,
        "cos": cos,
        "sin": sin,
        "sign": sign,
        "delta": delta,
        "exp_iwt": exp_iwt,
        "inv_t": inv_t,
        "si": si,
        "exp": np.exp,
    }
    # Local variables for function 2
    local_vars2 = {
        "t": (t - shift2) / hscale2,
        "np": np,
        "rect": rect,
        "tri": tri,
        "step": step,
        "cos": cos,
        "sin": sin,
        "sign": sign,
        "delta": delta,
        "exp_iwt": exp_iwt,
        "inv_t": inv_t,
        "si": si,
        "exp": np.exp,
    }
    try:
        y1 = stretch1 * eval(func1_str, local_vars1) if func1_str.strip() != "" else np.zeros_like(t)
    except Exception as e:
        return {"error": f"Error evaluating Function 1: {e}"}
    if func2_str.strip() != "":
        try:
            y2 = stretch2 * eval(func2_str, local_vars2)
        except Exception as e:
            return {"error": f"Error evaluating Function 2: {e}"}
    else:
        y2 = None

    # Build the plot
    fig, ax = plt.subplots()
    ax.plot(t, y1, label="Function 1", color="blue")
    if y2 is not None:
        ax.plot(t, y2, label="Function 2", color="green")
    ax.set_title("Plot of Functions")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("f(t)")
    ax.grid(True)
    ax.legend()

    # Force whole number ticks on x and y axes
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    transformation_label = (
        f"F1: y = {stretch1:.2f} · f((t - {shift1:.2f}) / {hscale1:.2f})\n"
        f"F2: y = {stretch2:.2f} · f((t - {shift2:.2f}) / {hscale2:.2f})"
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return {"plot_data": plot_data, "transformation_label": transformation_label}
