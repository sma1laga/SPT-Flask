from flask import Blueprint, render_template, request
import io, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

discrete_convolution_bp = Blueprint(
    "discrete_convolution", __name__, template_folder="../templates/discrete"
)

@discrete_convolution_bp.route("/", methods=["GET", "POST"])
def discrete_convolution():
    error     = None
    plot_data = None
    func1_str = ""
    func2_str = ""
    ds_str    = "1.0"

    if request.method == "POST":
        func1_str = request.form.get("func1", "").strip()
        func2_str = request.form.get("func2", "").strip()
        ds_str    = request.form.get("ds",  "1.0").strip()

        # allow decimal sampling interval
        try:
            ds = float(ds_str)
            if ds <= 0:
                raise ValueError("must be > 0")
        except Exception as e:
            error = f"Invalid sampling interval: {e}"

        if error is None:
            result = compute_discrete_convolution(func1_str, func2_str, ds)
            if "error" in result:
                error = result["error"]
            else:
                plot_data = result["plot_data"]

    return render_template(
        "discrete_convolution.html",
        error=error,
        plot_data=plot_data,
        func1=func1_str,
        func2=func2_str,
        ds=ds_str
    )

def compute_discrete_convolution(func1_str, func2_str, ds):
    # build n = -10…10 in steps of ds (allows fractional)
    n = np.arange(-10, 10 + ds, ds)

    safe_ctx = {
        "n": n, "np": np,
        "rect": rect, "tri": tri, "step": step,
        "cos": cos, "sin": sin, "sign": sign,
        "delta": delta, "exp_iwt": exp_iwt, "inv_t": inv_t,
        "si": si
    }

    # evaluate sequence 1
    try:
        y1 = eval(func1_str, safe_ctx) if func1_str else np.zeros_like(n, dtype=float)
    except Exception as e:
        return {"error": f"Error evaluating Sequence 1: {e}"}

    # evaluate sequence 2
    try:
        y2 = eval(func2_str, safe_ctx) if func2_str else np.zeros_like(n, dtype=float)
    except Exception as e:
        return {"error": f"Error evaluating Sequence 2: {e}"}

    # discrete convolution
    y_conv = np.convolve(y1, y2, mode="full")
    n_conv = np.arange(2*n.min(), 2*n.max() + ds, ds)

    # plot
    fig = plt.figure(figsize=(12, 8))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1,1])

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    axc = fig.add_subplot(gs[1,:])

    ax1.stem(n, y1)
    ax1.set_title("Sequence 1")
    ax1.set_xlabel("n")
    ax1.grid(True)

    ax2.stem(n, y2)
    ax2.set_title("Sequence 2")
    ax2.set_xlabel("n")
    ax2.grid(True)

    axc.stem(n_conv, y_conv)
    axc.set_title("Discrete Convolution")
    axc.set_xlabel("n")
    axc.grid(True)

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    return {"plot_data": img_b64}
