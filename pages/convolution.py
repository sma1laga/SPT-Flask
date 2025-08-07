# pages/convolution.py
from flask import Blueprint, render_template
import numpy as np
from scipy.signal import convolve
from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

convolution_bp = Blueprint("convolution", __name__)

@convolution_bp.route("/", methods=["GET"])
def convolution():
    # Renders the convolution page; plots are generated client-side via AJAX + Plotly
    return render_template("convolution.html")

@convolution_bp.route("/update", methods=["GET", "POST"])
def convolution_update():
    # Deprecated: computation now happens client-side
    return render_template("convolution.html")

def compute_convolution(func1_str, func2_str):
    # 1. Initial wide axis to locate non-zero regions of the input functions.
    #    This prevents large shifts from truncating the result.  The range is
    #    intentionally generous; it will later be trimmed to the active region.
    t_scan = np.linspace(-100.0, 100.0, 8000)
    dt_scan = t_scan[1] - t_scan[0]

    # 2. Safe-Eval-Kontext
    ctx = {
        "t": t_scan, "np": np,
        "rect": rect, "tri": tri, "step": step,
        "cos": cos, "sin": sin, "sign": sign,
        "delta": delta, "exp_iwt": exp_iwt,
        "inv_t": inv_t, "si": si, "exp": np.exp, "sqrt": np.sqrt
    }

    # 3. Auswertung der beiden Funktionen
    try:
        y1_scan = eval(func1_str, ctx) if func1_str.strip() else np.zeros_like(t_scan)
    except Exception as e:
        return {"error": f"Error evaluating Function 1: {e}"}
    try:
        y2_scan = eval(func2_str, ctx) if func2_str.strip() else np.zeros_like(t_scan)
    except Exception as e:
        return {"error": f"Error evaluating Function 2: {e}"}

    # 4. Determine the active region based on the evaluated functions.  Use a
    #    small threshold relative to the maximum amplitude to filter out noise.
    amp1 = np.max(np.abs(y1_scan)) if y1_scan.size else 0.0
    amp2 = np.max(np.abs(y2_scan)) if y2_scan.size else 0.0

    def active_limits(y, amp):
        if amp <= 0:
            return None
        mask = np.abs(y) > 0.01 * amp
        if mask.any():
            i0, i1 = np.argmax(mask), len(mask) - np.argmax(mask[::-1]) - 1
            return t_scan[i0], t_scan[i1]
        return None

    r1 = active_limits(y1_scan, amp1)
    r2 = active_limits(y2_scan, amp2)

    # When a signal touches the scan boundaries it is effectively unbounded.
    # Restrict the display window so edge artefacts remain hidden.
    disp_width = 20.0
    edge_margin = 0.05 * (t_scan[-1] - t_scan[0])

    def adjust_region(r):
        if r is None:
            return (-10.0, 10.0)
        left_touch = r[0] <= t_scan[0] + edge_margin
        right_touch = r[1] >= t_scan[-1] - edge_margin
        if left_touch and right_touch:
            return (-disp_width/2, disp_width/2)
        if left_touch:
            return (r[1] - disp_width, r[1])
        if right_touch:
            return (r[0], r[0] + disp_width)
        return r

    r1 = adjust_region(r1)
    r2 = adjust_region(r2)

    margin = 2.0


    # Axes for each signal individually
    t1_min, t1_max = r1
    t2_min, t2_max = r2
    t1_min -= margin; t1_max += margin
    t2_min -= margin; t2_max += margin

    conv_min = r1[0] + r2[0]
    conv_max = r1[1] + r2[1]
    conv_min -= margin; conv_max += margin

    # Final axes after trimming
    t1 = np.linspace(t1_min, t1_max, 4000)
    t2 = np.linspace(t2_min, t2_max, 4000)
    t_conv = np.linspace(conv_min, conv_max, 4000)

    ctx1 = ctx.copy(); ctx1["t"] = t1
    ctx2 = ctx.copy(); ctx2["t"] = t2
    y1 = eval(func1_str, ctx1) if func1_str.strip() else np.zeros_like(t1)
    y2 = eval(func2_str, ctx2) if func2_str.strip() else np.zeros_like(t2)

    # 4b. Convolution on a wide axis to avoid boundary effects
    t_full = np.linspace(2*t_scan[0], 2*t_scan[-1], 2*len(t_scan) - 1)
    y_conv_full = convolve(y1_scan, y2_scan, mode="full") * dt_scan

    # 5. Interpolate the full convolution back to the trimmed axis
    y_conv = np.interp(t_conv, t_full, y_conv_full)

    # 6. JSON-Ausgabe für Plotly
    return {
        "t1":      t1.tolist(),
        "t2":      t2.tolist(),
        "t_conv":  t_conv.tolist(),
        "y1":      y1.tolist(),
        "y2":      y2.tolist(),
        "y_conv":  y_conv.tolist()
    }
