# pages/fourier_page.py
from flask import Blueprint, render_template, request, jsonify
import io, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from utils.math_utils import rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si

fourier_bp = Blueprint("fourier", __name__)

@fourier_bp.route("/", methods=["GET", "POST"])
def fourier():
    error = None
    plot_data = None
    func_str = ""
    phase_deg = 0.0
    show_zeros = False

    if request.method == "POST":
        func_str = request.form.get("func", "")
        try:
            phase_deg = float(request.form.get("phase", 0))
        except:
            phase_deg = 0.0
        show_zeros = (request.form.get("showZeros") == "on")  # e.g. checkbox

        result = compute_fourier(func_str, np.deg2rad(phase_deg), show_zeros)
        if "error" in result:
            error = result["error"]
        else:
            plot_data = result["plot_data"]

    return render_template("fourier.html",
                           error=error,
                           plot_data=plot_data,
                           func=func_str,
                           phase=phase_deg,
                           show_zeros=show_zeros)

@fourier_bp.route("/update", methods=["POST"])
def update_fourier():
    data = request.get_json(force=True)
    func_str = data.get("func", "")
    try:
        phase_deg = float(data.get("phase", 0))
    except:
        phase_deg = 0.0
    show_zeros = data.get("showZeros", False)

    result = compute_fourier(func_str, np.deg2rad(phase_deg), show_zeros)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)

def compute_fourier(func_str, phase_rad, show_zeros=False):
    """
    Compute Fourier transform and generate a 3-panel plot.
    The original function is plotted with x-axis fixed from -20 to 20.
    If show_zeros is True, the function's real part is scanned for zero (or near-zero)
    transitions. Candidate zeros are grouped into symmetric pairs (within tolerance) and
    each pair is marked on the plot with a matching color. Each marker is annotated
    with its (rounded) absolute x-coordinate. This way, if the zeros are away from 0,
    the labels will "follow" their markers.
    """
    # Always plot from -20 to 20:
    t = np.linspace(-20, 20, 2000)
    dt = t[1] - t[0]
    local_vars = {
        "t": t,
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
        "exp": np.exp
    }
    try:
        y = eval(func_str, local_vars) * np.exp(1j * phase_rad)
    except Exception as e:
        return {"error": f"Error evaluating function: {e}"}
    
    # Compute Fourier Transform
    window = np.hanning(len(t))
    y_windowed = y * window
    Yf = fft(y_windowed) * dt
    freqs = fftfreq(len(t), dt)
    omega = 2 * np.pi * freqs
    omega_norm = omega / (2 * np.pi)
    magnitude = np.abs(Yf)
    if np.max(magnitude) > 0:
        magnitude /= np.max(magnitude)
    phase_spectrum = np.angle(Yf)
    phase_spectrum[magnitude < 1e-6] = 0

    # Create figure with three subplots (figure size increased for readability)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    
    # --- Plot Original Function (ax1) ---
    y_re = y.real
    y_im = y.imag
    plotted = False
    if not np.allclose(y_re, 0, atol=1e-6):
        ax1.plot(t, y_re, color="blue", linewidth=2, label="Real")
        plotted = True
    if not np.allclose(y_im, 0, atol=1e-6):
        ax1.plot(t, y_im, color="orange", linestyle=":", linewidth=2, label="Imag")
        plotted = True
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_title("Original Function")
    ax1.set_xlabel("t")
    if plotted:
        ax1.legend()
    
    # Always fix x-axis from -20 to 20.
    ax1.set_xlim(-20, 20)

    # If "Show Zeros" is requested and the function has a real component, detect and annotate zeros.
    if show_zeros and plotted:
        candidate_edges = find_function_edges(t, y_re, max_crossings=20, threshold_frac=0.01)
        paired_zeros = group_symmetric_zeros(candidate_edges, tol=0.05)
        # Define a palette of colors to differentiate pairs
        palette = ["magenta", "cyan", "lime", "gold", "purple", "brown", "pink"]
        for i, x_val in enumerate(paired_zeros):
            color = palette[i % len(palette)]
            # Compute the y-value via interpolation for both positive and negative
            y_val_pos = np.interp(x_val, t, y_re)
            y_val_neg = np.interp(-x_val, t, y_re)
            # Plot markers on both sides
            ax1.scatter(x_val, y_val_pos, color=color, s=50, zorder=10)
            ax1.scatter(-x_val, y_val_neg, color=color, s=50, zorder=10)
            # Annotate near the markers; place the label below each marker
            ax1.annotate(f"{x_val:.2f}",
                         xy=(x_val, y_val_pos),
                         xytext=(0, -15), textcoords="offset points",
                         ha="center", va="top", color=color, fontsize=8, fontweight="bold")
            ax1.annotate(f"{x_val:.2f}",
                         xy=(-x_val, y_val_neg),
                         xytext=(0, -15), textcoords="offset points",
                         ha="center", va="top", color=color, fontsize=8, fontweight="bold")

    # --- Plot Magnitude Spectrum (ax2) ---
    ax2.plot(omega_norm, magnitude, color="red", linewidth=2, label="Magnitude")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_title("Magnitude Spectrum")
    ax2.set_xlim(-2, 2)
    ax2.legend()

    # --- Plot Phase Spectrum (ax3) ---
    ax3.plot(omega_norm, phase_spectrum, color="green", linewidth=2, label="Phase")
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.set_title("Phase Spectrum")
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-np.pi, np.pi)
    ax3.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    
    transformation_label = f"Phase Shift: {np.rad2deg(phase_rad):.2f}°"
    return {"plot_data": plot_data, "transformation_label": transformation_label}

def find_function_edges(t, y_re, max_crossings=10, threshold_frac=0.01):
    """
    A heuristic to find candidate zero or step-edge locations in y_re.
    Returns up to max_crossings candidate x-values.
    """
    edges = []
    if len(t) < 2 or np.max(np.abs(y_re)) < 1e-12:
        return np.array(edges)
    y_absmax = np.max(np.abs(y_re))
    step_thresh = threshold_frac * y_absmax
    for i in range(len(t) - 1):
        y1, y2 = y_re[i], y_re[i+1]
        if y1 * y2 < 0:
            frac = -y1 / (y2 - y1)
            x_cross = t[i] + frac * (t[i+1] - t[i])
            edges.append(x_cross)
        else:
            if (abs(y1) < step_thresh and abs(y2) > step_thresh) or (abs(y2) < step_thresh and abs(y1) > step_thresh):
                x_edge = (t[i] + t[i+1]) / 2
                edges.append(x_edge)
        if len(edges) >= max_crossings:
            break
    return np.array(edges)


def group_symmetric_zeros(zeros, tol=0.05):
    """
    Given candidate zero x-values (possibly positive and negative),
    group them into symmetric pairs by comparing absolute values.
    Returns an array of representative positive x-values for each pair.
    """
    if len(zeros) == 0:
        return np.array([])
    zeros = np.array(zeros)
    pos = zeros[zeros >= 0]
    neg = zeros[zeros < 0]
    pos = np.sort(pos)
    neg = np.sort(-neg)  # absolute values of negative zeros
    groups = []
    for xp in pos:
        if np.any(np.abs(neg - xp) < tol):
            groups.append(xp)
    groups = np.unique(np.round(groups, 2))
    return np.sort(groups)

def group_symmetric_zeros(zeros, tol=0.05):
    """
    Given an array of candidate zero/edge x-values (which may be positive or negative),
    group them into symmetric pairs if the absolute differences between a positive value
    and a negative value (in absolute terms) are within tol.
    Returns a sorted array of representative positive x-values (i.e. the absolute distance
    from zero) for those pairs.
    """
    zeros = np.array(zeros)
    # Separate positive and negative values:
    pos = zeros[zeros >= 0]
    neg = zeros[zeros < 0]
    pos = np.sort(pos)
    neg = np.sort(-neg)  # sort absolute values of negatives in ascending order
    groups = []
    # For each positive candidate, check if there is a corresponding negative candidate 
    # with a similar absolute value.
    for xp in pos:
        # Look for a negative candidate with similar absolute value.
        diffs = np.abs(neg - xp)
        if np.any(diffs < tol):
            groups.append(xp)
    groups = np.array(groups)
    groups = np.sort(groups)
    return groups