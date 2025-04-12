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
    Computes the Fourier transform for the user-supplied function (func_str) and returns a
    figure with three subplots:
      - Original time-domain function (x-axis fixed to [-20, 20])
      - Magnitude spectrum
      - Phase spectrum (unwrapped then re-wrapped into [-π, π])
      
    If show_zeros is True, the function scans the real part of the original function for
    candidate zero (or edge) points, groups them symmetrically (if they exist as pairs), and
    plots the paired zeros using markers of a consistent color along with text annotations.
    """
    import numpy as np
    import io, base64
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft, fftfreq, fftshift

    # 1. Set up a time axis that always spans -20 to 20.
    t = np.linspace(-20, 20, 4000)
    dt = t[1] - t[0]
    
    # 2. Set up the evaluation dictionary (using your math utilities):
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
    
    # 3. Use no window (identity window) for a clean transform:
    window = np.ones_like(t)
    y_windowed = y * window
    
    # 4. Compute the Fourier transform and shift it so frequency 0 is centered.
    Yf = np.fft.fft(y_windowed) * dt
    Yf_shifted = fftshift(Yf)
    f = fftfreq(len(t), d=dt)
    f_shifted = fftshift(f)
    
    # 5. Compute magnitude and phase.
    magnitude = np.abs(Yf_shifted)
    phase_raw = np.angle(Yf_shifted)
    
    # Unwrap phase and re-wrap into [-π, π]
    phase_unwrapped = np.unwrap(phase_raw)
    phase_wrapped = (phase_unwrapped + np.pi) % (2 * np.pi) - np.pi
    
    # Optionally zero out phase where magnitude is very tiny, to avoid noise.
    tiny_threshold = 1e-6
    phase_wrapped[magnitude < tiny_threshold] = 0
    
    # (Optional) Normalize magnitude if desired (comment this out to show absolute scaling)
    if np.max(magnitude) > 0:
        magnitude /= np.max(magnitude)
    
    # 6. Create a figure with three subplots.
    fig, (ax_time, ax_mag, ax_phase) = plt.subplots(1, 3, figsize=(14, 4))
    
    # Left subplot: Original function (time-domain)
    ax_time.plot(t, y.real, label="Real", color='blue')
    if not np.allclose(y.imag, 0, atol=1e-6):
        ax_time.plot(t, y.imag, label="Imag", color='orange', linestyle=':')
    ax_time.axhline(0, color='k', ls='--', lw=0.8)
    ax_time.axvline(0, color='k', ls='--', lw=0.8)
    ax_time.set_title("Original Function")
    ax_time.legend()
    # Force x-axis to be from -20 to 20
    ax_time.set_xlim(-20, 20)
    
    # *** HERE is where we insert the "Show Zeros" logic ***
    if show_zeros:
        # Find candidate zero (edge) points in the real part.
        candidate_edges = find_function_edges(t, y.real, max_crossings=20, threshold_frac=0.01)
        # Group these candidate points into symmetric pairs (by absolute value) within tolerance.
        paired_zeros = group_symmetric_zeros(candidate_edges, tol=0.05)
        # Define a palette of colors to use for each symmetric pair.
        palette = ["magenta", "cyan", "lime", "gold", "purple", "brown", "pink"]
        for i, x_val in enumerate(paired_zeros):
            color = palette[i % len(palette)]
            # For each symmetric pair, plot markers at +x_val and -x_val using linear interpolation for y.
            y_val_pos = np.interp(x_val, t, y.real)
            y_val_neg = np.interp(-x_val, t, y.real)
            ax_time.scatter(x_val, y_val_pos, color=color, s=50, zorder=10)
            ax_time.scatter(-x_val, y_val_neg, color=color, s=50, zorder=10)
            # Annotate the markers with the rounded absolute x-value.
            ax_time.annotate(f"{x_val:.2f}",
                             xy=(x_val, y_val_pos),
                             xytext=(0, -15), textcoords="offset points",
                             ha="center", va="top", fontsize=8, color=color, fontweight="bold")
            ax_time.annotate(f"{x_val:.2f}",
                             xy=(-x_val, y_val_neg),
                             xytext=(0, -15), textcoords="offset points",
                             ha="center", va="top", fontsize=8, color=color, fontweight="bold")
    # *** End of "Show Zeros" logic ***

    # Middle subplot: Magnitude Spectrum
    ax_mag.plot(f_shifted, magnitude, color='red', linewidth=2)
    ax_mag.axhline(0, color='k', ls='--', lw=0.8)
    ax_mag.axvline(0, color='k', ls='--', lw=0.8)
    ax_mag.set_title("Magnitude Spectrum")
    ax_mag.set_xlim(-2, 2)
    ax_mag.legend(["Magnitude"])
    
    # Right subplot: Phase Spectrum
    ax_phase.plot(f_shifted, phase_wrapped, color='green', linewidth=2)
    ax_phase.axhline(0, color='k', ls='--', lw=0.8)
    ax_phase.axvline(0, color='k', ls='--', lw=0.8)
    ax_phase.set_ylim(-np.pi, np.pi)
    ax_phase.set_title("Phase Spectrum")
    ax_phase.set_xlim(-2, 2)
    ax_phase.legend(["Phase"])
    
    buf = io.BytesIO()
    plt.tight_layout()
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