
# pages/demos/filter_demo.py
from flask import Blueprint, render_template, request, jsonify, current_app
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec

from scipy.io import wavfile
from scipy.signal import lfilter, dimpulse

# helper utils provided by the app
from utils.img import fig_to_base64
from utils.audio import wav_data_url, read_mono_audio

# ---------- Demo blueprint ----------
demos_filter_bp = Blueprint("demos_filter", __name__, template_folder="../../templates")

RC_PARAMS = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "font.size": 12,
}

# Audio files are expected under static/demos/audio
AUDIO_MAP = { # TODO images
    "Fanfare":            "fanfare.wav",
    "Arnold":             "terminator.wav",
    "A-Team":             "ateam_short.wav",
    "Piano":              "piano_mono32.wav",
    "Armstrong":          "neil_armstrong_mono32.wav",
    "Snare":              "drum_mono32.wav",
}
IMAGE_MAP = {
    "Jack Sparrow":       "jack.png",
    "Monalisa":  "mona_lisa.png",
    "Trumpets":   "trumpets.png",
}

# EXACT coefficient sets from the notebook (German labels)
EXS = {
    'Task 4.2': {  # warning: unstable
        'a': np.array([1,-2,-5,6], dtype=float),
        'b': np.array([0,1,3,1], dtype=float)},
    'Task 5.1': {
        'a': np.array([1,-.8], dtype=float),
        'b': np.array([1,0], dtype=float)},
    'Task 5.2': {
        'a': np.array([1,-.75,.125], dtype=float),
        'b': np.array([0,0,1], dtype=float)},
    'Task 6.1': {
        'a': np.array([1,0,0], dtype=float),
        'b': np.array([1,-2,2], dtype=float)},
    'Task 6.1 (A)': {
        'a': np.array([1,-1,.5], dtype=float),
        'b': np.array([1,-2,2], dtype=float)},
    'Task 6.1 (M)': {
        'a': np.array([1,0,0], dtype=float),
        'b': np.array([1,-1,.5], dtype=float)},
    'Task 6.3 (TP)': {
        'a': np.array([1,-.5,0], dtype=float),
        'b': np.array([2,0,1], dtype=float)},
    'Task 6.3 (HP)': {
        'a': np.array([1,.5,0], dtype=float),
        'b': np.array([2,0,1], dtype=float)},
    'Task 7.2': {
        'a': np.array([1,-1,.25,0], dtype=float),
        'b': np.array([0,1,1,.25], dtype=float)},
    'Bandpass': {
        'a': np.array([1,0,.81], dtype=float),
        'b': np.array([1,0,-1], dtype=float)},
}

PLOTS = ["Magnitude Response [dB]", "Phase Response [deg]", "Pole-Zero Plot", "Impulse Response"]

def _format_coeff(value: float) -> str:
    """Return a compact string representation for a coefficient."""
    if np.isclose(value, 0.0):
        return "0"
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.3g}"


def _poly_to_tex(coeffs: np.ndarray) -> str:
    """Create a TeX expression for a polynomial in z^{-1}."""
    terms = []
    for k, coeff in enumerate(coeffs):
        if np.isclose(coeff, 0.0):
            continue

        sign = "-" if coeff < 0 else "+"
        coeff_abs = abs(coeff)
        power = "" if k == 0 else f"z^{{-{k}}}"

        if np.isclose(coeff_abs, 1.0) and power:
            coeff_str = ""
        else:
            coeff_str = _format_coeff(coeff_abs)

        term_body = f"{coeff_str}{power}" if coeff_str else power
        terms.append((sign, term_body.strip()))

    if not terms:
        return "0"

    first_sign, first_term = terms[0]
    expression = first_term if first_sign == "+" else f"-{first_term}"

    for sign, term in terms[1:]:
        expression += f" {sign} {term}"

    return expression


def transfer_function_tex(a: np.ndarray, b: np.ndarray) -> str:
    """Return a TeX string for H(z) with the provided coefficients."""
    num = _poly_to_tex(np.asarray(b, dtype=float))
    den = _poly_to_tex(np.asarray(a, dtype=float))
    return rf"H(z) = \frac{{{num}}}{{{den}}}"

# ------------ helpers -------------
def _image_path(filename: str) -> str:
    static_root = current_app.static_folder  # .../static
    return os.path.join(static_root, "demos", "images", filename)

def _load_image(name: str):
    fname = IMAGE_MAP.get(name)
    if not fname:
        raise ValueError(f"Unknown image item: {name}")
    path = _image_path(fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    # read as float in [0,1]; convert to grayscale 
    img = plt.imread(path).astype(np.float64)
    if img.ndim == 3:
        # drop alpha if present
        img = img[..., :3]
        # luminance
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    # ensure finite
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    # normalize to [0,1] if necesarry @paul
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img)
    return img

def _audio_path(filename: str) -> str:
    static_root = current_app.static_folder  # .../static
    return os.path.join(static_root, "demos", "audio", filename)

def _load_audio(name: str):
    fname = AUDIO_MAP.get(name)
    if not fname:
        raise ValueError(f"Unknown audio item: {name}")
    path = _audio_path(fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    fs, x = read_mono_audio(path)
    return fs, x

def get_roots(poly: np.ndarray, tol=1e-7):
    """Return real/imag arrays and multiplicity countssize (used for pole-zero)."""
    if poly is None or np.asarray(poly).size == 0:
        return np.array([]), np.array([]), []
    roots = np.roots(np.asarray(poly, dtype=float)).astype(complex)
    used = np.zeros(roots.size, dtype=bool)
    re_list, im_list, counts = [], [], []
    for i, r in enumerate(roots):
        if used[i]:
            continue
        mask = np.isclose(roots, r, atol=tol, rtol=0)
        used |= mask
        re_list.append(np.real(r))
        im_list.append(np.imag(r))
        counts.append(int(mask.sum()))
    return np.array(re_list), np.array(im_list), counts

def _freq_response(a: np.ndarray, b: np.ndarray):
    """Discrete-time H(e^{jΩ}) with Ω∈[0,π], normalized w_norm∈[0,1]."""
    w_norm = np.linspace(0.0, 1.0, 1025)
    z = np.exp(1j * np.pi * w_norm)
    den = np.polyval(a, z)
    keep = ~np.isclose(den, 0.0)
    z = z[keep]
    w_norm = w_norm[keep]
    H = np.polyval(b, z) / den[keep]
    H[np.isclose(H, 0.0)] = 0.0
    return w_norm, H

def get_mag_dB(a: np.ndarray, b: np.ndarray):
    w_norm, H = _freq_response(a, b)
    mag = np.abs(H)
    keep = mag > 0
    return w_norm[keep], 20.0 * np.log10(mag[keep])

def get_phase(a: np.ndarray, b: np.ndarray):
    w_norm, H = _freq_response(a, b)
    return w_norm, np.angle(H, deg=True)

def impulse_response(a: np.ndarray, b: np.ndarray, n=51):
    k, h = dimpulse((b, a, 1), n=n)
    h = np.asarray(h[0]).squeeze()
    k = np.asarray(k).squeeze()
    return k, h

# ------------- routes --------------

@demos_filter_bp.route("/", methods=["GET"])
def page():
    defaults = {
        "x_type": "Fanfare",
        "ex": "Task 5.1",
        "plot": "Magnitude Response [dB]",
    }
    return render_template(
        "demos/filter_demo.html",
        audio_options=list(AUDIO_MAP.keys()),
        image_options=list(IMAGE_MAP.keys()),
        ex_options=list(EXS.keys()),
        plots=PLOTS,
        defaults=defaults,
    )

@demos_filter_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        x_type   = (data.get("x_type") or "Fanfare").strip()
        ex       = (data.get("ex") or "Task 5.1").strip()
        plot_str = (data.get("plot") or "Magnitude Response [dB]").strip()

        # coefficients strictly from dropdown (no manual input)
        coeffs = EXS.get(ex)
        if coeffs is None:
            raise ValueError(f"Unknown coefficient set: {ex}")
        a = coeffs["a"]
        b = coeffs["b"]
        # load input and filter (audio: 1-D; image: 2-D)
        is_image = False
        fs = None

        if x_type in AUDIO_MAP:
            fs, x = _load_audio(x_type)          # 1-D signal
            # IIR filtering
            y = lfilter(b, a, x)
            # sanitize for plotting/audio
            ymask = np.isfinite(y)
            ymax = np.max(y[ymask])
            ymin = np.min(y[ymask])
            y = np.nan_to_num(y, nan=0, posinf=ymax, neginf=ymin)

        elif x_type in IMAGE_MAP:
            is_image = True
            x = _load_image(x_type)              # 2-D sig array in [0,1]
            # separable IIR: filter along columns, then rows
            # keeps demo  fastas it needs to be
            y = lfilter(b, a, x, axis=1)
            y = lfilter(b, a, y, axis=0)
            # sanitize and normalize for display
            y = np.nan_to_num(y)
            ymask = np.isfinite(y)
            if np.any(ymask):
                y_min, y_max = float(np.min(y[ymask])), float(np.max(y[ymask]))
            else:
                y_min, y_max = 0.0, 1.0
            if y_max > y_min:
                y_disp = (y - y_min) / (y_max - y_min)
            else:
                y_disp = np.zeros_like(y)
        else:
            raise ValueError(f"Unknown input item: {x_type}")

        hz_tex = transfer_function_tex(a, b)

        # plotting
        with plt.rc_context(RC_PARAMS):
            # figure: 3x2 grid (top: x and y; middle: selected plot; bottom: H(z))
            fig = plt.figure(figsize=(10, 7), layout="constrained")
            gs = GridSpec(3, 2, height_ratios=[1.0, 2.2, 0.5], figure=fig)
            x_ax = fig.add_subplot(gs[0, 0])
            y_ax = fig.add_subplot(gs[0, 1])
            plot_ax = fig.add_subplot(gs[1, :])
            hz_ax = fig.add_subplot(gs[2, :])

            hz_ax.axis("off")
            hz_ax.set_title("Transfer Function", fontsize=12, pad=6)
            hz_ax.text(
                0.5,
                0.5,
                rf"${hz_tex}$",
                ha="center",
                va="center",
                fontsize=16,
            )
            if not is_image:
                # ---- audio: time series ----
                x_ax.set_title("Input Signal")
                x_ax.set_aspect('auto')
                x_ax.margins(x=0)
                x_ax.grid(True)
                x_ax.set_xlabel(r"Index $k$ $[\times 10^3]$")
                x_ax.set_ylabel("$x[k]$")
                x_ax.set_ylim(-1.1, 1.1)
                x_ax.plot(np.arange(len(x))/1e3, x, linewidth=0.5)

                y_ax.set_title("Filtered Signal")
                y_ax.set_aspect('auto')
                y_ax.margins(x=0)
                y_ax.grid(True)
                y_ax.set_xlabel(r"Index $k$ $[\times 10^3]$")
                y_ax.set_ylabel("$y[k]$")
                y_ax.plot(np.arange(len(y))/1e3, y, linewidth=0.5)
                if ex != "Task 4.2":
                    ylim = y_ax.get_ylim()
                    ylim_abs = max(abs(ylim[0]), abs(ylim[1]))
                    y_ax.set_ylim(-ylim_abs, ylim_abs)
                else: 
                    y_ax.set_ylim(ymin, ymax)
            else:
                # ---- Image: show grayscale images ----
                x_ax.set_title("Input Image")
                x_ax.imshow(x, cmap="gray", vmin=0, vmax=1, aspect="auto")
                x_ax.axis("off")

                y_ax.set_title("Filtered Image")
                # use normalizedd display version
                y_disp_local = y_disp if 'y_disp' in locals() else y
                y_disp_local = np.clip(y_disp_local, 0.0, 1.0)
                y_ax.imshow(y_disp_local, cmap="gray", vmin=0, vmax=1, aspect="auto")
                y_ax.axis("off")


            # Main plot by selection
            plot_ax.grid(True)

            if plot_str == "Pole-Zero Plot":
                plot_ax.set_xlabel(r"$\mathrm{Re}\{z\}$")
                plot_ax.set_ylabel(r"$\mathrm{Im}\{z\}$")
                plot_ax.set_title(r"Pole-Zero Plot")
                xlim_abs = 3.5
                plot_ax.set_xlim(-xlim_abs, xlim_abs)
                plot_ax.set_xticks(np.arange(-3, 3.5, 1))
                plot_ax.xaxis.set_minor_locator(MultipleLocator(0.5))
                ylim_abs = 1.2
                plot_ax.set_ylim(-ylim_abs, ylim_abs)
                plot_ax.set_yticks(np.arange(-1, 1.5, 0.5))
                plot_ax.set_aspect("equal")
                # unit circle
                an = np.linspace(0, 2*np.pi, 100)
                plot_ax.plot(np.cos(an), np.sin(an), color="gray", linestyle="dotted", linewidth=1)

                # zeros & poles
                re_z, im_z, counts_z = get_roots(b)
                plot_ax.plot(re_z, im_z, 'o', markerfacecolor='none', markersize=6, color='tab:blue', linewidth=1)
                for i, count in enumerate(counts_z):
                    if count > 1:
                        plot_ax.annotate(str(count), (re_z[i], im_z[i]), textcoords="offset points", xytext=(-10, 5), ha='left')

                re_p, im_p, counts_p = get_roots(a)
                plot_ax.plot(re_p, im_p, 'x', markersize=6, color='tab:blue', linewidth=1)
                for i, count in enumerate(counts_p):
                    if count > 1:
                        plot_ax.annotate(str(count), (re_p[i], im_p[i]), textcoords="offset points", xytext=(10, 5), ha='right')

            elif plot_str == "Impulse Response":
                plot_ax.set_title("Impulse Response")
                plot_ax.set_xlabel("Index $k$")
                plot_ax.set_ylabel("$h[k]$")
                plot_ax.set_aspect('auto')
                plot_ax.margins(x=0)
                n = 51
                xlim = (-0.5, n - 0.5)
                plot_ax.set_xlim(*xlim)
                plot_ax.hlines(0, xlim[0], xlim[1], color='black', linewidth=1)
                k, h = impulse_response(a, b, n=n)
                plot_ax.vlines(k, 0, h)
                plot_ax.plot(k, h, 'o', markersize=5)

            elif plot_str == "Phase Response [deg]":
                plot_ax.set_title("Phase Response [deg]")
                plot_ax.set_xlabel(r"Frequency $\Omega$")
                plot_ax.set_ylabel(r"$\mathrm{arg}\{H(\mathrm{e}^{\mathrm{j}\Omega})\}$ [deg]")
                plot_ax.set_aspect('auto')
                plot_ax.set_xlim(0, 1)
                plot_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                plot_ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
                plot_ax.set_ylim(-190, 190)
                plot_ax.set_yticks(np.arange(-180, 181, 45))
                w, ph = get_phase(a, b)
                plot_ax.plot(w, ph, linewidth=1)

            else:  # "Magnitude Response [dB]"
                plot_ax.set_title("Magnitude Response [dB]")
                plot_ax.set_xlabel(r"Frequency $\Omega$")
                plot_ax.set_ylabel(r"$|H(\mathrm{e}^{\mathrm{j}\Omega})|$ [dB]")
                plot_ax.set_aspect('auto')
                plot_ax.set_xlim(0, 1)
                plot_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                plot_ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
                w, mag_db = get_mag_dB(a, b)
                plot_ax.plot(w, mag_db, linewidth=1)

            png = fig_to_base64(fig)

        if not is_image:
            x_audio = wav_data_url(x.astype(np.float32), fs)
            y_norm = y / (np.max(np.abs(y)) + 1e-9)  
            y_audio = wav_data_url(y_norm.astype(np.float32), fs)
        else:
            x_audio = ""
            y_audio = ""

        return jsonify({
            "image": png,
            "x_audio": x_audio,
            "y_audio": y_audio,
        })


    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
