
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
from utils.audio import wav_data_url

# ---------- Demo blueprint ----------
demos_filter_bp = Blueprint("demos_filter", __name__, template_folder="../../templates")

# Audio files are expected under static/demos/audio
AUDIO_MAP = {
    "Fanfare":            "fanfare.wav",
    "Arnold":             "terminator.wav",
    "A-Team":             "ateam_short.wav",
    "Piano":              "piano_mono32.wav",
    "Armstrong":          "neil_armstrong_mono32.wav",
    "Snare":              "drum_mono32.wav",
}

# EXACT coefficient sets from the notebook (German labels)
EXS = {
    'Aufgabe 4.2': {  # warning: unstable
        'a': np.array([1,-2,-5,6], dtype=float),
        'b': np.array([0,1,3,1], dtype=float)},
    'Aufgabe 5.1': {
        'a': np.array([1,-.8], dtype=float),
        'b': np.array([1,0], dtype=float)},
    'Aufgabe 5.2': {
        'a': np.array([1,-.75,.125], dtype=float),
        'b': np.array([0,0,1], dtype=float)},
    'Aufgabe 6.1': {
        'a': np.array([1,0,0], dtype=float),
        'b': np.array([1,-2,2], dtype=float)},
    'Aufgabe 6.1 (A)': {
        'a': np.array([1,-1,.5], dtype=float),
        'b': np.array([1,-2,2], dtype=float)},
    'Aufgabe 6.1 (M)': {
        'a': np.array([1,0,0], dtype=float),
        'b': np.array([1,-1,.5], dtype=float)},
    'Aufgabe 6.3 (TP)': {
        'a': np.array([1,-.5,0], dtype=float),
        'b': np.array([2,0,1], dtype=float)},
    'Aufgabe 6.3 (HP)': {
        'a': np.array([1,.5,0], dtype=float),
        'b': np.array([2,0,1], dtype=float)},
    'Aufgabe 7.2': {
        'a': np.array([1,-1,.25,0], dtype=float),
        'b': np.array([0,1,1,.25], dtype=float)},
    'Bandpass': {
        'a': np.array([1,0,.81], dtype=float),
        'b': np.array([1,0,-1], dtype=float)},
}

PLOTS = ["Magnitude response (dB)", "Phase response (deg)", "Pole–zero plot", "Impulse response"]

# ------------ helpers -------------

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
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    # normalize to float32 [-1, 1]
    if "int" in str(x.dtype):
        x = x.astype(np.float32)
        x -= np.mean(x)
        m = np.max(np.abs(x)) + 1e-12
        x = x / m
    else:
        x = x.astype(np.float32)
    return fs, x

def get_roots(poly: np.ndarray, tol=1e-7):
    """Return real/imag arrays and multiplicity counts (used for pole–zero)."""
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
        "ex": "Aufgabe 5.1",
        "plot": "Magnitude response (dB)",
    }
    return render_template(
        "demos/filter_demo.html",
        audio_options=list(AUDIO_MAP.keys()),
        ex_options=list(EXS.keys()),
        plots=PLOTS,
        defaults=defaults,
    )

@demos_filter_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        x_type   = (data.get("x_type") or "Fanfare").strip()
        ex       = (data.get("ex") or "Aufgabe 5.1").strip()
        plot_str = (data.get("plot") or "Magnitude response (dB)").strip()

        # coefficients strictly from dropdown (no manual input)
        coeffs = EXS.get(ex)
        if coeffs is None:
            raise ValueError(f"Unknown coefficient set: {ex}")
        a = coeffs["a"]
        b = coeffs["b"]

        # load audio and filter
        fs, x = _load_audio(x_type)
        y = lfilter(b, a, x)

        # figure: 2x2 grid (top: x and y; bottom: selected plot)
        fig = plt.figure(figsize=(10, 7), layout="constrained")
        gs = GridSpec(2, 2, figure=fig)
        x_ax = fig.add_subplot(gs[0, 0])
        y_ax = fig.add_subplot(gs[0, 1])
        plot_ax = fig.add_subplot(gs[1, :])

        # Input plot
        x_ax.set_title("Input signal")
        x_ax.set_aspect('auto')
        x_ax.margins(x=0)
        x_ax.grid(True)
        x_ax.set_xlabel("Index $k$")
        x_ax.set_ylabel("$x[k]$")
        # keep the notebook's normalization range
        x_ax.set_ylim(-1.1, 1.1)
        x_ax.plot(np.arange(len(x)), x, linewidth=0.5)

        # Output plot
        y_ax.set_title("Filtered signal")
        y_ax.set_aspect('auto')
        y_ax.margins(x=0)
        y_ax.grid(True)
        y_ax.set_xlabel("Index $k$")
        y_ax.set_ylabel("$y[k]$")
        y_ax.plot(np.arange(len(y)), y, linewidth=0.5)
        if "unstable" not in ex.lower():
            ylim = y_ax.get_ylim()
            ylim_abs = max(abs(ylim[0]), abs(ylim[1]))
            y_ax.set_ylim(-ylim_abs, ylim_abs)

        # Main plot by selection
        plot_ax.grid(True)

        if plot_str == "Pole–zero plot":
            plot_ax.set_xlabel(r"$\mathrm{Re}\{z\}$")
            plot_ax.set_ylabel(r"$\mathrm{Im}\{z\}$")
            plot_ax.set_title(r"Pole–zero plot $H(z)$")
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

        elif plot_str == "Impulse response":
            plot_ax.set_title("Impulse response")
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

        elif plot_str == "Phase response (deg)":
            plot_ax.set_title("Phase response (deg)")
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

        else:  # "Magnitude response (dB)"
            plot_ax.set_title("Magnitude response (dB)")
            plot_ax.set_xlabel(r"Frequency $\Omega$")
            plot_ax.set_ylabel(r"$|H(\mathrm{e}^{\mathrm{j}\Omega})|$ [dB]")
            plot_ax.set_aspect('auto')
            plot_ax.set_xlim(0, 1)
            plot_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            plot_ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
            w, mag_db = get_mag_dB(a, b)
            plot_ax.plot(w, mag_db, linewidth=1)

        png = fig_to_base64(fig)
        plt.close(fig)

        # audio players
        x_audio = wav_data_url(x.astype(np.float32), fs)
        y_norm = y / (np.max(np.abs(y)) + 1e-9)  # avoid clipping
        y_audio = wav_data_url(y_norm.astype(np.float32), fs)

        return jsonify({
            "image": png,
            "x_audio": x_audio,
            "y_audio": y_audio,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
