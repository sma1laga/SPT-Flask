# pages/demos/kapitel8_audio.py (faster,)
from flask import Blueprint, render_template, request, jsonify, current_app
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.io import wavfile
from scipy.signal import convolve, fftconvolve

from utils.img import fig_to_base64
from utils.audio import wav_data_url

demos_kapitel8_audio_bp = Blueprint(
    "demos_kapitel8_audio", __name__, template_folder="../../templates"
)

AUDIO_MAP = {
    "Piano":     "piano_mono32.wav",
    "Fanfare": "fanfare.wav",
}

M_OPTIONS = [16, 32, 64, 128]  # length M
PLOT_WINDOW = 16384            # samples used for spectra plots 
RFFT_TICKS = (0.0, 0.5, 1.0)   # normalized ticks 

def _audio_path(filename: str) -> str:
    return os.path.join(current_app.static_folder, "demos", "audio", filename)

@lru_cache(maxsize=8)
def _load_mono_float_cached(path: str):
    """Load+normalize audio once; cache by path."""
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float32)
    x -= float(np.mean(x))
    m = float(np.max(np.abs(x))) or 1.0
    x /= m
    return fs, x

def _lowpass_h(M: int, fg_norm: float, use_hann: bool):
    """
    Diskreter Tiefpass:
        h[k] = fg_norm * sinc(fg_norm * (k - k0)), k0 = ceil(M/2)
    fg_norm = Ωg/π in (0, 1]
    """
    k = np.arange(-M//2 + 1, M//2 + 1, dtype=np.float32)
    h = fg_norm * np.sinc(fg_norm * k)
    if use_hann:
        h *= np.hanning(M).astype(np.float32)
    return h.astype(np.float32)

def _apply_fast(x, h):
    """Switch to FFT-convolution when the signal is long enough to benefit."""
    N, M = len(x), len(h)
    if N >= 20000:
        return fftconvolve(x, h, mode="same")
    else:
        return convolve(x, h, mode="same")

def _abs_rfft_normed(x):
    """Magnitude of rFFT on a short window for plotting (cheap)."""
    n = min(len(x), PLOT_WINDOW)
    if n <= 0:
        return np.zeros(1)
    X = np.abs(np.fft.rfft(x[:n]))
    return X

def _plot_zoomed_spectrum(ax, X, title):
    """Plot |DFT| und zoome automatisch auf den relevanten Bereich."""
    ax.clear()
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("Ω")
    ax.set_ylabel("Betrag")
    ax.plot(np.arange(len(X)), X, linewidth=0.9)

    e = X * X
    cs = np.cumsum(e)
    tot = cs[-1] + 1e-12
    right = int(np.searchsorted(cs / tot, 0.995)) + 20
    right = max(64, min(right, len(X) - 1))

    ax.set_xlim(0, right)
    ax.margins(x=0)

    ax.set_xticks([0, right // 2, right])
    ax.set_xticklabels(["0", "π/2", "π"])


@demos_kapitel8_audio_bp.route("/", methods=["GET"])
def page():
    defaults = {"x_type": "Piano", "M": 128, "fg": 0.20, "win": True}
    return render_template(
        "demos/kapitel8_audio.html",
        audio_options=list(AUDIO_MAP.keys()),
        defaults=defaults,
        M_options=M_OPTIONS
    )

@demos_kapitel8_audio_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        x_type = (data.get("x_type") or "Piano").strip()
        M      = int(data.get("M") or 128)
        fg     = float(data.get("fg") or 0.20)  # Ωg/π
        win    = bool(data.get("win"))

        if x_type not in AUDIO_MAP:
            return jsonify(error=f"Unknown x_type: {x_type}"), 400
        if M not in M_OPTIONS:
            return jsonify(error=f"Unsupported M: {M}"), 400
        if not (0.0 < fg <= 1.0):
            return jsonify(error=f"fg (Ωg/π) must be in (0, 1], got {fg}"), 400

        path = _audio_path(AUDIO_MAP[x_type])
        if not os.path.exists(path):
            return jsonify(error=f"Audio file not found: {AUDIO_MAP[x_type]}", path=path), 500

        fs, x = _load_mono_float_cached(path)  # cached
        h     = _lowpass_h(M, fg, win)
        y     = _apply_fast(x, h)

        # ---------- plotting lightweight ----------
        fig, axs = plt.subplots(2, 2, figsize=(9.0, 5.0))
        ax_xf, ax_h, ax_yf, ax_H = axs.flatten()

        # DFT(x) (line plot; stem with 16k markers is too heavy)
        X = _abs_rfft_normed(x)
        _plot_zoomed_spectrum(ax_xf, X, "DFT des Originalsignals")
        ax_xf.set_title("DFT des Originalsignals")
        ax_xf.set_xlabel("Ω")
        ax_xf.set_ylabel("Betrag")
        ax_xf.plot(np.arange(len(X)), X, linewidth=0.8)
        # place ticks at 0, π/2, π:
        ax_xf.set_xticks([0, len(X)//2, len(X)-1])
        ax_xf.set_xticklabels(["0", "π/2", "π"])

        # h[k] (M≤128 → stem is cheap, keep it)
        ax_h.grid(True, alpha=0.25)
        ax_h.set_title("Tiefpass (Zeitbereich)")
        ax_h.set_xlabel("k")
        ax_h.set_ylabel("h[k]")
        ax_h.hlines(0, 0, M+1, color='black', linewidth=1)
        ml2, sl2, bl2 = ax_h.stem(np.arange(1, M+1), h, basefmt='none')
        ml2.set_markerfacecolor('none')
        ax_h.set_xlim(0, M+1)

        # DFT(y) (same trick: short window, line)
        Y = _abs_rfft_normed(y)
        _plot_zoomed_spectrum(ax_yf, Y, "DFT des gefilterten Signals")
        ax_yf.set_title("DFT des gefilterten Signals")
        ax_yf.set_xlabel("Ω")
        ax_yf.set_ylabel("Betrag")
        ax_yf.plot(np.arange(len(Y)), Y, linewidth=0.8)
        ax_yf.set_xticks([0, len(Y)//2, len(Y)-1])
        ax_yf.set_xticklabels(["0", "π/2", "π"])

        # |H(e^{jΩ})| (length M -> tiny; plot full rFFT)
        H = np.abs(np.fft.rfft(h))
        ax_H.grid(True, alpha=0.25)
        ax_H.set_title("Tiefpass (Frequenzbereich)")
        ax_H.set_xlabel("Ω")
        ax_H.set_ylabel("|H(e^{jΩ})|")
        ax_H.plot(np.arange(len(H)), H, linewidth=1.0)
        ax_H.set_xticks([0, len(H)//2, len(H)-1])
        ax_H.set_xticklabels(["0", "π/2", "π"])

        fig.tight_layout(h_pad=2.5, pad=2.5)
        png = fig_to_base64(fig)

        # audio (normalize y for playback)
        y_norm = y / (np.max(np.abs(y)) + 1e-12)
        return jsonify({
            "image": png,
            "x_audio": wav_data_url(x, fs),
            "y_audio": wav_data_url(y_norm, fs)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
