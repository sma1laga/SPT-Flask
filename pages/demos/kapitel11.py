# pages/demos/kapitel11.py
from flask import Blueprint, render_template, request, jsonify, current_app
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample, correlate, butter, filtfilt, get_window

from utils.img import fig_to_base64
from utils.audio import wav_data_url

demos_kapitel11_bp = Blueprint(
    "demos_kapitel11", __name__, template_folder="../../templates"
)

SIG_OPTIONS = ["Sprache", "Meeresrauschen", "Verkehr", "Weißes Rauschen", "Piano"]
PLOT_OPTIONS = [
    ("Zeitverlauf", "zeit"),
    ("Betragsspektrum |X(f)|", "dft"),
    ("Autokorrelation (AKF)", "akf"),
    ("Leistungsdichtespektrum (LDS)", "lds"),
]

# Files we try to use if present
AUDIO_FILES = {
    "Piano":   "piano_mono32.wav",
    "Sprache": "neil_armstrong_mono32.wav",  # fallback to synthetic speech if missing
}
DFT_XLIMS_HZ = {
    "Sprache": 9000,
    "Meeresrauschen": 10000,
    "Verkehr": 8000,
    "Weißes Rauschen": 10000,
    "Piano": 2500,
}

# ----- helpers -----

def _audio_path(filename: str) -> str:
    return os.path.join(current_app.static_folder, "demos", "audio", filename)

def _load_mono_float(path: str):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float32)
    x -= float(np.mean(x))
    m = float(np.max(np.abs(x))) or 1.0
    x /= m
    return fs, x

def _to_pow2_and_fade(x, fs, target_len_pow2=True):
    # Cut to power-of-two length for fast FFTs; add a soft fade-in/out
    n = len(x)
    if target_len_pow2:
        n2 = 1 << int(np.log2(n))
        x = x[:n2]
        n = len(x)
    n_in, n_out = max(64, n // 256), max(256, n // 16)
    win_in = np.hanning(2 * n_in)[:n_in]
    win_out = np.hanning(2 * n_out)[n_out:]
    x[:n_in] *= win_in
    x[-n_out:] *= win_out
    return x

def _synth_white(fs, n):
    return np.random.randn(n).astype(np.float32)

def _synth_pink(fs, n):
    b, a = butter(1, 0.1)  # normalized cutoff ~0.1 of Nyquist
    x = filtfilt(b, a, np.random.randn(n)).astype(np.float32)
    x /= (np.max(np.abs(x)) + 1e-12)
    return x

def _synth_traffic(fs, n):
    t = np.arange(n) / fs
    base = 20 + 10 * np.sin(2*np.pi*0.1*t)
    sig = 0.7*np.sin(2*np.pi*base*t) \
        + 0.2*np.sin(2*np.pi*2*base*t) \
        + 0.1*np.sin(2*np.pi*3*base*t)
    sig += 0.3 * _synth_pink(fs, n)
    sig = sig.astype(np.float32)
    sig /= (np.max(np.abs(sig)) + 1e-12)
    return sig

def _synth_speech_like(fs, n):
    t = np.arange(n) / fs
    env = 0.6 + 0.4*np.sin(2*np.pi*0.7*np.sin(2*np.pi*0.25*t)*t)
    f1, f2, f3 = 180, 600, 1200
    sig = (0.6*np.sin(2*np.pi*f1*t) +
           0.3*np.sin(2*np.pi*f2*t) +
           0.2*np.sin(2*np.pi*f3*t))
    sig += 0.1*np.random.randn(n)
    sig = (env * sig).astype(np.float32)
    sig /= (np.max(np.abs(sig)) + 1e-12)
    return sig

def _prepare_signal(x_type: str):
    """Return (fs, x) normalized and prepared for analysis."""
    if x_type in AUDIO_FILES:
        path = _audio_path(AUDIO_FILES[x_type])
        if os.path.exists(path):
            fs, x = _load_mono_float(path)
        else:
            fs = 32000
            n = 32768
            x = _synth_speech_like(fs, n) if x_type == "Sprache" else _synth_white(fs, n)
    else:
        fs = 32000
        n = 32768
        if x_type == "Weißes Rauschen":
            x = _synth_white(fs, n)
        elif x_type == "Meeresrauschen":
            x = _synth_pink(fs, n)
        elif x_type == "Verkehr":
            x = _synth_traffic(fs, n)
        else:
            x = _synth_white(fs, n)
    x = _to_pow2_and_fade(x, fs, target_len_pow2=True)
    return fs, x

# ---------- frequency-domain helpers (Hz, single-sided) ----------

def _freq_axis_hz(N: int, fs: float):
    """Single-sided frequency axis in Hz to match Jupyter."""
    return np.fft.rfftfreq(N, d=1.0/fs)

def _mag_spectrum_single_sided(x: np.ndarray, fs: float):
    """
    Single-sided |X(f)| with a Hann window and amplitude correction
    so that component frequencies line up and magnitudes are comparable.
    """
    N = len(x)
    win = get_window("hann", N, fftbins=True).astype(np.float32)
    xw = x * win

    # Coherent gain for Hann window (sum(win)/N) to avoid amplitude bias
    cg = np.sum(win) / N

    X = np.fft.rfft(xw)
    f = _freq_axis_hz(N, fs)

    # Single-sided scaling: divide by N and window gain; double non-DC/non-Nyquist bins
    mag = np.abs(X) / (N * cg + 1e-12)
    if N % 2 == 0 and len(mag) > 2:
        mag[1:-1] *= 2.0
    elif len(mag) > 1:
        mag[1:] *= 2.0

    return f, mag

def _periodogram_psd(x: np.ndarray, fs: float):
    """
    Simple periodogram PSD estimate (single-sided) in power/Hz.
    Returns f [Hz], Sxx.
    """
    N = len(x)
    win = get_window("hann", N, fftbins=True).astype(np.float32)
    xw = x * win
    U = (np.sum(win**2) / N)  # window power normalization
    X = np.fft.rfft(xw)
    Sxx = (1.0 / (fs * N)) * (np.abs(X) ** 2) / (U + 1e-12)

    # Single-sided doubling for non-DC/non-Nyquist
    if N % 2 == 0 and len(Sxx) > 2:
        Sxx[1:-1] *= 2.0
    elif len(Sxx) > 1:
        Sxx[1:] *= 2.0

    f = _freq_axis_hz(N, fs)
    return f, Sxx

# ---------- routes ----------

@demos_kapitel11_bp.route("/", methods=["GET"])
def page():
    defaults = {"x_type": "Sprache", "plt": "zeit"}
    return render_template("demos/kapitel11.html",
                           sig_options=SIG_OPTIONS,
                           plot_options=PLOT_OPTIONS,
                           defaults=defaults)

@demos_kapitel11_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        x_type = (data.get("x_type") or "Sprache").strip()
        plt_type = (data.get("plt") or "zeit").strip()

        fs, x = _prepare_signal(x_type)

        if plt_type == "dft":
            # --- Single-sided Betragsspektrum |X(f)| in Hz ---
            N = len(x)
            # Hann-Fenster + amplitude-korrigierte Skalierung 
            win = np.hanning(N).astype(np.float32)
            xw = x * win
            cg = np.sum(win) / N  # coherent gain
            X = np.fft.rfft(xw)
            f_hz = np.fft.rfftfreq(N, d=1.0/fs)

            mag = np.abs(X) / (N * cg + 1e-12)
            if N % 2 == 0 and len(mag) > 2:
                mag[1:-1] *= 2.0   # single-sided Verdopplung, außer DC/Niquist
            elif len(mag) > 1:
                mag[1:] *= 2.0

            # Achsenlimit je nach Signaltyp, aber nie > fs/2
            fmax_req = DFT_XLIMS_HZ.get(x_type, fs/2)
            fmax = min(float(fmax_req), fs/2)

            fig, ax = plt.subplots(figsize=(6.8, 4.0))
            ax.plot(f_hz, mag, linewidth=0.9)
            ax.set_title("Betragsspektrum |X(f)|")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel("|X(f)|")
            ax.set_xlim(0, fmax)

            # sinnvolle Ticks (0, fmax/2, fmax), ohne > fs/2 zu gehen
            ax.set_xticks([0, fmax/2, fmax])
            ax.grid(True, alpha=0.25)


        elif plt_type == "akf":
            from scipy.signal import correlate
            r = correlate(x, x, mode='full')
            lags = np.arange(-len(x)+1, len(x))
            tau = lags / fs
            fig, ax = plt.subplots(figsize=(6.8, 4.0))
            ax.plot(tau, r, linewidth=0.9)
            ax.set_title("Autokorrelation (AKF)")
            ax.set_xlabel("τ [s]")
            ax.set_ylabel("r[τ]")
            ax.grid(True, alpha=0.25)

        elif plt_type == "lds":
            f, Sxx = _periodogram_psd(x, fs)
            fig, ax = plt.subplots(figsize=(6.8, 4.0))
            ax.plot(f, Sxx, linewidth=0.9)
            ax.set_title("Leistungsdichtespektrum (LDS)")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel("Sxx(f) [Power/Hz]")
            ax.set_xlim(0, fs/2)
            ax.set_xticks([0, fs/4, fs/2])
            ax.grid(True, alpha=0.25)

        else:
            t = np.arange(len(x)) / fs
            fig, ax = plt.subplots(figsize=(6.8, 4.0))
            ax.plot(t, x, linewidth=0.9)
            ax.set_title("Zeitverlauf")
            ax.set_xlabel("t [s]")
            ax.set_ylabel("x[k]")
            ax.grid(True, alpha=0.25)

        png = fig_to_base64(fig)
        x_audio = wav_data_url(x, fs)
        return jsonify({"image": png, "x_audio": x_audio})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=str(e)), 500
