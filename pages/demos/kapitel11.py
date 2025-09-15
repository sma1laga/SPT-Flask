# pages/demos/kapitel11.py
from flask import Blueprint, render_template, request, jsonify, current_app
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt
from scipy.signal import resample, correlate

from utils.img import fig_to_base64
from utils.audio import wav_data_url, read_mono_audio

demos_kapitel11_bp = Blueprint(
    "demos_kapitel11", __name__, template_folder="../../templates"
)

SIG_OPTIONS = ["Speech", "Ocean Sounds", "Traffic", "White Noise", "Piano"]
PLOT_OPTIONS = [
    ("Time Domain", "time"),
    ("Magnitude Spectrum", "dft"),
    ("Autocorrelation (ACF)", "acf"),
    ("Power Spectral Density (PSD)", "psd"),
]

# Files we try to use if present
AUDIO_FILES = {
    "Speech": "hollera.wav",
    "Ocean Sounds":   "welle_kurz.wav",
    "Traffic":   "strasse_kurz.wav",
    "White Noise":   "white_noise.wav",
    "Piano":   "piano_mono32.wav",
}

# ----- helpers -----
def _audio_path(filename: str) -> str:
    return os.path.join(current_app.static_folder, "demos", "audio", filename)

def _cut_pow2(x):
    n_2 = 2**int(np.log2(len(x)))
    return x[:n_2]

def _fade(x, n:list, type:str='out'):
    assert type in ['in', 'out', 'inout']
    assert len(n) < 3
    n_in, n_out = n if len(n)==2 else (n[0], n[0])

    x_faded = np.array(x)
    if 'in' in type:
        x_faded[:n_in] *= np.hanning(2*n_in)[:n_in]
    if 'out' in type:
        x_faded[-n_out:] *= np.hanning(2*n_out)[n_out:]
    return x_faded

def _prepare_signal(x_type: str):
    """Return (fs, x) normalized and prepared for analysis."""
    path = _audio_path(AUDIO_FILES[x_type])
    if os.path.exists(path):
        fs, x = read_mono_audio(path)
    else:
        raise FileNotFoundError(f"File not found: {AUDIO_FILES[x_type]}")
    if x_type == "Piano": # cut/resample audio for performance
        x = _cut_pow2(x)
        start = np.argmax(np.abs(x) > 0)
        x = x[start:start+len(x)//2]
        x = resample(x, len(x)//4)
        fs //= 4
        x = _fade(x, [100, len(x)//8], type='inout')
    return fs, x

def _abs_rfft(x, fs=None):
    """Frequencies and magnitudes of real-valued FFT."""
    if fs is None:
        return np.abs(np.fft.rfft(x))
    return np.abs(np.fft.rfft(x)), np.fft.rfftfreq(len(x), d=1.0/fs)

def _acf(x):
    """Calculate autocorrelation of x."""
    return correlate(x, x, mode='full')

# ---------- routes ----------
@demos_kapitel11_bp.route("/", methods=["GET"])
def page():
    defaults = {"x_type": "Speech", "plt": "time"}
    return render_template("demos/kapitel11.html",
                           sig_options=SIG_OPTIONS,
                           plot_options=PLOT_OPTIONS,
                           defaults=defaults)

@demos_kapitel11_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        x_type = (data.get("x_type", "Speech")).strip()
        plt_type = (data.get("plt", "time")).strip()

        fs, x = _prepare_signal(x_type)

        fig, ax = plt.subplots(figsize=(6.8, 4.0), layout="constrained")
        ax.grid(True)
        if plt_type == "dft":
            n_fft = len(x)
            mag, f_hz = _abs_rfft(x, fs)
            ax.plot(f_hz[:n_fft//4+2], mag[:n_fft//4+2], linewidth=0.5)
            ax.set_title("Magnitude Spectrum")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("$|X(f)|$")

            if x_type == "Piano":
                # get current xticks and labels
                xticks = ax.get_xticks()
                xticklabels = ax.get_xticklabels()
                note_hz = 400
                print(xticks, xticklabels)
                xticks[2] = note_hz  # add A4 for piano
                xticklabels[2] = f"{note_hz} ≙ A4"
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
            ax.margins(x=0, y=0)
        elif plt_type == "acf":
            acf = correlate(x, x, mode='full')
            lag_ms = np.arange(-len(x)+1, len(x)) / fs * 1000  # in ms
            ax.plot(lag_ms, acf/acf.max())
            ax.set_xlim(-25, 25)
            ax.margins(x=0)
            ax.set_title("Normalized Autocorrelation Function (ACF)")
            ax.set_xlabel("Time-Lag [ms]")
            ax.set_ylabel(r"$\varphi_{xx}(\tau) / \varphi_{xx}(\tau=0)$")
        elif plt_type == "psd":
            psd = _abs_rfft(_acf(x))
            ax.plot(np.linspace(0, np.pi/2, len(psd)), psd, linewidth=0.5)
            ax.set_title("Power Spectral Density (PSD)")
            ax.set_xlabel(r"$\Omega$")
            ax.set_ylabel(r"$|\Phi_{xx}(\mathrm{e}^{\mathrm{j}\Omega})|$")
            ax.set_xticks([0, np.pi/4, np.pi/2])
            ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"])
            ax.set_ylim(0,2e6)
            ax.margins(x=0)
        else: # time
            t_ms = np.arange(len(x)) / fs * 1e3
            start, length = 10000, 5000
            ax.plot(t_ms[start:start+length], x[start:start+length], linewidth=0.5)
            ax.margins(x=0)
            ax.set_title("Amplitude")
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("$x[k]$")
            ax.set_ylim(-1, 1)

        png = fig_to_base64(fig)
        x_audio = wav_data_url(x, fs)
        return jsonify({"image": png, "x_audio": x_audio})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=str(e)), 500
