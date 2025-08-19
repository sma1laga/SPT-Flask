# pages/demos/kapitel4.py
from flask import Blueprint, render_template, request, jsonify, current_app
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
from matplotlib import rcParams
rcParams["text.usetex"] = False
rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
from scipy.io import wavfile
from utils.img import fig_to_base64
from utils.audio import wav_data_url

demos_kapitel4_bp = Blueprint(
    "demos_kapitel4", __name__, template_folder="../../templates"
)

# Notebook options → filenames (put these wavs in static/demos/audio/)
AUDIO_MAP = {
    "Lalala":    "lalalaaa.wav",
    "Elise":     "piano_mono32.wav",
    "Armstrong": "neil_armstrong_mono32.wav",
    "Snare":     "drum_mono32.wav",
}

# Notebook used len_DFT = 4096*4
DEFAULT_LEN_DFT = 4096 * 4

def _audio_path(filename: str) -> str:
    static_root = current_app.static_folder  # .../static
    return os.path.join(static_root, "demos", "audio", filename)

def _load_mono_float(path: str):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float32)
    # remove DC, normalize like in the notebook
    x -= float(np.mean(x))
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > 0:
        x /= m
    return fs, x

def _crop_with_padding(x: np.ndarray, ind_start: int, M: int):
    """Take x[ind_start:ind_start+M]; if it runs out, pad zeros (faithful to 'window of length M')."""
    if ind_start < 0: ind_start = 0
    end = ind_start + M
    if end <= len(x):
        return x[ind_start:end]
    # pad zeros to reach length M
    part = x[ind_start:] if ind_start < len(x) else np.zeros(0, dtype=np.float32)
    pad = np.zeros(M - part.size, dtype=np.float32)
    return np.concatenate([part, pad], axis=0)

@demos_kapitel4_bp.route("/", methods=["GET"])
def page():
    defaults = {
        "x_type": "Lalala",
        "pos": 0.8,
        "len_dft": DEFAULT_LEN_DFT
    }
    return render_template(
        "demos/kapitel4.html",
        audio_options=list(AUDIO_MAP.keys()),
        defaults=defaults
    )

@demos_kapitel4_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        x_type  = (data.get("x_type") or "Lalala").strip()
        pos     = float(data.get("pos", 0.8))
        M       = int(data.get("len_dft", DEFAULT_LEN_DFT))

        filename = AUDIO_MAP.get(x_type)
        if not filename:
            return jsonify(error=f"Unknown x_type: {x_type}"), 400
        path = _audio_path(filename)
        if not os.path.exists(path):
            return jsonify(error=f"Audio file not found: {filename}", path=path), 500

        fs, x = _load_mono_float(path)
        if x.size == 0:
            return jsonify(error="Loaded audio is empty."), 500

        # compute start index like the nb: round(pos*(len(x)-M))
        span = max(len(x) - M, 0)
        ind_start = int(round(pos * span))

        # window x_M and DFT magnitude
        xM = _crop_with_padding(x, ind_start, M)
        X  = np.abs(np.fft.fft(xM))

        # frequency axis (Hz) and keep 0..Nyquist to match usual display
        half = M // 2
        f = fs * np.arange(half) / M
        Xh = X[:half]

        # ------- Plot mirroring the notebook layout -------
        # 2x2 grid with bottom spanning both columns
        fig, axs = plt.subplots(2, 2, figsize=(9.0, 5.4))
        x_axis = axs[0, 0]
        x_crop_axis = axs[0, 1]
        gs = axs[1, 0].get_gridspec()
        for ax in axs[1, :]:
            ax.remove()
        x_DFT_axis = fig.add_subplot(gs[1, :])

        # full x with crop markers
        t = np.arange(len(x)) / fs
        x_axis.plot(t, x, lw=0.5)
        # show the crop window on the full signal
        t0 = ind_start / fs
        t1 = (ind_start + M) / fs
        x_axis.axvline(t0, color="C1", linewidth=2)
        x_axis.axvline(t1, color="C1", linewidth=2)
        x_axis.margins(x=0)
        x_axis.set_title("Input")
        x_axis.set_xlabel("Time [s]")
        x_axis.set_ylabel("x[k]")
        x_axis.grid(True)
        x_axis.set_ylim([-1, 1])

        # cropped window
        tM = np.arange(M) / fs + t0
        x_crop_axis.plot(tM, xM, lw=0.5)
        x_crop_axis.margins(x=0)
        x_crop_axis.set_title(f"Window, M={M}")
        x_crop_axis.set_xlabel("Time [s]")
        x_crop_axis.set_ylabel("x̃[k]")
        x_crop_axis.grid(True)
        x_crop_axis.set_ylim([-1, 1])

        # |DFT| magnitude
        x_DFT_axis.plot(f, Xh, lw=1)
        x_DFT_axis.margins(x=0)
        x_DFT_axis.set_title("Spectrum (Magnitude)")
        x_DFT_axis.set_xlabel("Frequency [Hz]")
        x_DFT_axis.set_ylabel("|X[μ]|")
        x_DFT_axis.grid(True)
        x_DFT_axis.set_xlim(0, fs*25/512)

        png = fig_to_base64(fig)

        # audio: full and cropped (normalize cropped for playback)
        x_audio = wav_data_url(x, fs)
        xM_norm = xM / (np.max(np.abs(xM)) + 1e-12)
        xM_audio = wav_data_url(xM_norm, fs)

        return jsonify({
            "image": png,
            "x_audio": x_audio,
            "xM_audio": xM_audio
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
