# pages/demos/kapitel2.py
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
from scipy.signal import convolve
from utils.img import fig_to_base64
from utils.audio import wav_data_url

demos_kapitel2_bp = Blueprint(
    "demos_kapitel2", __name__, template_folder="../../templates"
)

# Map UI names -> WAV filenames you placed under static/demos/audio/
AUDIO_MAP = {
    "Arnold":     "terminator.wav",
    "Elise":      "piano_mono32.wav",
    "Armstrong":  "neil_armstrong_mono32.wav",
    "Snare":      "drum_mono32.wav",
}

def _audio_path(filename: str) -> str:
    # Use Flask's configured static folder (absolute path)
    static_root = current_app.static_folder  # .../static
    return os.path.join(static_root, "demos", "audio", filename)

def _load_mono_float(path: str):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    # convert to float normalize
    x = x.astype(np.float32)
    mean = float(np.mean(x))
    if not np.isfinite(mean):
        mean = 0.0
    x = x - mean
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > 0:
        x = x / m
    return fs, x

def _impulse_response(fs: int, attenuation: float, delay_sec: float):
    L = max(int(round(1.2 * fs)), 1)
    h = np.zeros(L, dtype=np.float32)
    h[0] = 1.0
    delay_idx = int(round(delay_sec * fs))
    if 0 <= delay_idx < L:
        h[delay_idx] += float(attenuation)
    return h

def _compute(fs, x, h):
    return convolve(h, x, mode="full")

@demos_kapitel2_bp.route("/", methods=["GET"])
def page():
    defaults = {"x_type": "Arnold", "delay": 0.5, "attenuation": 0.5}
    return render_template(
        "demos/kapitel2.html",
        audio_options=list(AUDIO_MAP.keys()),
        defaults=defaults,
    )

@demos_kapitel2_bp.route("/health", methods=["GET"])
def health():
    missing = []
    for name, fn in AUDIO_MAP.items():
        p = _audio_path(fn)
        if not os.path.exists(p):
            missing.append({"label": name, "file": fn, "path": p})
    return jsonify({"missing": missing})

@demos_kapitel2_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        x_type = (data.get("x_type") or "Arnold").strip()
        delay = float(data.get("delay", 0.5))
        attenuation = float(data.get("attenuation", 0.5))

        filename = AUDIO_MAP.get(x_type)
        if not filename:
            return jsonify(error=f"Unknown x_type: {x_type}"), 400

        path = _audio_path(filename)
        if not os.path.exists(path):
            return jsonify(error=f"Audio file not found: {filename}", path=path), 500

        fs, x = _load_mono_float(path)
        if x.size == 0:
            return jsonify(error="Loaded audio is empty."), 500

        h = _impulse_response(fs, attenuation, delay)
        y = _compute(fs, x, h)

        # Plot x, h, y
        fig, axes = plt.subplots(3, 1, figsize=(8, 6))
        t_x = np.arange(len(x))/fs
        t_y = np.arange(len(y))/fs

        axes[0].plot(t_x, x)
        axes[0].margins(x=0)
        axes[0].set_title("Eingang")
        axes[0].set_ylabel(r"x[k]")
        axes[0].set_xlabel("s")

        # h[k] stem plot
        h_nonzero_ind = np.nonzero(h)[0]
        axes[1].vlines(h_nonzero_ind/fs, 0, h[h_nonzero_ind])
        axes[1].plot(h_nonzero_ind/fs, h[h_nonzero_ind], 'o', linewidth=1)
        axes[1].set_ylim(0,2)
        axes[1].set_xlim(0, len(h)/fs)
        axes[1].margins(x=0)
        axes[1].set_title("Impulsantwort")
        axes[1].set_ylabel(r"h[k]")
        axes[1].set_xlabel("s")

        axes[2].plot(t_y, y)
        axes[2].margins(x=0)
        axes[2].set_title("Ausgang")
        axes[2].set_ylabel(r"y[k] = x[k]✳h[k]")
        axes[2].set_xlabel("s")

        for ax in axes:
            ax.grid(True, alpha=0.25)

        png = fig_to_base64(fig)

        # Normalize y for playback
        y_norm = y / (np.max(np.abs(y)) + 1e-12)
        x_audio = wav_data_url(x, fs)
        y_audio = wav_data_url(y_norm, fs)

        return jsonify({"image": png, "x_audio": x_audio, "y_audio": y_audio})

    except Exception as e:
        # Log full traceback return message to the browser
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
