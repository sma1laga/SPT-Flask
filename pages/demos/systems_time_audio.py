# pages/demos/systems_time_audio.py
from __future__ import annotations
import os, io, base64
import numpy as np
from flask import Blueprint, render_template, request, jsonify, current_app
import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt
from utils.img import fig_to_base64

# SciPy
try:
    from scipy.io import wavfile
    from scipy.signal import convolve
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

demos_systems_time_audio_bp = Blueprint(
    "demos_systems_time_audio", __name__, template_folder="../../templates"
)

# --- Exact notebook options ----------------------------------------------------
X_OPTIONS = ["Elise", "Armstrong", "Snare"]
H_OPTIONS = ["Studio Booth", "Meeting Room", "Office", "Lecture Hall", "Great Hall"]


X_FILES = {
    "Elise": [
        "piano_mono32.wav",            # notebook name
        "elise.wav", "fur_elise.wav",  # fallbacks
    ],
    "Armstrong": [
        "neil_armstrong_mono32.wav",   # notebook name
        "armstrong.wav",
    ],
    "Snare": [
        "drum_mono32.wav",             # notebook name
        "snare_mono.wav", "snare.wav",
    ],
}
H_FILES = {
    "Studio Booth": ["air_booth_mono32.wav"],
    "Meeting Room": ["air_meeting_mono32.wav"],
    "Office": ["air_office_mono32.wav"],
    "Lecture Hall": ["air_lecture_mono32.wav"],
    "Great Hall": ["greatHall_mono32.wav", "air_great_hall_mono32.wav"],
}

# --- Helpers -------------------------------------------------------------------
def _audio_dir():
    return os.path.join(current_app.root_path, "static", "demos", "audio")

def _pick_file(candidates):
    """Return the first existing file path for given candidate basenames, else None."""
    root = _audio_dir()
    for name in candidates:
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
    return None

def _read_wav_like_notebook(path):
    """
    Notebook get_signal(path):
      sr, mono = wavfile.read(path)
      mono = mono.astype('float64')
      diff = max(mono) - min(mono)
      return (sr, mono / (diff/2))
    """
    if not SCIPY_OK:
        raise RuntimeError("SciPy is required (scipy.io.wavfile, scipy.signal.convolve).")
    sr, x = wavfile.read(path)
    # Convert to mono float64
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype("float64")
    low, high = float(np.min(x)), float(np.max(x))
    diff = float(high - low)
    norm = diff / 2.0 if diff > 0 else 1.0
    x = x / norm
    return int(sr), x

def _wav_bytes(fs, y):
    """Encode float mono [-1,1]-ish to 16-bit WAV bytes using SciPy."""
    import io
    from scipy.io import wavfile as _wf
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        y = np.zeros(1, dtype=np.float64)
    # scale/clip
    y = np.clip(y, -1.0, 1.0)
    y16 = (y * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    _wf.write(bio, fs, y16)
    return bio.getvalue()

def _to_data_url(mime, raw_bytes):
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _times_like_notebook(n_samples, fs):
    # np.linspace(0, len(sig)/fs, len(sig)) includes the end; I mirror it
    return np.linspace(0, n_samples / fs, n_samples)

def _make_figure(x_t, x, h_t, h, y_t, y):
    # Notebook used: plt.subplots(1,3, figsize=(config["fig_width"], config["fig_height"]*0.4))
    # We aproximate with a wide, short canvas.
    fig, (ax_x, ax_h, ax_y) = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax in (ax_x, ax_h, ax_y):
        ax.grid(True)
        ax.set_xlabel("$t$")
    ax_x.set_ylabel("$x(t)$")
    ax_h.set_ylabel("$h(t)$")
    ax_y.set_ylabel("$y(t) = x(t) * h(t)$")

    ax_x.set_title("Input signal")
    ax_h.set_title("Impulse response")
    ax_y.set_title("Output signal")

    ax_x.plot(x_t, x, color="red", linewidth=1)
    ax_h.plot(h_t, h, color="blue", linewidth=1)
    ax_y.plot(y_t, y, color="green", linewidth=1)

    plt.tight_layout(pad=2.0)
    return fig

# --- Routes --------------------------------------------------------------------
@demos_systems_time_audio_bp.route("/", methods=["GET"])
def page():
    # Defaults as in the notebook
    defaults = dict(
        x_choice="Elise",
        h_choice="Office",
    )
    return render_template(
        "demos/systems_time_audio.html",
        x_options=X_OPTIONS,
        h_options=H_OPTIONS,
        defaults=defaults,
    )

@demos_systems_time_audio_bp.route("/compute", methods=["POST"])
def compute():
    if not SCIPY_OK:
        return jsonify({"error": "SciPy required (scipy.io.wavfile, scipy.signal.convolve)."}), 500

    data = request.get_json(force=True) or {}
    x_choice = data.get("x_choice", "Elise")
    h_choice = data.get("h_choice", "Office")

    # Resolve files 
    x_path = _pick_file(X_FILES.get(x_choice, []))
    h_path = _pick_file(H_FILES.get(h_choice, []))

    if not x_path:
        return jsonify({"error": f"Audio file for '{x_choice}' not found in static/demos/audio."}), 400
    if not h_path:
        return jsonify({"error": f"Impulse response for '{h_choice}' not found in static/demos/audio."}), 400

    # Load & normalize
    x_fs, x_sig = _read_wav_like_notebook(x_path)
    h_fs, h_sig = _read_wav_like_notebook(h_path)
    if x_fs != h_fs:
        return jsonify({"error": f"Sample rate mismatch: x={x_fs} Hz, h={h_fs} Hz (notebook asserts equality)."}), 400

    # Convolution 
    y_sig = convolve(h_sig, x_sig, mode="full")
    maxabs = np.max(np.abs(y_sig)) if y_sig.size else 1.0
    if maxabs > 0:
        y_sig = y_sig / maxabs

    fs = int(x_fs)
    x_t = _times_like_notebook(len(x_sig), fs)
    h_t = _times_like_notebook(len(h_sig), fs)
    y_t = _times_like_notebook(len(y_sig), fs)

    # Plot
    fig = _make_figure(x_t, x_sig, h_t, h_sig, y_t, y_sig)
    img = fig_to_base64(fig)
    plt.close(fig)

    # Audio players as data URLs
    x_url = _to_data_url("audio/wav", _wav_bytes(fs, x_sig))
    h_url = _to_data_url("audio/wav", _wav_bytes(fs, h_sig))
    y_url = _to_data_url("audio/wav", _wav_bytes(fs, y_sig))

    return jsonify(dict(
        image=img,
        audio_x=x_url,
        audio_h=h_url,
        audio_y=y_url,
    ))
