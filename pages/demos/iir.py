# Plotly backend for IIR demo: returns arrays
from flask import Blueprint, render_template, request, jsonify, url_for
import os
from pathlib import Path
import numpy as np
from scipy.signal import convolve

from utils.audio import read_mono_audio, wav_data_url

demos_iir_bp = Blueprint("demos_iir", __name__, template_folder="././templates")

AUDIO_MAP = {
    "Vader":      "static/demos/audio/father.wav",
    "Elise":      "static/demos/audio/piano_mono32.wav",
    "Armstrong":  "static/demos/audio/neil_armstrong_mono32.wav",
    "Snare":      "static/demos/audio/drum_mono32.wav",
}

_audio_cache = {}

def _filter(x: np.ndarray, K: int, a: float):
    if K == 0 or a == 0:
        return x.astype(np.float32, copy=True)
    h = np.zeros(2*len(x)+1) # output length: 3 * input length
    h[::K] = np.power(a, np.arange((len(h)-1)//K + 1))
    return convolve(h, x).astype(np.float32)

def _load_audio(x_type: str):
    try:
        from flask import current_app
        base = Path(current_app.root_path)
    except Exception:
        base = Path(os.getcwd())
    if x_type in _audio_cache:
        return _audio_cache[x_type]

    path = (base / AUDIO_MAP.get(x_type)).resolve()
    fs, x = read_mono_audio(str(path))
    _audio_cache[x_type] = (int(fs), x.astype(np.float32))
    return _audio_cache[x_type]

def _clip_fs(x: np.ndarray, fs: int, max_fs=16000):
    """downsample to max_fs if needed, only integer factors"""
    if fs >= 2*max_fs:
        factor = fs // max_fs
        x = x[::factor]
        fs = fs // factor
    return fs, x

def _impulses_for_plot(a: float, K: int, length: int):
    if length <= 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    if K <= 0:
        return np.array([0], dtype=int), np.array([1.0], dtype=np.float32)
    i_max = (length - 1) // K
    i = np.arange(i_max + 1, dtype=np.int64)
    k = (i * K).astype(int)
    amps = (a ** i.astype(np.float32))
    mask_nonzero = np.nonzero(amps)
    return k[mask_nonzero], amps[mask_nonzero]


@demos_iir_bp.route("/", methods=["GET"])
def page():
    defaults = {"x_type": "Vader", "delay": 0.5, "a": 0.5}

    audio_map_abs = {
        name: url_for("static", filename=path.split("static/", 1)[1])
        if path.startswith("static/") else path
        for name, path in AUDIO_MAP.items()
    }

    return render_template(
        "demos/iir.html",
        defaults=defaults,
        audio_choices=list(AUDIO_MAP.keys()),
        audio_map=audio_map_abs, 
    )
    
@demos_iir_bp.route("/update", methods=["POST"])
def update():
    try:
        data = request.get_json(force=True) or {}
        x_key   = data.get("x_type", "Vader")
        d   = float(data.get("delay", 0.5))
        a = float(np.clip(float(data.get("a", 0.5)), 0, 1))

        # ---------- RENDER ----------
        fs, x = _load_audio(x_key)
        fs, x = _clip_fs(x, fs, max_fs=16000) # 32kHz or higher will be downsampled
        K = int(round(d * fs))
        y_full = _filter(x, K, a)

        plt_len = int(round(1.2 * len(x)))
        k_stems, amp_stems = _impulses_for_plot(a, K, plt_len)
        k = np.arange(plt_len, dtype=int)
        payload = {
            "k": k[:len(x)].tolist(),    "x": x.tolist(),
            "k_y": k.tolist(),  "y": y_full[:plt_len].tolist(),
            "h_k": k_stems.tolist(), "h_amp": amp_stems.tolist(),
            "x_ylim": [-1.5, 1.5], "h_ylim": [0.0, 1.1], "y_ylim": [-1.5, 1.5],
            "xrange": [0, plt_len-1],
            "y_audio": wav_data_url(y_full, fs),  
        }
        return jsonify(payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
