# Plotly backend for IIR demo: returns arrays
from flask import Blueprint, render_template, request, jsonify, url_for  
import numpy as np
from io import BytesIO
import base64, wave
import threading
from functools import lru_cache

try:
    from scipy.signal import lfilter as _lfilter
except Exception:
    _lfilter = None

demos_iir_bp = Blueprint("demos_iir", __name__, template_folder="././templates")

AUDIO_MAP = {
    "Vader":      "static/demos/audio/father.wav",
    "Elise":      "static/demos/audio/piano_mono32.wav",
    "Armstrong":  "static/demos/audio/neil_armstrong_mono32.wav",
    "Snare":      "static/demos/audio/drum_mono32.wav",
}

_audio_cache = {}
_preview_lock = threading.Lock()
_last = {"key": None, "payload": None}


@lru_cache(maxsize=64)
def _filter_and_tail_cached(x_key: str, d_q: float, a_q: int):
    """
    Return (fs, x, y_full) for a quantized parameter set.
    - d_q is delay in seconds, snapped to 0.05 steps
    - a_q is attenuation in 0.05 steps, i.e..; a = a_q / 20.0
    Cached per (x_key, d_q, a_q) so repeat visits are instant.
    """
    fs, x = _load_audio(x_key)
    a = a_q / 20.0
    K = max(1, int(round(d_q * fs)))

    # Short tail so echoes are audible on playback/plot
    tail = int(min(3 * fs, 10 * K))
    x_pad = np.pad(x, (0, tail))

    y_full = _iir_echo_filter(x_pad, a, K)
    return fs, x, y_full

def _load_audio(x_type: str):
    import os
    from pathlib import Path
    try:
        from flask import current_app
        base = Path(current_app.root_path)
    except Exception:
        base = Path(os.getcwd())
    if x_type in _audio_cache:
        return _audio_cache[x_type]

    rel = AUDIO_MAP.get(x_type)
    path = (base / rel).resolve() if rel else None
    if path and path.exists():
        try:
            try:
                from scipy.io import wavfile
                fs, data = wavfile.read(str(path))
            except Exception:
                import soundfile as sf  # type: ignore
                data, fs = sf.read(str(path), dtype="float32", always_2d=False)
            if getattr(data, "ndim", 1) == 2:
                data = data[:, 0]
            x = data.astype(np.float32)
            m = float(np.max(np.abs(x)) or 1.0)
            if m > 0:
                x = x / m
            _audio_cache[x_type] = (int(fs), x)
            return _audio_cache[x_type]
        except Exception:
            pass

    # synthetic fallback
    fs = 16000
    t = np.arange(0, int(3.0 * fs)) / fs
    x = 0.6 * np.sin(2 * np.pi * (200 + 800 * t) * t) * np.exp(-2 * t)
    _audio_cache[x_type] = (fs, x.astype(np.float32))
    return _audio_cache[x_type]

def _iir_echo_filter(x: np.ndarray, a: float, K: int) -> np.ndarray:
    """y[n] = x[n] + a*y[n-K] (|a|<1, K>=1)."""
    if K <= 0:
        return x.astype(np.float32, copy=True)
    if _lfilter is not None:
        b = np.array([1.0], dtype=np.float32)
        den = np.zeros(K + 1, dtype=np.float32); den[0] = 1.0; den[K] = -a
        return _lfilter(b, den, x).astype(np.float32)
    y = x.astype(np.float32, copy=True)
    for n in range(K, len(x)):
        y[n] += a * y[n - K]
    return y

def _impulses_for_plot(a: float, K: int, length: int):
    if length <= 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    if K <= 0:
        return np.array([0], dtype=int), np.array([1.0], dtype=np.float32)
    i_max = (length - 1) // K
    if i_max < 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    i = np.arange(i_max + 1, dtype=np.int64)
    k = (i * K).astype(int)
    amps = (a ** i.astype(np.float32))
    mask = amps > 1e-3                   # drop tiny stems (faster)
    return k[mask], amps[mask].astype(np.float32)

def _wav_data_url(fs: int, x: np.ndarray) -> str:
    xx = np.clip(x.astype(np.float32), -1.0, 1.0)
    pcm = (xx * 32767.0).astype("<i2").tobytes()
    bio = BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(fs))
        wf.writeframes(pcm)
    return "data:audio/wav;base64," + base64.b64encode(bio.getvalue()).decode("ascii")

def _downsample(y: np.ndarray, max_pts: int = 4000):
    n = len(y)
    if n <= max_pts:
        return np.arange(n, dtype=int), y
    idx = np.linspace(0, n - 1, max_pts).astype(int)
    idx[0] = 0; idx[-1] = n - 1
    return idx, y[idx]

def _fixed_int_dtick(n: int, target_ticks: int = 6) -> int:
    if n <= 1:
        return 1
    step = int(round((n - 1) / max(1, target_ticks - 1)))
    return max(1, step)

@demos_iir_bp.route("/", methods=["GET"])
def page():
    defaults = {"x_type": "Vader", "delay": 0.50, "a": 0.5}

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
        delay   = float(data.get("delay", 0.50))
        a_float = float(np.clip(float(data.get("a", 0.5)), 0.0, 0.99))
        preview = bool(data.get("preview", False))

        # Quantize to 0.05 steps for stability adn cache keys
        d_q = round(delay * 20) / 20.0           
        a_q = int(round(a_float * 20))           

        # ---------- PREVIEW: short window, arrays only (no audio) ----------
        if preview:
            if not _preview_lock.acquire(blocking=False):
                return jsonify(getattr(update, "_last_preview", {"preview": True}))
            try:
                fs, x = _load_audio(x_key)                       
                K = max(1, int(round(d_q * fs)))
                n = min(int(0.35 * fs), len(x))                  # 350 mS
                xw = x[:n]
                a = a_q / 20.0                                   # use quantized a
                y = _iir_echo_filter(xw, a, K)

                # impulses 
                k_stems, a_stems = _impulses_for_plot(a, K, n)

                # downsample + round 
                kx, x_ds = _downsample(xw, 1200)
                ky, y_ds = _downsample(y, 1200)
                x_ds = np.round(x_ds, 3); y_ds = np.round(y_ds, 3)

                # y limits 
                m = float(np.max(np.abs(y_ds))) if len(y_ds) else 1.0
                ylim_y = 1.0 if not np.isfinite(m) or m <= 1.0 else float(np.ceil(m * 10.0) / 10.0)

                payload = {
                    "preview": True,
                    "fs": fs, "K": K,
                    "k": kx.tolist(),    "x": x_ds.tolist(),
                    "k_y": ky.tolist(),  "y": y_ds.tolist(),
                    "h_k": k_stems.tolist(), "h_amp": a_stems.tolist(),
                    "x_ylim": [-1.0, 1.0], "h_ylim": [0.0, 1.0], "y_ylim": [-ylim_y, ylim_y],
                    "dtick": _fixed_int_dtick(n),
                }
                update._last_preview = payload
                return jsonify(payload)
            finally:
                _preview_lock.release()

        # ---------- FULL RENDER ----------
        key = (x_key, d_q, a_q, "full")
        if _last["key"] == key:
            return jsonify(_last["payload"])

        fs, x, y_full = _filter_and_tail_cached(x_key, d_q, a_q)
        K = max(1, int(round(d_q * fs)))  # for impulses

        n = min(int(round(1.2 * len(x))), len(y_full))
        a = a_q / 20.0
        k_stems, a_stems = _impulses_for_plot(a, K, n)
        if k_stems.size > 1200:
            k_stems = k_stems[:1200]; a_stems = a_stems[:1200]

        kx, x_ds = _downsample(x[:n], 1800)
        ky, y_ds = _downsample(y_full[:n], 1800)
        x_ds = np.round(x_ds, 3); y_ds = np.round(y_ds, 3)

        m = float(np.max(np.abs(y_ds))) if len(y_ds) else 1.0
        ylim_y = 1.0 if not np.isfinite(m) or m <= 1.0 else float(np.ceil(m * 10.0) / 10.0)

        payload = {
            "preview": False,
            "fs": fs, "K": K,
            "k": kx.tolist(),    "x": x_ds.tolist(),
            "k_y": ky.tolist(),  "y": y_ds.tolist(),
            "h_k": k_stems.tolist(), "h_amp": a_stems.tolist(),
            "x_ylim": [-1.0, 1.0], "h_ylim": [0.0, 1.0], "y_ylim": [-ylim_y, ylim_y],
            "dtick": _fixed_int_dtick(n),
            "y_audio": _wav_data_url(fs, y_full),  
        }
        _last.update({"key": key, "payload": payload})
        return jsonify(payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
