# pages/demos/quantisation.py
from __future__ import annotations

import os
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request

from utils.audio import read_mono_audio, wav_data_url
from utils.img import fig_to_base64


demos_quantisation_bp = Blueprint(
    "demos_quantisation", __name__, template_folder="../../templates"
)

AUDIO_MAP = {
    "Fanfare": "fanfare.wav",
    "Piano (Mono32)": "piano_mono32.wav",
}

BIT_OPTIONS = [16, 10, 7, 4]
MODE_LABELS = {
    "even": "Even (Uniform)",
    "logarithmic": "Logarithmic (μ-law)",
}
MU_LAW_MU = 255.0


def _audio_path(filename: str) -> str:
    return os.path.join(current_app.static_folder, "demos", "audio", filename)


@lru_cache(maxsize=8)
def _load_audio_cached(path: str):
    fs, signal = read_mono_audio(path)
    return fs, signal.astype(np.float32, copy=False)


def _uniform_quantize(x: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits
    scale = float(levels - 1)
    normalized = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
    indices = np.round(normalized * scale)
    quantized = (indices / scale) * 2.0 - 1.0
    return quantized.astype(np.float32, copy=False)


def _mu_law_compress(x: np.ndarray, mu: float = MU_LAW_MU) -> np.ndarray:
    clipped = np.clip(x, -1.0, 1.0)
    return np.sign(clipped) * np.log1p(mu * np.abs(clipped)) / np.log1p(mu)


def _mu_law_expand(y: np.ndarray, mu: float = MU_LAW_MU) -> np.ndarray:
    clipped = np.clip(y, -1.0, 1.0)
    return np.sign(clipped) * (np.expm1(np.abs(clipped) * np.log1p(mu)) / mu)


def _log_quantize(x: np.ndarray, bits: int) -> np.ndarray:
    compressed = _mu_law_compress(x)
    quantized = _uniform_quantize(compressed, bits)
    return _mu_law_expand(quantized)


def _snr_db(x: np.ndarray, y: np.ndarray) -> float:
    error = x - y
    power_signal = float(np.sum(x * x))
    power_error = float(np.sum(error * error))
    if power_error <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(power_signal / power_error)


def _render_plot(fs: int, original: np.ndarray, quantized: np.ndarray):
    total_len = min(len(original), len(quantized))
    N = min(total_len, max(1, int(fs * 0.03)))  # show ≈40 ms
    if total_len <= 0:
        raise ValueError("Audio signal is empty")

    start_idx = 0
    if total_len > N:
        energy = np.square(original[:total_len], dtype=np.float64)
        cumsum = np.concatenate(([0.0], np.cumsum(energy, dtype=np.float64)))
        window_energy = cumsum[N:] - cumsum[:-N]
        start_idx = int(np.argmax(window_energy))

    end_idx = start_idx + N
    t = (np.arange(N) + start_idx) / fs * 1000.0  # ms (absolute time)
    original_window = original[start_idx:end_idx]
    quantized_window = quantized[start_idx:end_idx]
    err = quantized_window - original_window

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.2, 5.2), layout="constrained")

    if len(t):
        ax0.set_title(
            "Waveform ({:.1f}–{:.1f} ms)".format(t[0], t[-1])
        )
    else:
        ax0.set_title("Waveform")
    ax0.set_xlabel("Time [ms]")
    ax0.set_ylabel("Amplitude")
    ax0.grid(True)
    ax0.plot(t, original_window, label="Original", color="black", lw=1.2)
    ax0.plot(t, quantized_window, label="Quantised", color="C1", lw=1.0, alpha=0.85)
    ax0.legend(loc="upper right")

    ax1.set_title("Quantisation Error")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Error")
    ax1.grid(True)
    ax1.plot(t, err, color="C3", lw=1.0)

    return fig_to_base64(fig)


@demos_quantisation_bp.route("/", methods=["GET"])
def page():
    defaults = {"song": "Fanfare", "bits": 16, "mode": "even"}
    return render_template(
        "demos/quantisation.html",
        audio_options=list(AUDIO_MAP.keys()),
        bit_options=BIT_OPTIONS,
        mode_labels=MODE_LABELS,
        defaults=defaults,
    )


@demos_quantisation_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify(error="Invalid JSON payload"), 400

    song = (data.get("song") or "").strip() or "Fanfare"
    bits = int(data.get("bits", 16))
    mode = (data.get("mode") or "even").strip().lower()

    if song not in AUDIO_MAP:
        return jsonify(error=f"Unknown song: {song}"), 400
    if bits not in BIT_OPTIONS:
        return jsonify(error=f"Unsupported bit depth: {bits}"), 400
    if mode not in MODE_LABELS:
        return jsonify(error=f"Unknown quantisation mode: {mode}"), 400

    path = _audio_path(AUDIO_MAP[song])
    if not os.path.exists(path):
        return jsonify(error=f"Audio file not found: {AUDIO_MAP[song]}", path=path), 500

    fs, signal = _load_audio_cached(path)

    if mode == "logarithmic":
        quantised = _log_quantize(signal, bits)
    else:
        quantised = _uniform_quantize(signal, bits)

    snr = _snr_db(signal, quantised)
    img = _render_plot(fs, signal, quantised)

    response = {
        "image": img,
        "x_audio": wav_data_url(signal, fs),
        "y_audio": wav_data_url(quantised, fs),
        "snr_db": snr,
        "levels": int(2 ** bits),
        "step": 2.0 / (2 ** bits - 1),
        "mode_label": MODE_LABELS[mode],
    }

    return jsonify(response)