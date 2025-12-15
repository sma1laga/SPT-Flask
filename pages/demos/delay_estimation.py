"""Delay Estimation (time domain) demo"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Tuple
import warnings


import numpy as np
from flask import Blueprint, jsonify, render_template, request
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning


demos_delay_estimation_bp = Blueprint(
    "demos_delay_estimation", __name__, template_folder="../../templates"
)


FS = 8000
TOTAL_SAMPLES = 100_000
DISP_LEN = 400
MAX_LAG = 200
MAX_DELAY_MS = 50.0
PRECISION = 1000
DEFAULT_SNR = 10.0
DEFAULT_DELAY_MS = 5.0


def _normalize_audio(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x))
    if peak == 0:
        return x.astype(np.float32)
    return (x / peak).astype(np.float32)


@lru_cache(maxsize=1)
def _load_speech_signal() -> np.ndarray:
    """Load the static speech sample"""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=WavFileWarning)
        sr, data = wavfile.read("static/audio/examp1l.wav")
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = _normalize_audio(data.astype(np.float32))

    if sr != FS:
        raise ValueError(
            f"examp1l.wav must be {FS} Hz; got {sr} Hz. Please regenerate the asset."
        )

    start, end = 24772, 124771 + 1
    if len(data) < end:
        raise ValueError(
            "examp1l.wav is too short; expected at least 124771 samples at 8 kHz."
        )

    return data[start:end]


def _generate_base_signals(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    k = np.arange(TOTAL_SAMPLES)
    sinusoid = np.sin(2 * np.pi * (100 / FS) * k).astype(np.float32)
    noise = rng.standard_normal(TOTAL_SAMPLES).astype(np.float32)
    speech = _load_speech_signal()
    return {"noise": noise, "speech": speech, "sinusoid": sinusoid}


_RNG = np.random.default_rng()
_BASE_SIGNALS = _generate_base_signals(_RNG)
_NOISE_1 = _RNG.standard_normal(TOTAL_SAMPLES).astype(np.float32)
_NOISE_2 = _RNG.standard_normal(TOTAL_SAMPLES).astype(np.float32)


def _xcorr_unbiased(
    x: np.ndarray, y: np.ndarray, max_lag: int
) -> Tuple[np.ndarray, np.ndarray]:
    lags = np.arange(-max_lag, max_lag + 1)
    n = len(x)
    corr = np.zeros_like(lags, dtype=np.float32)

    for idx, lag in enumerate(lags):
        if lag >= 0:
            overlap = n - lag
            corr[idx] = float(np.dot(x[:overlap], y[lag:lag + overlap])) / overlap
        else:
            overlap = n + lag
            corr[idx] = float(np.dot(x[-lag:-lag + overlap], y[:overlap])) / overlap

    return lags, corr


def _slice_for_delay(signal1: np.ndarray, signal2: np.ndarray, delay_samples: int):
    """Crop overlapping regions to match the MATLAB offset logic."""

    offset = int(round(MAX_DELAY_MS / 1000.0 * FS)) + 1

    start1 = offset
    end1 = TOTAL_SAMPLES + delay_samples - offset
    start2 = -delay_samples + offset
    end2 = TOTAL_SAMPLES - offset

    return signal1[start1:end1], signal2[start2:end2]


def _prepare_signals(signal_kind: str, snr_db: float, delay_ms: float):
    base_signal = _BASE_SIGNALS.get(signal_kind, _BASE_SIGNALS["noise"])

    signal_variance = float(np.var(base_signal))
    noise_variance = float(signal_variance * 10 ** (-(snr_db / 10.0)))
    noise_scale = math.sqrt(noise_variance)

    signal1 = base_signal + _NOISE_1 * noise_scale
    signal2 = base_signal + _NOISE_2 * noise_scale

    delay_samples = int(round(delay_ms / 1000.0 * FS))
    sliced1, sliced2 = _slice_for_delay(signal1, signal2, delay_samples)

    lags, xcorr = _xcorr_unbiased(sliced1, sliced2, MAX_LAG)

    time_ms = (np.arange(1, DISP_LEN + 1) / FS) * 1000.0
    lag_ms = lags / FS * 1000.0

    return {
        "time_ms": time_ms,
        "signal1": sliced1[:DISP_LEN],
        "signal2": sliced2[:DISP_LEN],
        "lags_ms": lag_ms,
        "xcorr": xcorr,
        "signal_variance": signal_variance,
        "noise_variance": noise_variance,
        "delay_ms": delay_ms,
    }


@demos_delay_estimation_bp.route("/", methods=["GET"])
def page():
    defaults = {
        "snr": DEFAULT_SNR,
        "delay_ms": DEFAULT_DELAY_MS,
        "signal_kind": "noise",
    }
    return render_template("demos/delay_estimation.html", defaults=defaults)


@demos_delay_estimation_bp.route("/compute", methods=["POST"])
def compute():
    payload = request.get_json(force=True) or {}
    snr_db = float(payload.get("snr", DEFAULT_SNR))
    snr_db = round(snr_db * 10) / 10.0
    snr_db = max(-10.0, min(40.0, snr_db))

    delay_ms = float(payload.get("delay_ms", DEFAULT_DELAY_MS))
    delay_ms = round(delay_ms * PRECISION) / PRECISION
    delay_ms = max(-MAX_DELAY_MS, min(MAX_DELAY_MS, delay_ms))

    signal_kind = str(payload.get("signal_kind", "noise"))

    data = _prepare_signals(signal_kind, snr_db, delay_ms)

    return jsonify(
        {
            "time_ms": data["time_ms"].tolist(),
            "signal1": data["signal1"].tolist(),
            "signal2": data["signal2"].tolist(),
            "lags_ms": data["lags_ms"].tolist(),
            "xcorr": data["xcorr"].tolist(),
            "stats": {
                "signal_variance": data["signal_variance"],
                "noise_variance": data["noise_variance"],
                "snr_db": snr_db,
                "delay_ms": delay_ms,
            },
        }
    )