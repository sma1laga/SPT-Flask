"""Harmonic detection"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from flask import Blueprint, jsonify, render_template, request


demos_harmonic_detection_bp = Blueprint(
    "demos_harmonic_detection", __name__, template_folder="../../templates"
)

FS = 8000
TONE_FREQ = 100.0
TOTAL_SAMPLES = 100_000
DISP_LEN = 235
MAX_LAG = 150


@dataclass
class HarmonicData:
    time_ms: np.ndarray
    harmonic: np.ndarray
    noise: np.ndarray
    combined: np.ndarray
    lags: np.ndarray
    acorr: np.ndarray
    norm_acorr: np.ndarray
    freqs: np.ndarray
    psd_db: np.ndarray
    signal_variance: float
    noise_variance: float
    sample_count: int
    snr_db: float


def _generate_base_signals(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    k = np.arange(TOTAL_SAMPLES)
    harmonic = np.sin(2 * np.pi * (TONE_FREQ / FS) * k)
    noise = rng.standard_normal(TOTAL_SAMPLES)
    return harmonic, noise


_RNG = np.random.default_rng()
_BASE_HARMONIC, _BASE_NOISE = _generate_base_signals(_RNG)


def _biased_autocorrelation(x: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute biased autocorrelation limited to +-max_lag"""

    full_corr = np.correlate(x, x, mode="full") / len(x)
    mid = len(full_corr) // 2
    start = max(mid - max_lag, 0)
    end = min(mid + max_lag + 1, len(full_corr))
    lags = np.arange(start - mid, end - mid)
    segment = full_corr[start:end]

    max_val = np.max(segment)
    norm_segment = segment / max_val if max_val != 0 else segment
    return lags, segment, norm_segment


def _power_spectral_density(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.fft.fft(x)
    n = len(x)
    pyy = y * np.conj(y) / n
    half = n // 2 + 1
    freqs = FS / 2 * np.arange(half) / max(half - 1, 1)
    pyy_half = np.maximum(np.real(pyy[:half]), 1e-12)
    psd_db = 10 * np.log10(pyy_half)
    return freqs, psd_db


def _prepare_data(snr_db: float, sample_count: int) -> HarmonicData:
    harmonic, white_noise = _BASE_HARMONIC, _BASE_NOISE

    signal_variance = float(np.var(harmonic))
    noise_variance = float(signal_variance * 10 ** (-(snr_db / 10.0)))

    scaled_noise = white_noise * math.sqrt(noise_variance)
    combined = harmonic + scaled_noise

    sample_count = max(100, min(int(sample_count), TOTAL_SAMPLES))
    windowed = combined[:sample_count]

    lags, acorr, norm_acorr = _biased_autocorrelation(windowed, MAX_LAG)
    freqs, psd_db = _power_spectral_density(windowed)

    time_ms = (np.arange(DISP_LEN) / FS) * 1000.0

    return HarmonicData(
        time_ms=time_ms,
        harmonic=harmonic[:DISP_LEN],
        noise=scaled_noise[:DISP_LEN],
        combined=combined[:DISP_LEN],
        lags=lags / FS * 1000.0,
        acorr=acorr,
        norm_acorr=norm_acorr,
        freqs=freqs,
        psd_db=psd_db,
        signal_variance=signal_variance,
        noise_variance=noise_variance,
        sample_count=sample_count,
        snr_db=snr_db,
    )


@demos_harmonic_detection_bp.route("/", methods=["GET"])
def page():
    defaults = {
        "snr": 10.0,
        "sample_count": 10_000,
        "view_mode": "autocorrelation",
    }
    return render_template("demos/harmonic_detection.html", defaults=defaults)


@demos_harmonic_detection_bp.route("/compute", methods=["POST"])
def compute():
    payload = request.get_json(force=True) or {}
    snr_db = float(payload.get("snr", 10.0))
    snr_db = round(snr_db * 10) / 10.0
    snr_db = max(-20.0, min(40.0, snr_db))

    sample_count = int(payload.get("sample_count", 10_000))

    data = _prepare_data(snr_db, sample_count)

    response = {
        "time_ms": data.time_ms.tolist(),
        "harmonic": data.harmonic.tolist(),
        "noise": data.noise.tolist(),
        "combined": data.combined.tolist(),
        "acorr": {
            "lags_ms": data.lags.tolist(),
            "clipped": data.acorr.tolist(),
            "normalized": data.norm_acorr.tolist(),
        },
        "psd": {
            "freqs": data.freqs.tolist(),
            "values_db": data.psd_db.tolist(),
        },
        "stats": {
            "signal_variance": data.signal_variance,
            "noise_variance": data.noise_variance,
            "snr_db": data.snr_db,
            "sample_count": data.sample_count,
        },
    }
    return jsonify(response)