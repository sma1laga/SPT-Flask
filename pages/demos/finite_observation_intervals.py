from __future__ import annotations

import numpy as np
from flask import Blueprint, jsonify, render_template, request
from functools import lru_cache
from scipy import signal
from scipy.io import wavfile


demos_finite_observation_intervals_bp = Blueprint(
    "demos_finite_observation_intervals", __name__, template_folder="../../templates"
)

FS = 8000
NO_SAMPLES = 16000
DISP_LEN = 570
MAX_LAG = 200
DEFAULT_FREQ = 200.0
DEFAULT_SAMPLES = 8000
PRECISION = 1


@lru_cache(maxsize=1)
def _load_speech_signal() -> np.ndarray:
    sr, data = wavfile.read("static/audio/examp1l.wav")
    if data.ndim > 1:
        data = data.mean(axis=1)

    peak = float(np.max(np.abs(data)))
    normalized = data.astype(np.float32) if peak == 0 else data.astype(np.float32) / peak

    if sr != FS:
        raise ValueError(
            f"examp1l.wav must be {FS} Hz; got {sr} Hz. Please regenerate the asset."
        )

    start, end = 24772, 124771 + 1
    if len(normalized) < end:
        raise ValueError("examp1l.wav is too short; expected at least 124771 samples at 8 kHz.")

    return normalized[start:end]


def _design_lowpass_coeffs() -> np.ndarray:
    return signal.firwin(numtaps=101, cutoff=2000, window="hamming", fs=FS)


def _ensure_length(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) >= length:
        return x[:length]
    reps = int(np.ceil(length / len(x)))
    return np.tile(x, reps)[:length]


_COEFFS = _design_lowpass_coeffs().astype(np.float32)
_RNG = np.random.default_rng(0)
_WHITE_NOISE = _RNG.standard_normal(NO_SAMPLES + len(_COEFFS)).astype(np.float32)
_COLORED_NOISE = signal.lfilter(_COEFFS, [1.0], _WHITE_NOISE).astype(np.float32)
_WHITE_NOISE = _WHITE_NOISE[:NO_SAMPLES]


def _build_signal(signal_kind: str, freq_hz: float) -> np.ndarray:
    k = np.arange(NO_SAMPLES, dtype=np.float32)

    if signal_kind == "sinusoid":
        return np.sin(2 * np.pi * (freq_hz / FS) * k)
    if signal_kind == "white_noise":
        return _WHITE_NOISE.copy()
    if signal_kind == "colored_noise":
        return _COLORED_NOISE[:NO_SAMPLES].copy()
    if signal_kind == "speech":
        return _ensure_length(_load_speech_signal(), NO_SAMPLES)

    return _WHITE_NOISE.copy()


def _biased_autocorr(values: np.ndarray) -> np.ndarray:
    acorr_full = np.correlate(values, values, mode="full") / len(values)
    center = len(acorr_full) // 2
    start = center - MAX_LAG
    end = center + MAX_LAG + 1
    return acorr_full[start:end]


def _real_mode(signal_values: np.ndarray, sample_count: int):
    segment = _ensure_length(signal_values, sample_count)

    time_ms = (np.arange(min(DISP_LEN, len(segment))) / FS) * 1000.0
    signal_display = segment[:DISP_LEN]

    acorr = _biased_autocorr(segment)
    acorr_norm = acorr / np.max(np.abs(acorr)) if np.max(np.abs(acorr)) > 0 else acorr

    fft_values = np.fft.rfft(segment, n=len(segment))
    psd = (fft_values * np.conj(fft_values)).real / len(segment)
    psd_db = 10 * np.log10(psd + 1e-12)
    freqs = np.fft.rfftfreq(len(segment), 1 / FS)

    return {
        "time_ms": time_ms,
        "signal": signal_display,
        "acorr_lags": np.arange(-MAX_LAG, MAX_LAG + 1),
        "acorr": acorr_norm,
        "psd_freqs": freqs,
        "psd_db": psd_db,
    }


def _ideal_mode(signal_kind: str, freq_hz: float):
    display_signal = _build_signal(signal_kind, freq_hz)
    display_signal = _ensure_length(display_signal, DISP_LEN)

    if signal_kind == "sinusoid":
        acorr = _ensure_length(_build_signal(signal_kind, freq_hz), 2 * MAX_LAG + 1)
        psd = np.ones(FS // 2, dtype=np.float64) * 0.00001
        freq_idx = int(round(freq_hz))
        freq_idx = max(0, min(freq_idx, len(psd) - 1))
        psd[freq_idx] = 10.0
        freqs = np.arange(len(psd))
    elif signal_kind == "white_noise":
        acorr = np.zeros(2 * MAX_LAG + 1, dtype=np.float64)
        acorr[MAX_LAG] = 1.0
        psd = np.ones(FS // 2, dtype=np.float64)
        freqs = np.arange(len(psd))
    elif signal_kind == "colored_noise":
        coeff_acorr = _biased_autocorr(_COEFFS)
        acorr = coeff_acorr / np.max(np.abs(coeff_acorr))
        coeff_fft = np.fft.rfft(_COEFFS, n=NO_SAMPLES)
        psd = (coeff_fft * np.conj(coeff_fft)).real / len(coeff_fft)
        freqs = np.fft.rfftfreq(len(coeff_fft), 1 / FS)
    else:
        return _real_mode(_build_signal(signal_kind, freq_hz), DEFAULT_SAMPLES)

    psd_db = 10 * np.log10(psd + 1e-12)

    return {
        "time_ms": (np.arange(len(display_signal)) / FS) * 1000.0,
        "signal": display_signal,
        "acorr_lags": np.arange(-MAX_LAG, MAX_LAG + 1),
        "acorr": acorr,
        "psd_freqs": freqs,
        "psd_db": psd_db,
    }


@demos_finite_observation_intervals_bp.route("/", methods=["GET"])
def page():
    defaults = {
        "frequency": DEFAULT_FREQ,
        "sample_count": DEFAULT_SAMPLES,
        "signal_kind": "sinusoid",
        "mode": "real",
    }
    return render_template("demos/finite_observation_intervals.html", defaults=defaults)


@demos_finite_observation_intervals_bp.route("/compute", methods=["POST"])
def compute():
    payload = request.get_json(force=True) or {}

    freq_hz = float(payload.get("frequency", DEFAULT_FREQ))
    freq_hz = max(10.0, min(3500.0, round(freq_hz, PRECISION)))

    sample_count = int(payload.get("sample_count", DEFAULT_SAMPLES))
    sample_count = max(200, min(NO_SAMPLES, sample_count))

    signal_kind = str(payload.get("signal_kind", "sinusoid"))
    mode = str(payload.get("mode", "real"))

    if mode == "ideal":
        data = _ideal_mode(signal_kind, freq_hz)
        mode_used = "ideal" if signal_kind != "speech" else "real"
    else:
        data = _real_mode(_build_signal(signal_kind, freq_hz), sample_count)
        mode_used = "real"

    return jsonify(
        {
            "time_ms": data["time_ms"].tolist(),
            "signal": data["signal"].tolist(),
            "acorr": {
                "lags": data["acorr_lags"].tolist(),
                "values": data["acorr"].tolist(),
            },
            "psd": {
                "freqs": data["psd_freqs"].tolist(),
                "values_db": data["psd_db"].tolist(),
            },
            "meta": {
                "frequency": freq_hz,
                "sample_count": sample_count,
                "signal_kind": signal_kind,
                "mode": mode_used,
            },
        }
    )