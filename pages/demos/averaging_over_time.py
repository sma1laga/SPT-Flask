"""Averaging over time or ensemble demo

Shows three sample realizations of a stochastic process and compares their
individual time statistics with the ensemble statistics under configurable
probabilities
"""
from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request
from scipy.io import wavfile
from scipy.signal import lfilter, resample


demos_averaging_over_time_bp = Blueprint(
    "demos_averaging_over_time", __name__, template_folder="../../templates"
)

AUDIO_SET = [
    ("strasse_kurz.wav", "Street speech"),
    ("neil_armstrong_mono32.wav", "Speech 2"),
    ("piano_mono32.wav", "Piano (mono)"),
]

DEFAULT_FS = 8000
DEFAULT_SAMPLES = 83_000  # ~10 seconds like the MATLAB demo
DEFAULT_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]


def _audio_root() -> str:
    return os.path.join(current_app.root_path, "static", "demos", "audio")


def _load_audio(name: str, fs: int = DEFAULT_FS, target_len: int = DEFAULT_SAMPLES) -> np.ndarray:
    """Load and lightly normalize an audio clip to a fixed lengt"""

    path = os.path.join(_audio_root(), name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    clip_fs, audio = wavfile.read(path)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Normalize amplitude
    peak = np.max(np.abs(audio))
    audio = audio / peak if peak > 0 else audio

    if clip_fs != fs:
        audio = resample(audio, int(len(audio) * (fs / float(clip_fs))))

    if len(audio) >= target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    return audio


def _validate_weights(raw: Sequence[float]) -> Tuple[np.ndarray, float]:
    weights = np.array([float(w) for w in raw], dtype=np.float64)
    total = float(np.sum(weights))
    if np.any(weights < 0):
        raise ValueError("Weights must be nonnegative and sum to 1.0")
    if not 0.99 <= total <= 1.01:
        raise ValueError("Weights must sum to 1.0 (within ±0.01)")
    return weights, total


def _generate_signals(signal_type: str) -> Tuple[np.ndarray, int, list[str]]:

    fs = DEFAULT_FS
    n = DEFAULT_SAMPLES
    t = np.arange(n) / fs
    rng = np.random.default_rng()

    if signal_type == "audio":
        signals = []
        labels = []
        for fname, label in AUDIO_SET:
            signals.append(_load_audio(fname, fs=fs, target_len=n))
            labels.append(label)
        return np.vstack(signals), fs, labels

    if signal_type == "ar-noise":
        signals = []
        labels = ["AR(1) noise #1", "AR(1) noise #2", "AR(1) noise #3"]
        for _ in range(3):
            w = rng.standard_normal(n)
            y = lfilter([1], [1, -0.9], w)
            signals.append(y)
        return np.vstack(signals), fs, labels

    if signal_type == "random-amplitude-sine":
        signals = []
        labels = ["Random amplitude sine #1", "Random amplitude sine #2", "Random amplitude sine #3"]
        freq = 100.0
        for _ in range(3):
            amp = rng.uniform(0.4, 1.25)
            signals.append(amp * np.sin(2 * np.pi * freq * t))
        return np.vstack(signals), fs, labels

    # random phase sine 
    signals = []
    labels = ["Random phase sine #1", "Random phase sine #2", "Random phase sine #3"]
    freq = 100.0
    for _ in range(3):
        phase = rng.uniform(-np.pi, np.pi)
        signals.append(np.sin(2 * np.pi * freq * t + phase))
    return np.vstack(signals), fs, labels


def _compute_statistics(signals: np.ndarray, weights: np.ndarray, signal_type: str) -> dict:
    means = signals.mean(axis=1)
    variances = signals.var(axis=1)
    ensemble_mean = np.average(signals, axis=0, weights=weights)
    ensemble_var = np.average((signals - ensemble_mean) ** 2, axis=0, weights=weights)

    mean_lim = float(np.max(np.abs(np.concatenate([ensemble_mean, [np.dot(weights, means)]]))))
    mean_lim = mean_lim if mean_lim > 0 else 1e-3
    var_lim = float(np.max(ensemble_var))
    var_lim = var_lim if var_lim > 0 else 1e-3

    # y-axis limits 
    if signal_type == "random-amplitude-sine":
        sig_max = float(np.max(np.abs(signals)))
        signal_lims = [[-sig_max, sig_max]] * 3
    else:
        signal_lims = [[-float(np.max(np.abs(sig))) or -1e-3, float(np.max(np.abs(sig))) or 1e-3] for sig in signals]

    return {
        "means": means.tolist(),
        "variances": variances.tolist(),
        "ensemble_mean": ensemble_mean,
        "ensemble_var": ensemble_var,
        "ensemble_mean_scalar": float(np.dot(weights, means)),
        "ensemble_var_scalar": float(np.dot(weights, variances)),
        "y_limits": {
            "signals": signal_lims,
            "mean": [-mean_lim, mean_lim],
            "var": [0.0, var_lim * 1.1 + 1e-3],
        },
    }


@demos_averaging_over_time_bp.route("/", methods=["GET"])
def page():
    defaults = {
        "signal_type": "audio",
        "weights": DEFAULT_WEIGHTS,
    }
    return render_template("demos/averaging_over_time.html", defaults=defaults)


@demos_averaging_over_time_bp.route("/compute", methods=["POST"])
def compute():
    payload = request.get_json(force=True) or {}
    signal_type = payload.get("signal_type", "audio")
    use_custom_weights = bool(payload.get("use_custom_weights", False))
    raw_weights = payload.get("weights", DEFAULT_WEIGHTS)

    if use_custom_weights:
        try:
            weights, weight_sum = _validate_weights(raw_weights)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    else:
        weights, weight_sum = np.array(DEFAULT_WEIGHTS, dtype=np.float64), 1.0

    try:
        signals, fs, labels = _generate_signals(signal_type)
    except FileNotFoundError as exc:
        return jsonify({"error": f"Audio file not found: {exc}"}), 400

    stats = _compute_statistics(signals, weights, signal_type)
    ensemble_mean = stats["ensemble_mean"]
    ensemble_var = stats["ensemble_var"]

    t = np.arange(signals.shape[1]) / fs
    response = {
        "t": t.tolist(),
        "signals": [sig.tolist() for sig in signals],
        "signal_labels": labels,
        "ensemble_mean": ensemble_mean.tolist(),
        "ensemble_var": ensemble_var.tolist(),
        "ensemble_mean_scalar": stats["ensemble_mean_scalar"],
        "ensemble_var_scalar": stats["ensemble_var_scalar"],
        "means": stats["means"],
        "variances": stats["variances"],
        "weights": weights.tolist(),
        "weight_sum": weight_sum,
        "y_limits": stats["y_limits"],
    }
    return jsonify(response)