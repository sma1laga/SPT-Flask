"""Spatial prediction demo for image compression."""
from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Mapping, Tuple

import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request
from PIL import Image


@dataclass
class PredictionConfig:
    key: str
    label: str
    description: str
    weights: Mapping[Tuple[int, int], float]


@dataclass
class PredictionView:
    label: str
    description: str
    weights_grid: list[list[str]]
    entropy: float
    compression_factor: float
    error_src: str


IMAGE_NAME = "lenna.png"


PREDICTIONS = [
    PredictionConfig(
        key="none",
        label="No prediction",
        description="Reference with zero-valued predictor (shows raw pixel values).",
        weights={},
    ),
    PredictionConfig(
        key="west",
        label="Left pixel",
        description="Predict current sample using the pixel directly to the left.",
        weights={(0, -1): 1.0},
    ),
    PredictionConfig(
        key="nw_average",
        label="0.5·W + 0.25·(NW+N)",
        description="Weighted mix of west, north-west, and north pixels.",
        weights={
            (0, -1): 0.5,
            (-1, -1): 0.25,
            (-1, 0): 0.25,
        },
    ),
    PredictionConfig(
        key="smooth",
        label="Smoothed upper row + west",
        description="Blend of west, north-west, north, and north-east neighbours.",
        weights={
            (0, -1): 0.35,
            (-1, -1): 0.15,
            (-1, 0): 0.35,
            (-1, 1): 0.15,
        },
    ),
]

PREDICTION_BY_KEY = {config.key: config for config in PREDICTIONS}


def _image_path() -> str:
    static_root = current_app.static_folder
    return os.path.join(static_root, "demos", "images", IMAGE_NAME)

@lru_cache(maxsize=1)
def _load_grayscale() -> np.ndarray:
    path = _image_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float64)
    return arr


def _entropy_bits(values: np.ndarray) -> float:
    flat = values.ravel()
    uniques, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    return float(-np.sum(probs * np.log2(probs)))


def _encode_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _predict_image(img: np.ndarray, weights: Mapping[Tuple[int, int], float]) -> np.ndarray:
    height, width = img.shape
    pred = np.zeros_like(img, dtype=np.float64)

    for y in range(height):
        for x in range(width):
            acc = 0.0
            for (dy, dx), weight in weights.items():
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    acc += weight * img[ny, nx]
            pred[y, x] = acc
    return pred


def _error_to_image(error: np.ndarray) -> str:
    min_val = float(np.min(error))
    max_val = float(np.max(error))
    if math.isclose(max_val, min_val):
        normalized = np.full_like(error, 0.5, dtype=np.float64)
    else:
        normalized = (error - min_val) / (max_val - min_val)
    visual = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return _encode_png(visual)


def _weights_grid(weights: Mapping[Tuple[int, int], float]) -> list[list[str]]:
    grid: list[list[str]] = [["0" for _ in range(3)] for _ in range(3)]
    for (dy, dx), weight in weights.items():
        row, col = 1 + dy, 1 + dx
        if 0 <= row < 3 and 0 <= col < 3:
            grid[row][col] = f"{weight:g}"
    grid[1][1] = "S"
    return grid


def _build_prediction_views(img: np.ndarray, base_entropy: float) -> Dict[str, PredictionView]:
    results: Dict[str, PredictionView] = {}
    for config in PREDICTIONS:
        prediction = _predict_image(img, config.weights)
        error = np.rint(img - prediction)
        entropy = _entropy_bits(error)
        compression_factor = base_entropy / entropy if entropy else math.inf
        view = PredictionView(
            label=config.label,
            description=config.description,
            weights_grid=_weights_grid(config.weights),
            entropy=entropy,
            compression_factor=compression_factor,
            error_src=_error_to_image(error),
        )
        results[config.key] = view
    return results

def _quantize_error(error: np.ndarray, bits: int, step: float) -> np.ndarray:
    levels = 2 ** bits
    half_levels = levels / 2
    scaled = np.rint(error / step)
    clipped = np.clip(scaled, -half_levels, half_levels - 1)
    return clipped * step


def _compute_quantization(
    img: np.ndarray, *, config: PredictionConfig, bits: int, step: float, base_entropy: float
) -> Dict[str, float | str]:
    prediction = _predict_image(img, config.weights)
    prediction_error = img - prediction
    quantized_error = _quantize_error(prediction_error, bits, step)
    reconstructed = np.clip(prediction + quantized_error, 0, 255)
    quantization_error = reconstructed - img

    entropy = _entropy_bits(np.rint(quantized_error))
    compression_factor = base_entropy / entropy if entropy else math.inf

    mse = float(np.mean(np.square(img - reconstructed)))
    signal_power = float(np.mean(np.square(img)))
    snr = math.inf if mse == 0 else 10.0 * math.log10(signal_power / mse)
    psnr = math.inf if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))

    return {
        "prediction_error_src": _error_to_image(prediction_error),
        "reconstructed_src": _encode_png(reconstructed),
        "quantization_error_src": _error_to_image(quantization_error),
        "entropy": entropy,
        "compression_factor": compression_factor,
        "snr": snr,
        "psnr": psnr,
    }



demos_spatial_prediction_bp = Blueprint(
    "demos_spatial_prediction", __name__, template_folder="../../templates"
)


@demos_spatial_prediction_bp.route("/", methods=["GET"], endpoint="page")
def index():
    img = _load_grayscale()
    base_entropy = _entropy_bits(np.rint(img))
    predictions = _build_prediction_views(img, base_entropy)

    return render_template(
        "demos/spatial_prediction.html",
        original_src=_encode_png(img),
        predictions={key: vars(value) for key, value in predictions.items()},
        base_entropy=base_entropy,
    )


@demos_spatial_prediction_bp.route("/compute", methods=["POST"], endpoint="compute")
def compute():
    payload = request.get_json(silent=True) or {}
    key = payload.get("key", "none")
    bits = int(payload.get("bits", 4))
    step = float(payload.get("step", 4.0))

    if bits < 1:
        return jsonify({"error": "Quantization level must be at least 1 bit."}), 400
    if step <= 0:
        return jsonify({"error": "Quantization step size must be positive."}), 400

    config = PREDICTION_BY_KEY.get(key)
    if not config:
        return jsonify({"error": f"Unknown predictor '{key}'."}), 400

    img = _load_grayscale()
    base_entropy = _entropy_bits(np.rint(img))

    result = _compute_quantization(img, config=config, bits=bits, step=step, base_entropy=base_entropy)
    return jsonify(result)