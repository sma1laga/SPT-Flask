"""Spatial prediction demo for image compression."""
from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
from flask import Blueprint, current_app, render_template
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


def _image_path() -> str:
    static_root = current_app.static_folder
    return os.path.join(static_root, "demos", "images", IMAGE_NAME)


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