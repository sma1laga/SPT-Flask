"""Zonal DCT image coding demo page"""
from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from flask import Blueprint, current_app, render_template
from PIL import Image
from scipy.fft import dctn, idctn

from .demo_images import cached_demo_image

@dataclass
class ZonalPreset:
    key: str
    label: str
    description: str
    matrix: List[List[float]]


@dataclass
class ZonalResult:
    label: str
    description: str
    matrix: List[List[float]]
    dct_src: str
    reconstructed_src: str
    entropy: float
    snr: float
    psnr: float


BLOCK_SIZE = 8

PRESETS = [
    ZonalPreset(
        key="balanced",
        label="Balanced (JPEG style)",
        description="Gentle roll-off toward higher frequencies; keeps most detail with mild compression.",
        matrix=[
            [8, 7, 6, 4, 3, 2, 1, 0],
            [7, 6, 4, 3, 2, 1, 0, 0],
            [6, 4, 3, 2, 1, 0, 0, 0],
            [4, 3, 2, 1, 0, 0, 0, 0],
            [3, 2, 1, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ),
    ZonalPreset(
        key="texture_friendly",
        label="Texture-friendly",
        description="Retains more diagonal and mid-frequency energy to preserve textured regions while tapering extremes.",
        matrix=[
            [8, 7, 6, 5, 4, 2, 1, 0],
            [7, 6, 5, 4, 2, 1, 0, 0],
            [6, 5, 4, 3, 2, 1, 0, 0],
            [5, 4, 3, 2, 1, 0, 0, 0],
            [4, 2, 2, 1, 0, 0, 0, 0],
            [2, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ),
    ZonalPreset(
        key="soft_lowpass",
        label="Soft low-pass",
        description="Emphasizes DC/low-frequency content; suppresses diagonal details for higher compression",
        matrix=[
            [8, 6, 4, 2, 0, 0, 0, 0],
            [6, 4, 2, 0, 0, 0, 0, 0],
            [4, 2, 1, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ),
    ZonalPreset(
        key="directional_lowpass",
        label="Directional low-pass",
        description="Preserves DC plus horizontal/vertical harmonics while discarding diagonals to highlight blocking",
        matrix=[
            [8, 7, 6, 5, 3, 0, 0, 0],
            [7, 6, 5, 3, 0, 0, 0, 0],
            [6, 5, 4, 2, 0, 0, 0, 0],
            [5, 3, 2, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ),
    ZonalPreset(
        key="strong_zonal",
        label="Strong zonal mask",
        description="Keeps only the very low-frequency core; highlights blocking but maximizes coefff culling",
        matrix=[
            [8, 6, 4, 2, 0, 0, 0, 0],
            [6, 4, 2, 0, 0, 0, 0, 0],
            [4, 2, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ),
]

PRESET_BY_KEY = {preset.key: preset for preset in PRESETS}


def _image_path() -> str:
    _, path = cached_demo_image(current_app.static_folder)
    return str(path)


def _load_grayscale() -> np.ndarray:
    path = _image_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float64)
    return arr


def _encode_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _entropy_bits(values: np.ndarray) -> float:
    flat = np.rint(values).ravel()
    uniques, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    return float(-np.sum(probs * np.log2(probs)))


def _format_value(value: float) -> float:
    return float(value) if math.isfinite(value) else math.inf


def _block_dct_quantize(img: np.ndarray, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = img.shape
    coeffs = np.zeros_like(img, dtype=np.float64)
    reconstruction = np.zeros_like(img, dtype=np.float64)

    for y in range(0, height, BLOCK_SIZE):
        for x in range(0, width, BLOCK_SIZE):
            block = img[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE]
            dct_block = dctn(block, norm="ortho")
            quantized = np.where(
                matrix > 0, np.round(dct_block / matrix) * matrix, 0.0
            )
            coeffs[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE] = quantized
            reconstruction[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE] = idctn(
                quantized, norm="ortho"
            )

    return coeffs, reconstruction


def _dct_log_image(coeffs: np.ndarray) -> np.ndarray:
    magnitude = np.log10(1.0 + np.abs(coeffs))
    min_val = magnitude.min()
    max_val = magnitude.max()
    if math.isclose(max_val, min_val):
        normalized = np.zeros_like(magnitude)
    else:
        normalized = (magnitude - min_val) / (max_val - min_val)
    return np.clip(normalized * 255.0, 0, 255)


def _metrics(original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, float]:
    mse = float(np.mean(np.square(original - reconstructed)))
    signal_power = float(np.mean(np.square(original)))
    snr = math.inf if mse == 0 else 10.0 * math.log10(signal_power / mse)
    psnr = math.inf if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return snr, psnr


def _build_results() -> Dict[str, ZonalResult]:
    img = _load_grayscale()
    results: Dict[str, ZonalResult] = {}

    for preset in PRESETS:
        matrix = np.asarray(preset.matrix, dtype=np.float64)
        coeffs, reconstruction = _block_dct_quantize(img, matrix)

        log_image = _dct_log_image(coeffs)
        snr, psnr = _metrics(img, reconstruction)
        entropy = _entropy_bits(coeffs)

        results[preset.key] = ZonalResult(
            label=preset.label,
            description=preset.description,
            matrix=preset.matrix,
            dct_src=_encode_png(log_image),
            reconstructed_src=_encode_png(reconstruction),
            entropy=_format_value(entropy / (BLOCK_SIZE * BLOCK_SIZE)),
            snr=_format_value(snr),
            psnr=_format_value(psnr),
        )

    return results


zonal_dct_bp = Blueprint("zonal_dct", __name__, template_folder="../../templates")


@zonal_dct_bp.route("/", methods=["GET"], endpoint="page")
def index():
    results = _build_results()
    payload = {key: vars(value) for key, value in results.items()}
    return render_template(
        "demos/zonal_dct.html",
        presets=[vars(preset) for preset in PRESETS],
        results=payload,
        block_size=BLOCK_SIZE,
    )