"""Image compression demo page."""
from __future__ import annotations

import base64
import io
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from flask import Blueprint, current_app, render_template
from PIL import Image



@dataclass
class CompressionResult:
    label: str
    description: str
    image_src: str
    file_size: int
    compression_factor: float
    snr: float
    psnr: float
    width: int
    height: int


JPEG_WEAK_QUALITY = 75
JPEG_MEDIUM_QUALITY = 55
JPEG_STRONG_QUALITY = 35


def _load_image(image_path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        size = rgb.size
        arr = np.asarray(rgb, dtype=np.float64) / 255.0
    return arr, size


def _encode_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _encode_jpeg(arr: np.ndarray, quality: int) -> Tuple[bytes, np.ndarray]:
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(
        buf, format="JPEG", quality=quality, optimize=True
    )
    data = buf.getvalue()
    with Image.open(io.BytesIO(data)) as decoded:
        decoded_arr = np.asarray(decoded.convert("RGB"), dtype=np.float64) / 255.0
    return data, decoded_arr


def _zip_image(path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(path, arcname=os.path.basename(path))
    return buf.getvalue()


def _snr(signal: np.ndarray, reconstruction: np.ndarray) -> float:
    signal_power = float(np.mean(np.square(signal)))
    noise_power = float(np.mean(np.square(reconstruction - signal)))
    if noise_power <= 0:
        return math.inf
    return 10.0 * math.log10(signal_power / noise_power)


def _psnr(signal: np.ndarray, reconstruction: np.ndarray) -> float:
    mse = float(np.mean(np.square(reconstruction - signal)))
    if mse <= 0:
        return math.inf
    peak = 1.0
    return 10.0 * math.log10((peak * peak) / mse)


def _format_value(value: float) -> float:
    return float(value) if math.isfinite(value) else math.inf

def _cameraman_png(static_folder: str | Path) -> Path:
    static_root = Path(static_folder)
    png_path = static_root / "images" / "web" / "cameraman.png"
    if png_path.exists():
        return png_path

    tiff_path = static_root / "images" / "cameraman.tif"
    if not tiff_path.exists():
        raise FileNotFoundError("cameraman.tif not found; cannot build cameraman.png")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(tiff_path) as img:
        img.convert("RGB").save(png_path, format="PNG")
    return png_path


def _build_results() -> Dict[str, CompressionResult]:
    image_path = _cameraman_png(current_app.static_folder)
    img, _ = _load_image(image_path)
    height, width = img.shape[0], img.shape[1]
    original_size = os.path.getsize(image_path)


    # Losless ZIP
    zip_bytes = _zip_image(image_path)
    zip_size = len(zip_bytes)

    # JPEG variants
    weak_bytes, weak_img = _encode_jpeg(img, JPEG_WEAK_QUALITY)
    medium_bytes, medium_img = _encode_jpeg(img, JPEG_MEDIUM_QUALITY)
    strong_bytes, strong_img = _encode_jpeg(img, JPEG_STRONG_QUALITY)

    results = {
        "original": CompressionResult(
            label="Original",
            description="Uncompressed reference image.",
            image_src=_encode_png(img),
            file_size=original_size,
            compression_factor=1.0,
            snr=math.inf,
            psnr=math.inf,
            width=width,
            height=height,
        ),
        "zip": CompressionResult(
            label="Lossless ZIP",
            description="ZIP archiving without visible changes to the image.",
            image_src=_encode_png(img),
            file_size=zip_size,
            compression_factor=original_size / zip_size if zip_size else math.inf,
            snr=_format_value(_snr(img, img)),
            psnr=_format_value(_psnr(img, img)),
            width=width,
            height=height,
        ),
        "jpeg_weak": CompressionResult(
            label="Weak JPEG",
            description=f"JPEG with quality={JPEG_WEAK_QUALITY} (subtle compression).",
            image_src=f"data:image/jpeg;base64,{base64.b64encode(weak_bytes).decode('ascii')}",
            file_size=len(weak_bytes),
            compression_factor=original_size / len(weak_bytes) if weak_bytes else math.inf,
            snr=_format_value(_snr(img, weak_img)),
            psnr=_format_value(_psnr(img, weak_img)),
            width=width,
            height=height,
        ),
        "jpeg_medium": CompressionResult(
            label="Medium JPEG",
            description=f"JPEG with quality={JPEG_MEDIUM_QUALITY} (balanced compression).",
            image_src=f"data:image/jpeg;base64,{base64.b64encode(medium_bytes).decode('ascii')}",
            file_size=len(medium_bytes),
            compression_factor=original_size / len(medium_bytes) if medium_bytes else math.inf,
            snr=_format_value(_snr(img, medium_img)),
            psnr=_format_value(_psnr(img, medium_img)),
            width=width,
            height=height,
        ),
        "jpeg_strong": CompressionResult(
            label="Strong JPEG",
            description=f"JPEG with quality={JPEG_STRONG_QUALITY} (visible artifacts).",
            image_src=f"data:image/jpeg;base64,{base64.b64encode(strong_bytes).decode('ascii')}",
            file_size=len(strong_bytes),
            compression_factor=original_size / len(strong_bytes) if strong_bytes else math.inf,
            snr=_format_value(_snr(img, strong_img)),
            psnr=_format_value(_psnr(img, strong_img)),
            width=width,
            height=height,
        ),
    }
    return results


demos_compression_bp = Blueprint("demos_compression", __name__, template_folder="../../templates")


@demos_compression_bp.route("/", methods=["GET"], endpoint="page")
def index():
    results = _build_results()
    variant_payload = {key: vars(value) for key, value in results.items()}
    return render_template(
        "demos/compression.html",
        variants=variant_payload,
    )