"""Color space decomposition demo for the Lenna image"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, abort, current_app, render_template, url_for
from imageio.v2 import imread, imwrite
from skimage import color


@dataclass(frozen=True)
class ColorComponentView:
    name: str
    image_uri: str
    histogram_uri: str


@dataclass(frozen=True)
class ColorSpaceVariant:
    key: str
    title: str
    description: str
    components: List[ColorComponentView]


ROOT = Path(__file__).resolve().parent.parent.parent
LENNA_PATH = ROOT / "static" / "demos" / "images" / "lenna.png"


def _buffer_to_uri(buf: io.BytesIO) -> str:
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _array_to_uri(array: np.ndarray) -> str:
    buf = io.BytesIO()
    imwrite(buf, array.astype(np.uint8), format="png")
    return _buffer_to_uri(buf)


def _plot_histogram(freqs: Iterable[int]) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    values = np.arange(256)
    freqs_arr = np.array(list(freqs))
    ax.bar(values, freqs_arr, color="#4a90e2", edgecolor="none", alpha=0.86)
    ax.set_xlabel("Grey level")
    ax.set_ylabel("Frequency of occurrence")
    ax.set_xlim(-0.5, 255.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return _buffer_to_uri(buf)


def _normalize_to_uint8(component: np.ndarray, expected_range: Tuple[float, float] | None = None) -> np.ndarray:
    comp = np.asarray(component, dtype=np.float64)
    if expected_range is None:
        min_val = float(comp.min())
        max_val = float(comp.max())
    else:
        min_val, max_val = expected_range

    if max_val - min_val <= 1e-9:
        return np.zeros_like(comp, dtype=np.uint8)

    clipped = np.clip(comp, min_val, max_val)
    normalized = (clipped - min_val) / (max_val - min_val)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _component_view(name: str, component: np.ndarray, expected_range: Tuple[float, float] | None = None) -> ColorComponentView:
    scaled = _normalize_to_uint8(component, expected_range)
    freqs = np.bincount(scaled.ravel(), minlength=256)
    return ColorComponentView(
        name=name,
        image_uri=_array_to_uri(scaled),
        histogram_uri=_plot_histogram(freqs),
    )


def _prepare_variants() -> Dict[str, ColorSpaceVariant]:
    if not LENNA_PATH.exists():
        raise FileNotFoundError(f"Lenna image missing at {LENNA_PATH}")

    rgb_image = imread(LENNA_PATH)
    if rgb_image.ndim == 2:
        rgb_image = np.stack([rgb_image] * 3, axis=-1)
    if rgb_image.shape[-1] > 3:
        rgb_image = rgb_image[..., :3]
    if rgb_image.dtype != np.uint8:
        rgb_image = (255 * np.clip(rgb_image, 0, 1)).astype(np.uint8)

    rgb_float = rgb_image.astype(np.float64) / 255.0

    rgb_variant = ColorSpaceVariant(
        key="rgb",
        title="Decomposition into RGB components",
        description="Classic split into the red, green, and blue channels",
        components=[
            _component_view("R component", rgb_image[..., 0], (0.0, 255.0)),
            _component_view("G component", rgb_image[..., 1], (0.0, 255.0)),
            _component_view("B component", rgb_image[..., 2], (0.0, 255.0)),
        ],
    )

    yuv = color.rgb2yuv(rgb_float)
    yuv_variant = ColorSpaceVariant(
        key="yuv",
        title="Decomposition into YUV components",
        description="Luminance plus blue- and red-projected chrominance componentss",
        components=[
            _component_view("Y component", yuv[..., 0], (0.0, 1.0)),
            _component_view("U component", yuv[..., 1], (-0.6, 0.6)),
            _component_view("V component", yuv[..., 2], (-0.6, 0.6)),
        ],
    )

    yiq = color.rgb2yiq(rgb_float)
    yiq_variant = ColorSpaceVariant(
        key="yiq",
        title="Decomposition into YIQ components",
        description="Luminance together with in-phase (I) and quadrature (Q) chrominance axes.",
        components=[
            _component_view("Y component", yiq[..., 0], (0.0, 1.0)),
            _component_view("I component", yiq[..., 1], (-0.65, 0.65)),
            _component_view("Q component", yiq[..., 2], (-0.65, 0.65)),
        ],
    )

    ycbcr = color.rgb2ycbcr(rgb_float)
    ycbcr_variant = ColorSpaceVariant(
        key="ycbcr",
        title="Decomposition into YCbCr components",
        description="Broadcast-friendly luminance plus blue- and red-difference chroma channels.",
        components=[
            _component_view("Y component", ycbcr[..., 0], (0.0, 255.0)),
            _component_view("Cb component", ycbcr[..., 1], (0.0, 255.0)),
            _component_view("Cr component", ycbcr[..., 2], (0.0, 255.0)),
        ],
    )

    return {variant.key: variant for variant in [rgb_variant, yuv_variant, yiq_variant, ycbcr_variant]}


_VARIANTS: Dict[str, ColorSpaceVariant] | None = None


def _variants_cached() -> Dict[str, ColorSpaceVariant]:
    global _VARIANTS
    if _VARIANTS is None:
        _VARIANTS = _prepare_variants()
    return _VARIANTS


def _serialize_variant(variant: ColorSpaceVariant) -> Dict[str, object]:
    return {
        "key": variant.key,
        "title": variant.title,
        "description": variant.description,
        "components": [
            {
                "name": component.name,
                "image_uri": component.image_uri,
                "histogram_uri": component.histogram_uri,
            }
            for component in variant.components
        ],
    }


demos_color_spaces_bp = Blueprint("demos_color_spaces", __name__, template_folder="../../templates")


@demos_color_spaces_bp.route("/", methods=["GET"], endpoint="page")
def page() -> str:
    try:
        variants = {key: _serialize_variant(variant) for key, variant in _variants_cached().items()}
    except FileNotFoundError as exc:
        current_app.logger.exception("Color spaces demo assets missing")
        abort(500, description=str(exc))
    except Exception:
        current_app.logger.exception("Failed to prepare color space variants")
        abort(500, description="Failed to prepare color space variants")

    return render_template(
        "demos/color_spaces.html",
        variants=variants,
        original_src=url_for("static", filename="demos/images/lenna.png"),
    )
