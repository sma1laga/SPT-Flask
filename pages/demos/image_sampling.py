"""Image sampling demo comparing rect and quincunx patterns"""

from __future__ import annotations

import base64
import io
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, render_template
from PIL import Image

from utils.img import fig_to_base64


IMAGE_SIZE = 512
CIRCLE_FREQUENCY = 60


def _generate_circles() -> np.ndarray:
    """Generate the sinusoid rings"""
    coords = np.arange(1, IMAGE_SIZE + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="ij")
    r = np.sqrt((grid_x - 256.5) ** 2 + (grid_y - 256.5) ** 2)
    return 0.5 + 0.5 * np.sin(CIRCLE_FREQUENCY * 2 * np.pi / 256.0 * r)


def _encode_image(arr: np.ndarray) -> str:
    clipped = np.clip(arr, 0.0, 1.0)
    image = Image.fromarray((clipped * 255.0).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _rectangular_sampling(img: np.ndarray) -> np.ndarray:
    averaged = (
        img[0::2, 0::2]
        + img[0::2, 1::2]
        + img[1::2, 0::2]
        + img[1::2, 1::2]
    ) / 4.0
    return np.repeat(np.repeat(averaged, 2, axis=0), 2, axis=1)


def _qi(x: int) -> int:
    return x - 1


def _quincunx_sampling(img: np.ndarray) -> np.ndarray:
    sub = np.zeros_like(img)
    height, width = img.shape
    limit = min(height, width) // 4

    for x in range(2, limit):
        for y in range(2, limit):
            pixel_value = (
                0.125
                * (
                    img[_qi(4 * x - 3), _qi(4 * y - 3)]
                    + img[_qi(4 * x - 3), _qi(4 * y - 2)]
                    + img[_qi(4 * x - 2), _qi(4 * y - 3)]
                    + img[_qi(4 * x - 2), _qi(4 * y - 2)]
                )
                + 0.0625
                * (
                    img[_qi(4 * x - 4), _qi(4 * y - 3)]
                    + img[_qi(4 * x - 4), _qi(4 * y - 2)]
                    + img[_qi(4 * x - 3), _qi(4 * y - 4)]
                    + img[_qi(4 * x - 2), _qi(4 * y - 4)]
                    + img[_qi(4 * x - 1), _qi(4 * y - 3)]
                    + img[_qi(4 * x - 1), _qi(4 * y - 2)]
                    + img[_qi(4 * x - 3), _qi(4 * y - 1)]
                    + img[_qi(4 * x - 2), _qi(4 * y - 1)]
                )
            )

            sub[_qi(4 * x - 3), _qi(4 * y - 3)] = pixel_value
            sub[_qi(4 * x - 2), _qi(4 * y - 3)] = pixel_value
            sub[_qi(4 * x - 3), _qi(4 * y - 2)] = pixel_value
            sub[_qi(4 * x - 2), _qi(4 * y - 2)] = pixel_value

            pixel_value = (
                0.125
                * (
                    img[_qi(4 * x - 3), _qi(4 * y - 3)]
                    + img[_qi(4 * x - 3), _qi(4 * y - 2)]
                    + img[_qi(4 * x - 2), _qi(4 * y - 3)]
                    + img[_qi(4 * x - 2), _qi(4 * y - 2)]
                )
                + 0.0625
                * (
                    img[_qi(4 * x - 2), _qi(4 * y - 1)]
                    + img[_qi(4 * x - 2), _qi(4 * y)]
                    + img[_qi(4 * x - 1), _qi(4 * y - 2)]
                    + img[_qi(4 * x), _qi(4 * y - 2)]
                    + img[_qi(4 * x + 1), _qi(4 * y - 1)]
                    + img[_qi(4 * x + 1), _qi(4 * y)]
                    + img[_qi(4 * x - 1), _qi(4 * y + 1)]
                    + img[_qi(4 * x), _qi(4 * y + 1)]
                )
            )

            sub[_qi(4 * x - 1), _qi(4 * y - 1)] = pixel_value
            sub[_qi(4 * x), _qi(4 * y - 1)] = pixel_value
            sub[_qi(4 * x - 1), _qi(4 * y)] = pixel_value
            sub[_qi(4 * x), _qi(4 * y)] = pixel_value

    for x in range(2, limit):
        for y in range(2, limit):
            sub[_qi(4 * x - 1), _qi(4 * y - 3)] = 0.5 * (
                sub[_qi(4 * x - 2), _qi(4 * y - 3)] + sub[_qi(4 * x - 1), _qi(4 * y - 4)]
            )
            sub[_qi(4 * x), _qi(4 * y - 3)] = 0.5 * (
                sub[_qi(4 * x + 1), _qi(4 * y - 3)] + sub[_qi(4 * x), _qi(4 * y - 4)]
            )
            sub[_qi(4 * x - 1), _qi(4 * y - 2)] = 0.5 * (
                sub[_qi(4 * x - 2), _qi(4 * y - 2)] + sub[_qi(4 * x - 1), _qi(4 * y - 1)]
            )
            sub[_qi(4 * x), _qi(4 * y - 2)] = 0.5 * (
                sub[_qi(4 * x + 1), _qi(4 * y - 2)] + sub[_qi(4 * x), _qi(4 * y - 1)]
            )
            sub[_qi(4 * x - 3), _qi(4 * y - 1)] = 0.5 * (
                sub[_qi(4 * x - 4), _qi(4 * y - 1)] + sub[_qi(4 * x - 3), _qi(4 * y - 2)]
            )
            sub[_qi(4 * x - 2), _qi(4 * y - 1)] = 0.5 * (
                sub[_qi(4 * x - 1), _qi(4 * y - 1)] + sub[_qi(4 * x - 2), _qi(4 * y - 2)]
            )
            sub[_qi(4 * x - 3), _qi(4 * y)] = 0.5 * (
                sub[_qi(4 * x - 4), _qi(4 * y)] + sub[_qi(4 * x - 3), _qi(4 * y + 1)]
            )
            sub[_qi(4 * x - 2), _qi(4 * y)] = 0.5 * (
                sub[_qi(4 * x - 1), _qi(4 * y)] + sub[_qi(4 * x - 2), _qi(4 * y + 1)]
            )

    return sub


def _draw_rect_grid() -> str:
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for k in range(1, 8):
        ax.plot([0, 14], [2 * k - 1, 2 * k - 1], color="black", linewidth=1)
        ax.plot([2 * k - 1, 2 * k - 1], [0, 14], color="black", linewidth=1)
    for l in range(0, 3):
        for k in range(0, 3):
            ax.plot(4 * k + 3, 4 * l + 3, "r.", markersize=24)
    ax.plot([5, 9], [9, 9], "k--", linewidth=3.4)
    ax.plot([9, 9], [9, 5], "k--", linewidth=3.4)
    ax.plot([9, 5], [5, 5], "k--", linewidth=3.4)
    ax.plot([5, 5], [5, 9], "k--", linewidth=3.4)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig_to_base64(fig)


def _draw_quinc_grid() -> str:
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for k in range(1, 8):
        ax.plot([0, 14], [2 * k - 1, 2 * k - 1], color="black", linewidth=1)
        ax.plot([2 * k - 1, 2 * k - 1], [0, 14], color="black", linewidth=1)
    ax.plot(3, 3, "r.", markersize=24)
    ax.plot(3, 11, "r.", markersize=24)
    ax.plot(11, 3, "r.", markersize=24)
    ax.plot(11, 11, "r.", markersize=24)
    ax.plot(7, 7, "r.", markersize=24)
    ax.plot([3, 7], [7, 11], "k--", linewidth=3.4)
    ax.plot([7, 11], [11, 7], "k--", linewidth=3.4)
    ax.plot([11, 7], [7, 3], "k--", linewidth=3.4)
    ax.plot([7, 3], [3, 7], "k--", linewidth=3.4)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig_to_base64(fig)


def _compute_views() -> Dict[str, str]:
    original = _generate_circles()
    return {
        "original": _encode_image(original),
        "rectangular": _encode_image(_rectangular_sampling(original)),
        "quincunx": _encode_image(_quincunx_sampling(original)),
        "rect_grid": _draw_rect_grid(),
        "quinc_grid": _draw_quinc_grid(),
    }


demos_image_sampling_bp = Blueprint(
    "image_sampling", __name__, template_folder="../../templates"
)


@demos_image_sampling_bp.route("/", methods=["GET"])
def page():
    images = _compute_views()
    return render_template(
        "demos/image_sampling.html",
        images=images,
    )