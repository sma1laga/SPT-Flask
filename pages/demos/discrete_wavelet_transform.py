"""Discrete Wavelet Transform (DWT) demo for the selected reference image

This blueprint prepares four visualization stages: horizontal decomposition,
vertical decomposition, then two successive refinements of the low-pass
component
"""

from __future__ import annotations

import base64
import io
import os
from typing import Dict, List

import numpy as np
import pywt
from flask import Blueprint, current_app, render_template
from PIL import Image


from .demo_images import cached_demo_image


def _image_path() -> str:
    _, path = cached_demo_image(current_app.static_folder)
    return str(path)


def _load_gray_image() -> np.ndarray:
    path = _image_path()
    with Image.open(path) as img:
        gray = img.convert("L")
        return np.asarray(gray, dtype=np.float64) / 255.0


def _normalize(band: np.ndarray) -> np.ndarray:
    minimum = float(np.min(band))
    maximum = float(np.max(band))
    span = maximum - minimum
    if span <= 1e-12:
        return np.zeros_like(band)
    return (band - minimum) / span


def _to_image(band: np.ndarray, size: tuple[int, int]) -> Image.Image:
    normalized = _normalize(band)
    arr = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize(size, Image.BICUBIC)


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _blank_canvas(width: int, height: int) -> Image.Image:
    return Image.new("L", (width, height), color=128)


def _build_stage_images(image: np.ndarray) -> Dict[str, str]:

    height, width = image.shape

    # Stage 0 - original (grayscale) for a consistent look
    original_img = _to_image(image, (width, height))

    # Stage 1: Horizontal subband decomposition
    low_h, high_h = pywt.dwt(image, "haar", axis=1, mode="periodization")
    left = _to_image(low_h, (width // 2, height))
    right = _to_image(high_h, (width - left.width, height))
    stage1 = _blank_canvas(width, height)
    stage1.paste(left, (0, 0))
    stage1.paste(right, (left.width, 0))

    # Stage 2: Vertical decomposition of both horizontal bands
    ll, lh = pywt.dwt(low_h, "haar", axis=0, mode="periodization")
    hl, hh = pywt.dwt(high_h, "haar", axis=0, mode="periodization")

    quad_w, quad_h = width // 2, height // 2
    ll_img = _to_image(ll, (quad_w, quad_h))
    lh_img = _to_image(lh, (quad_w, height - quad_h))
    hl_img = _to_image(hl, (width - quad_w, quad_h))
    hh_img = _to_image(hh, (width - quad_w, height - quad_h))

    stage2 = _blank_canvas(width, height)
    stage2.paste(ll_img, (0, 0))
    stage2.paste(hl_img, (quad_w, 0))
    stage2.paste(lh_img, (0, quad_h))
    stage2.paste(hh_img, (quad_w, quad_h))

    # Stage 3: Horizontal decomposition of the low-pass (LL) component
    lll, llh = pywt.dwt(ll, "haar", axis=1, mode="periodization")
    lll_img = _to_image(lll, (quad_w // 2, quad_h))
    llh_img = _to_image(llh, (quad_w - lll_img.width, quad_h))

    stage3 = _blank_canvas(width, height)
    stage3.paste(lll_img, (0, 0))
    stage3.paste(llh_img, (lll_img.width, 0))
    stage3.paste(hl_img, (quad_w, 0))
    stage3.paste(lh_img, (0, quad_h))
    stage3.paste(hh_img, (quad_w, quad_h))

    # Stage 4 Vertical decomposition of the refined low-pass (LLL)
    llll, lllh = pywt.dwt(lll, "haar", axis=0, mode="periodization")
    llll_img = _to_image(llll, (lll_img.width, quad_h // 2))
    lllh_img = _to_image(lllh, (lll_img.width, quad_h - llll_img.height))

    stage4 = _blank_canvas(width, height)
    stage4.paste(llll_img, (0, 0))
    stage4.paste(lllh_img, (0, llll_img.height))
    stage4.paste(llh_img, (lll_img.width, 0))
    stage4.paste(hl_img, (quad_w, 0))
    stage4.paste(lh_img, (0, quad_h))
    stage4.paste(hh_img, (quad_w, quad_h))

    return {
        "original": _encode_png(original_img),
        "horizontal": _encode_png(stage1),
        "vertical": _encode_png(stage2),
        "lp_horizontal": _encode_png(stage3),
        "lp_vertical": _encode_png(stage4),
    }


def _stage_metadata(images: Dict[str, str]) -> List[Dict[str, str]]:
    return [
        {
            "key": "original",
            "title": "Original image",
            "subtitle": "Reference demo frame (grayscale).",
            "src": images["original"],
        },
        {
            "key": "horizontal",
            "title": "Horizontal subband decomposition",
            "subtitle": "Haar analysis along rows splits the image into low- and high-pass halves.",
            "src": images["horizontal"],
        },
        {
            "key": "vertical",
            "title": "Vertical subband decomposition",
            "subtitle": "Each horizontal band is decomposed along columns into LL, LH, HL, and HH.",
            "src": images["vertical"],
        },
        {
            "key": "lp_horizontal",
            "title": "Horizontal decomposition of LP component",
            "subtitle": "The LL quadrant is further separated into an even smoother LLL and a detail LLH band.",
            "src": images["lp_horizontal"],
        },
        {
            "key": "lp_vertical",
            "title": "Vertical decomposition of LP component",
            "subtitle": "A final vertical split of LLL reveals lll↓ (low) and lll↑ (detail) layers.",
            "src": images["lp_vertical"],
        },
    ]

def _build_reconstruction_images(image: np.ndarray) -> List[str]:
    coeffs = pywt.wavedec2(image, "haar", mode="periodization", level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

    def _zero_like(arr: np.ndarray) -> np.ndarray:
        return np.zeros_like(arr)

    flags = {
        "cA2": True,
        "cV2": False,
        "cH2": False,
        "cD2": False,
        "cV1": False,
        "cH1": False,
        "cD1": False,
    }

    order = [
        ("ll2", "cA2"),
        ("lh2", "cV2"),
        ("hl2", "cH2"),
        ("hh2", "cD2"),
        ("lh1", "cV1"),
        ("hl1", "cH1"),
        ("hh1", "cD1"),
    ]

    reconstructions: List[str] = []
    width, height = image.shape[1], image.shape[0]

    for _, flag in order:
        flags[flag] = True
        coeffs_partial = [
            cA2 if flags["cA2"] else _zero_like(cA2),
            (
                cH2 if flags["cH2"] else _zero_like(cH2),
                cV2 if flags["cV2"] else _zero_like(cV2),
                cD2 if flags["cD2"] else _zero_like(cD2),
            ),
            (
                cH1 if flags["cH1"] else _zero_like(cH1),
                cV1 if flags["cV1"] else _zero_like(cV1),
                cD1 if flags["cD1"] else _zero_like(cD1),
            ),
        ]
        recon = pywt.waverec2(coeffs_partial, "haar", mode="periodization")
        recon = np.clip(recon, 0.0, 1.0)
        recon_img = _to_image(recon, (width, height))
        reconstructions.append(_encode_png(recon_img))

    return reconstructions


def _reconstruction_steps(recon_images: List[str]) -> List[Dict[str, str]]:
    image_steps = iter(recon_images)
    return [
        {
            "key": "ll2",
            "label": "LL-2",
            "desc": "Start from the coarsest approximation; it carries the broad luminance structure.",
            "active": ["ll2"],
            "src": next(image_steps),
        },
        {
            "key": "lh2",
            "label": "LH-2",
            "desc": "Add vertical edges from the second-level low-pass band.",
            "active": ["ll2", "lh2"],
            "src": next(image_steps),
        },
        {
            "key": "hl2",
            "label": "HL-2",
            "desc": "Blend in horizontal edges at the same coarse scale.",
            "active": ["ll2", "lh2", "hl2"],
            "src": next(image_steps),
        },
        {
            "key": "hh2",
            "label": "HH-2",
            "desc": "Introduce diagonal detail to complete the level-2 block.",
            "active": ["ll2", "lh2", "hl2", "hh2"],
            "src": next(image_steps),
        },
        {
            "key": "lh1",
            "label": "LH-1",
            "desc": "Layer in medium-scale vertical contrast from the next level up.",
            "active": ["ll2", "lh2", "hl2", "hh2", "lh1"],
            "src": next(image_steps),
        },
        {
            "key": "hl1",
            "label": "HL-1",
            "desc": "Restore medium-scale horizontal contrast.",
            "active": ["ll2", "lh2", "hl2", "hh2", "lh1", "hl1"],
            "src": next(image_steps),
        },
        {
            "key": "hh1",
            "label": "HH-1",
            "desc": "Finish with diagonal detail to reach the full-resolution reconstruction order.",
            "active": ["ll2", "lh2", "hl2", "hh2", "lh1", "hl1", "hh1"],
            "src": next(image_steps),
        },
    ]



demos_discrete_wavelet_transform_bp = Blueprint(
    "demos_discrete_wavelet_transform", __name__, template_folder="../../templates"
)


@demos_discrete_wavelet_transform_bp.route("/", methods=["GET"], endpoint="page")
def page():
    image = _load_gray_image()
    stage_images = _build_stage_images(image)
    stages = _stage_metadata(stage_images)
    recon_images = _build_reconstruction_images(image)
    return render_template(
        "demos/discrete_wavelet_transform.html",
        stages=stages,
        recon_steps=_reconstruction_steps(recon_images),
    )