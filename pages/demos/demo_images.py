"""Utilities for selecting demo images for compression showcase"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Tuple
import numpy as np

DEMO_IMAGES = ("cameraman.tif", "pirate.tiff", "peppers.tiff")


def choose_demo_image(static_folder: str | Path) -> Tuple[str, Path]:
    static_root = Path(static_folder)
    name = random.choice(DEMO_IMAGES)
    path = static_root / "images" / name
    if not path.exists():
        raise FileNotFoundError(f"Demo image not found: {path}")
    return name, path


@lru_cache(maxsize=1)
def cached_demo_image(static_folder: str | Path) -> Tuple[str, Path]:
    return choose_demo_image(str(static_folder))


def static_image_filename(image_name: str) -> str:
    return f"images/{image_name}"


def browser_safe_image_filename(static_folder: str | Path) -> str:
    """Return a browser-friendly static filename, converting TIFFs to PNGs if needed."""

    image_name, image_path = cached_demo_image(static_folder)

    if image_path.suffix.lower() not in {".tif", ".tiff"}:
        return static_image_filename(image_name)

    web_dir = Path(static_folder) / "images" / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    png_name = f"{image_path.stem}.png"
    png_path = web_dir / png_name

    if not png_path.exists() or png_path.stat().st_mtime < image_path.stat().st_mtime:
        import imageio.v3 as iio

        img = iio.imread(image_path)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        iio.imwrite(png_path, img)

    return str(png_path.relative_to(static_folder)).replace("\\", "/")