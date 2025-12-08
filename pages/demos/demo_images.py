"""Utilities for selecting demo images for compression showcase"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Tuple

DEMO_IMAGES = ("cameraman.tiff", "pirate.tiff", "peppers.tiff")


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