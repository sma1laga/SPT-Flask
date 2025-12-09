"""Block matching dem"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from flask import Blueprint, render_template, url_for

from .demo_images import static_image_filename

@dataclass
class BlockMatchingDefaults:
    image_src: str
    canvas_size: int
    block_sizes: List[int]
    default_block: int
    default_search: int
    default_shift_x: int
    default_shift_y: int
    default_x: int
    default_y: int


demos_block_matching_bp = Blueprint(
    "demos_block_matching", __name__, template_folder="././templates"
)


@demos_block_matching_bp.route("/", methods=["GET"], endpoint="page")
def index():
    image_name = "peppers.png"
    defaults = BlockMatchingDefaults(
        image_src=url_for("static", filename=static_image_filename(image_name)),
        canvas_size=384,          # lil bigger than 320
        block_sizes=[8, 12, 16, 20, 24],
        default_block=16,
        default_search=10,
        default_shift_x=6,
        default_shift_y=-4,
        default_x=96,
        default_y=96,
    )
    return render_template("demos/block_matching.html", defaults=defaults)
