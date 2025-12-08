"""Lloyd-Max quantization demo page"""
from __future__ import annotations

from flask import Blueprint, render_template, url_for


demos_lloyd_max_bp = Blueprint(
    "demos_lloyd_max", __name__, template_folder="../../templates"
)


@demos_lloyd_max_bp.route("/", methods=["GET"], endpoint="page")
def index():
    """Render the interactive Lloyd-Max quantization demo."""

    return render_template(
        "demos/lloyd_max.html",
        lenna_src=url_for("static", filename="demos/images/lenna.png"),
    )