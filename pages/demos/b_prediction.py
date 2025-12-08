from __future__ import annotations

from dataclasses import dataclass

from flask import Blueprint, current_app, render_template, url_for

from .demo_images import cached_demo_image, static_image_filename

@dataclass
class BPredictionDefaults:
    image_src: str


b_prediction_bp = Blueprint("b_prediction", __name__, template_folder="././templates")


@b_prediction_bp.route("/", methods=["GET"], endpoint="page")
def index():
    image_name, _ = cached_demo_image(current_app.static_folder)
    defaults = BPredictionDefaults(
        image_src=url_for("static", filename=static_image_filename(image_name)),
    )
    return render_template("demos/b_prediction.html", defaults=defaults)