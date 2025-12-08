from __future__ import annotations

from dataclasses import dataclass

from flask import Blueprint, render_template, url_for


@dataclass
class BPredictionDefaults:
    image_src: str


b_prediction_bp = Blueprint("b_prediction", __name__, template_folder="././templates")


@b_prediction_bp.route("/", methods=["GET"], endpoint="page")
def index():
    defaults = BPredictionDefaults(
        image_src=url_for("static", filename="demos/images/lenna.png"),
    )
    return render_template("demos/b_prediction.html", defaults=defaults)