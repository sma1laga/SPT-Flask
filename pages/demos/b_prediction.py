from __future__ import annotations

from dataclasses import dataclass
from flask import Blueprint, render_template, url_for


@dataclass
class BPredictionDefaults:
    image_prev: str
    image_curr: str
    image_next: str

b_prediction_bp = Blueprint("b_prediction", __name__, template_folder="././templates")


@b_prediction_bp.route("/", methods=["GET"], endpoint="page")
def index():
    defaults = BPredictionDefaults(
        image_prev=url_for("static", filename="images/np1.png", _external=False),
        image_curr=url_for("static", filename="images/np2.png", _external=False),
        image_next=url_for("static", filename="images/np3.png", _external=False),
    )
    return render_template("demos/b_prediction.html", defaults=defaults)