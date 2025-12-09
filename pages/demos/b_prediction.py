from __future__ import annotations

from dataclasses import dataclass
import base64
import io
from flask import Blueprint, current_app, render_template
from PIL import Image
from .demo_images import cached_demo_image

@dataclass
class BPredictionDefaults:
    image_src: str


b_prediction_bp = Blueprint("b_prediction", __name__, template_folder="././templates")


@b_prediction_bp.route("/", methods=["GET"], endpoint="page")
def index():
    _, image_path = cached_demo_image(current_app.static_folder)

    with Image.open(image_path) as img:
        with io.BytesIO() as buffer:
            img.convert("RGB").save(buffer, format="PNG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode()

    defaults = BPredictionDefaults(image_src=f"data:image/png;base64,{encoded_image}")
    return render_template("demos/b_prediction.html", defaults=defaults)