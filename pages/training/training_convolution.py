# pages/training/training_convolution.py
from flask import Blueprint, render_template

training_convolution_bp = Blueprint("training_convolution", __name__, template_folder="templates/training")

@training_convolution_bp.route("/")
def training_convolution():
    return render_template("training_convolution.html")
