# pages/training/training_fourier.py
from flask import Blueprint, render_template

training_fourier_bp = Blueprint("training_fourier", __name__, template_folder="templates/training")

@training_fourier_bp.route("/")
def training_fourier():
    return render_template("training_fourier.html")
