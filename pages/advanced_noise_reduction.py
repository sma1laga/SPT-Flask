from flask import Blueprint, render_template

advanced_noise_reduction_bp = Blueprint("advanced_noise_reduction", __name__)
@advanced_noise_reduction_bp.route("/", methods=["GET"])

def advanced_noise_reduction():
    return render_template("advanced_noise_reduction.html")
