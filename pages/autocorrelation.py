from flask import Blueprint, render_template
from pages.convolution import compute_convolution

autocorrelation_bp = Blueprint("autocorrelation", __name__)

@autocorrelation_bp.route("/", methods=["GET"])
def autocorrelation():
    return render_template("autocorrelation.html")


def compute_autocorrelation(func_str):
    return compute_convolution(func_str, func_str)