from flask import Blueprint, render_template
from pages.convolution import compute_convolution
import re


autocorrelation_bp = Blueprint("autocorrelation", __name__)

@autocorrelation_bp.route("/", methods=["GET"])
def autocorrelation():
    return render_template("autocorrelation.html")


def _reverse_t(expr: str) -> str:
    """Return the expression with every standalone `t` replaced by `-t`."""
    return re.sub(r"\bt\b", "(-t)", expr)


def compute_autocorrelation(func_str: str):
    """Compute the autocorrelation via convolution with the time-reversed signal."""
    return compute_convolution(func_str, _reverse_t(func_str))