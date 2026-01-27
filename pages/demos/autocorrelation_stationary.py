# pages/demos/autocorrelation_stationary.py
"""Autocorrelation demo for stationary AR(1) processes"""

from flask import Blueprint, render_template


demos_autocorrelation_stationary_bp = Blueprint(
    "demos_autocorrelation_stationary", __name__, template_folder="../../templates"
)


@demos_autocorrelation_stationary_bp.route("/", methods=["GET"], endpoint="page")
def page():
    return render_template("demos/autocorrelation_stationary.html")