# pages/filter_design.py
from flask import Blueprint, render_template

filter_design_bp = Blueprint("filter_design", __name__)

@filter_design_bp.route("/")
def filter_design():
    return render_template("filter_design.html")
