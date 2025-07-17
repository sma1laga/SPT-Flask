# pages/theory.py
from flask import Blueprint, render_template

theory_bp = Blueprint("theory", __name__)

@theory_bp.route("/")
def theory():
    """Display the adapted table of contents."""
    return render_template("theory.html")
