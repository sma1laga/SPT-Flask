from __future__ import annotations

from flask import Blueprint, render_template

laplace_intuition_bp = Blueprint("laplace_intuition", __name__)


@laplace_intuition_bp.route("/")
def laplace_intuition():
    """Interactive Laplace intuition for a force mass–spring system"""
    return render_template("laplace_intuition.html")