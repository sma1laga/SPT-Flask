# pages/demos/menu.py
from flask import Blueprint, render_template
from .data import DEMOS

demos_menu_bp = Blueprint(
    "demos_menu", __name__, template_folder="../../templates"
)

@demos_menu_bp.route("/", methods=["GET"])
def index():
    """Render the demo overview.

    The demos are grouped in a nested dictionary so that the template can
    display them in separate "Course" and "Tutorial" cards for each of the
    two lecture parts (Signals and Systems I & II).
    """

    return render_template("demos/menu.html", demos=DEMOS)

