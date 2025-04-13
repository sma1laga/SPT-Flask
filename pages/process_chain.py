# pages/process_chain.py
from flask import Blueprint, render_template

process_chain_bp = Blueprint("process_chain", __name__)

@process_chain_bp.route("/")
def show_chain():
    """
    Renders an HTML page with a canvas-based chain builder.
    """
    return render_template("process_chain.html")
