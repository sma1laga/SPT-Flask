# pages/process_chain_web.py
from flask import Blueprint, render_template

process_chain_web_bp = Blueprint("process_chain_web", __name__)

@process_chain_web_bp.route("/")
def show_process_chain_web():
    """
    Renders the web-based process chain plot (HTML + JS).
    """
    return render_template("process_chain_web.html")
