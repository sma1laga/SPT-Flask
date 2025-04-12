# pages/process_chain.py
from flask import Blueprint, render_template

process_chain_bp = Blueprint("process_chain", __name__)

@process_chain_bp.route("/")
def show_chain():
    """
    A minimal route for 'Process Chain Plot'.
    Renders a simple template named 'process_chain.html'.
    """
    return render_template("process_chain.html")
