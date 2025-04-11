# pages/process_chain_plot.py
from flask import Blueprint, render_template

process_chain_bp = Blueprint("process_chain", __name__)

@process_chain_bp.route("/")
def show_chain():
    return render_template("process_chain_plot.html")
