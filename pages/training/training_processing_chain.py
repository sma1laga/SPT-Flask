# pages/training/training_processing_chain.py
from flask import Blueprint, render_template

training_processing_chain_bp = Blueprint("training_processing_chain", __name__)

@training_processing_chain_bp.route("/")
def training_processing_chain():
    return render_template("training_processing_chain.html")
