# pages/training/training_page.py
from flask import Blueprint, render_template

training_page_bp = Blueprint("training_page", __name__, template_folder="templates/training")

@training_page_bp.route("/")
def training_menu():
    return render_template("training_page.html")
