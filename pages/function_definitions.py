# pages/function_definitions.py
from flask import Blueprint, render_template

func_defs_bp = Blueprint("function_definitions", __name__)

@func_defs_bp.route("/")
def function_definitions():
    return render_template("function_definitions.html")
