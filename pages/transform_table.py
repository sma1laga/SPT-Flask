# pages/transform_table.py
from flask import Blueprint, render_template

transform_table_bp = Blueprint("transform_table", __name__)

@transform_table_bp.route("/")
def transform_table():
    return render_template("transform_table.html")
