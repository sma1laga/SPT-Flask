# pages/demos/convolution.py
from flask import Blueprint, render_template
from utils.dynamic_convolution import functions

# Blueprint in demos namespace
demos_convolution_bp = Blueprint(
    "demos_convolution", __name__
)

@demos_convolution_bp.route("/", methods=["GET"], endpoint="page")
def dynamic_convolution():
    return render_template(
        "/dynamic_convolution_demo.html",
        functions=functions,
        template_folder="../../templates"
    )
