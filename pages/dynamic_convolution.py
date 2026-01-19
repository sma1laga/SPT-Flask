# pages/dynamic_convolution.py
from flask import Blueprint, render_template, request
from utils.dynamic_convolution import functions, conv_json


dynamic_convolution_bp = Blueprint("dynamic_convolution", __name__)

@dynamic_convolution_bp.route("/", methods=["GET"])
def dynamic_convolution():
    return render_template("dynamic_convolution.html", functions=functions)

@dynamic_convolution_bp.route("/data", methods=["POST"])
def dynamic_data():
    return conv_json(request.get_json(force=True))
