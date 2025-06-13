from flask import Blueprint, render_template

speech_filter_input_bp = Blueprint("speech_filter_input", __name__)

@speech_filter_input_bp.route("/", methods=["GET"])
def speech_filter_input():
        return render_template("speech_filter_input.html")
