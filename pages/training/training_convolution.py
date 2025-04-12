# pages/training/training_convolution.py
from flask import Blueprint, render_template

# The blueprint is named "training_convolution"
training_convolution_bp = Blueprint("training_convolution", __name__)

@training_convolution_bp.route("/")
def training_convolution():
    # Looks for "training/training_convolution.html" OR "training_convolution.html" 
    # depending on your approach (see note below).
    return render_template("training_convolution.html")
