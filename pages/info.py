# pages/info.py
from flask import Blueprint, render_template

info_bp = Blueprint('info', __name__)

@info_bp.route('/bandpass_order')
def bandpass_order():
    # Renders the info page for bandpass order details.
    return render_template('bandpass_order.html')

@info_bp.route('/sampling')
def sampling():
    # Renders the info page for sampling details.
    return render_template('info_sampling.html')
