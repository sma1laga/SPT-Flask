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

@info_bp.route('/polezero')
def polezero():
    # This template displays the LaTeX document for pole–zero information (embedded as a PDF).
    return render_template('info_polezero.html')

@info_bp.route('/noise_reduction')
def noise_reduction():
    # Render an HTML template that embeds your LaTeX PDF or displays the theoretical explanation.
    return render_template('info_noise_reduction.html')

@info_bp.route('/about')
def about():
    return render_template('about.html')  # Create about.html template

@info_bp.route('/contact')
def contact():
    return render_template('contact.html')  # Create contact.html template