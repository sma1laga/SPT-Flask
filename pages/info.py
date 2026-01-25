# pages/info.py
from flask import Blueprint, render_template, request
import json
from datetime import timezone, datetime

CONTACT_LOG_FILE = 'contact.log'
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


@info_bp.route('/hall-of-fame')
def hall_of_fame():
    """community members who contributed"""
    return render_template('hall_of_fame.html')

@info_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        message = request.form.get('message', '').strip()
        if '@' not in email:
            error = 'Please enter a valid email address.'
            return render_template('contact.html', error=error, name=name,
                                   email=email, message=message)
        data = {
            'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'name': name,
            'email': email,
            'message': message,
        }
        try:
            with open(CONTACT_LOG_FILE, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception:
            pass
        success = 'Thank you for your report!'
        return render_template('contact.html', success=success)
    return render_template('contact.html')


@info_bp.route('/impressum')
def impressum():
    return render_template('impressum.html')


@info_bp.route('/privacy')
def privacy():
    return render_template('privacy_policy.html')


@info_bp.route('/terms')
def terms():
    return render_template('terms_of_service.html')


@info_bp.route('/news')
def news():
    releases = [
        # Default template with: version / codename / date / whats the acctual update (highlights)
        # @Paul - noch nicht wichtig, erst nach launch dann...
        # { 
        #     "version": "1.1.1vUI",
        #     "codename": "UI Refresh",
        #     "date": "September 20, 2025",
        #     "highlights": [
        #         "Refined dashboard styling for better readability.",
        #         "Introduced the dedicated news page for release notes.",
        #         "Improved footer accessibility links."
        #     ],
        # },

    ]
    return render_template('news.html', releases=releases)