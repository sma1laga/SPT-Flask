import uuid
import time
import json
from flask import Blueprint, request, session

analytics_bp = Blueprint('analytics', __name__)

LOG_FILE = 'analytics.log'


def _get_country() -> str:
    """Return country from request data or stored session."""
    country = None
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        country = payload.get('country')
    if not country:
        country = request.form.get('country')
    if country:
        session['country'] = country
        return country
    return session.get('country', 'Unknown')


@analytics_bp.before_app_request
def _start_timer():
    if 'analytics_id' not in session:
        session['analytics_id'] = str(uuid.uuid4())
        session['session_start'] = time.time()
    session['last_seen'] = time.time()


@analytics_bp.route('/analytics/country', methods=['POST'])
def save_country():
    """Persist the reported country in the session."""
    _get_country()
    return ('', 204)


@analytics_bp.after_app_request
def _log_request(response):
    try:
        if request.endpoint == 'analytics.save_country' or request.path == '/analytics/country':
            return response
        data = {
            'id': session.get('analytics_id'),
            'path': request.path,
            'method': request.method,
            'timestamp': time.time(),
            'session_start': session.get('session_start'),
            'last_seen': session.get('last_seen'),
            'session_length': session.get('last_seen', 0) - session.get('session_start', 0),
            'user_agent': request.headers.get('User-Agent'),
            'device': request.user_agent.platform,
            'country': _get_country(),
        }
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(data) + '\n')
    except Exception:
        pass
    return response