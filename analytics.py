import uuid
import time
import json
import requests
from flask import Blueprint, request, session

analytics_bp = Blueprint('analytics', __name__)

LOG_FILE = 'analytics.log'


def _get_country(ip: str) -> str:
    """Lookup country name from IP address using ipapi.co."""
    try:
        r = requests.get(f'https://ipapi.co/{ip}/country_name/', timeout=1)
        if r.ok:
            return r.text.strip() or 'Unknown'
    except Exception:
        pass
    return 'Unknown'


@analytics_bp.before_app_request
def _start_timer():
    if 'analytics_id' not in session:
        session['analytics_id'] = str(uuid.uuid4())
        session['session_start'] = time.time()
    session['last_seen'] = time.time()


@analytics_bp.after_app_request
def _log_request(response):
    try:
        ip = request.headers.get('X-Forwarded-For', request.remote_addr)
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
            'country': _get_country(ip),
        }
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(data) + '\n')
    except Exception:
        pass
    return response