import json
import hashlib
import traceback
from datetime import datetime, timezone
from flask import request
from flask import has_request_context

CRASH_LOG_FILE = 'crashes.log'


def log_exception(exc: Exception) -> None:
    """Write crash info to CRASH_LOG_FILE in JSON lines format."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    stack_trace = ''.join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )

    if has_request_context():
        headers = request.headers
        forwarded_for = headers.get('X-Forwarded-For', '')
        client_ip = forwarded_for.split(',')[0].strip() if forwarded_for else request.remote_addr
        user_agent = headers.get('User-Agent')
        client_key_source = f'{client_ip or "unknown"}|{user_agent or "unknown"}'
        client_fingerprint = hashlib.sha256(client_key_source.encode('utf-8')).hexdigest()[:16]
    else:
        headers = {}
        client_ip = None
        user_agent = None
        client_fingerprint = None

    data = {
        'timestamp': timestamp,
        'error_type': type(exc).__name__,
        'error_message': str(exc),
        'error': f'{type(exc).__name__}: {exc}',
        'stack_trace': stack_trace,
        'path': request.path if has_request_context() else None,
        'method': request.method if has_request_context() else None,
        'endpoint': request.endpoint if has_request_context() else None,
        'query_string': request.query_string.decode('utf-8', errors='replace') if has_request_context() else None,
        'args': request.args.to_dict(flat=True) if has_request_context() else {},
        'form': request.form.to_dict(flat=True) if has_request_context() else {},
        'json': request.get_json(silent=True) if has_request_context() else None,
        'client_ip': client_ip,
        'user_agent': user_agent,
        'referrer': headers.get('Referer') if has_request_context() else None,
        'origin': headers.get('Origin') if has_request_context() else None,
        'anonymous_user_id': client_fingerprint,
    }
    try:
        with open(CRASH_LOG_FILE, 'a') as fh:
            fh.write(json.dumps(data) + '\n')
    except Exception:
        pass