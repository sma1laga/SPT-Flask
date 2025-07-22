import json
from flask import request

CRASH_LOG_FILE = 'crashes.log'


def log_exception(exc: Exception) -> None:
    """Write crash info to CRASH_LOG_FILE in JSON lines format."""
    data = {
        'path': request.path,
        'method': request.method,
        'endpoint': request.endpoint,
        'args': request.args.to_dict(flat=True),
        'form': request.form.to_dict(flat=True),
        'json': request.get_json(silent=True),
        'error': f'{type(exc).__name__}: {exc}',
    }
    try:
        with open(CRASH_LOG_FILE, 'a') as fh:
            fh.write(json.dumps(data) + '\n')
    except Exception:
        pass