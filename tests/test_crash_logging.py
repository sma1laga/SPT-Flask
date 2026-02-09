import json
import crash_logging
from flask import Flask



def test_crash_logged_with_extended_context(tmp_path, monkeypatch):
    log_file = tmp_path / 'crash.log'
    monkeypatch.setattr(crash_logging, 'CRASH_LOG_FILE', str(log_file))

    app = Flask(__name__)
    app.config['TESTING'] = False

    @app.route('/cause_error')
    def cause_error():
        raise ValueError('boom')
    @app.errorhandler(Exception)
    def handle_exc(exc):
        crash_logging.log_exception(exc)
        return ('Internal Server Error', 500)

    with app.test_client() as client:
        resp = client.get(
            '/cause_error?sample=1',
            headers={
                'User-Agent': 'pytest-agent',
                'Referer': 'https://example.org/page',
                'Origin': 'https://example.org',
                'X-Forwarded-For': '203.0.113.5',
            },
        )
        assert resp.status_code == 500

    lines = log_file.read_text().splitlines()
    assert lines, 'log should contain one entry'
    data = json.loads(lines[0])
    assert data['endpoint'] == 'cause_error'
    assert data['path'] == '/cause_error'
    assert data['method'] == 'GET'
    assert data['args'] == {'sample': '1'}
    assert data['query_string'] == 'sample=1'
    assert data['error_type'] == 'ValueError'
    assert data['error_message'] == 'boom'
    assert data['client_ip'] == '203.0.113.5'
    assert data['user_agent'] == 'pytest-agent'
    assert data['referrer'] == 'https://example.org/page'
    assert data['origin'] == 'https://example.org'
    assert len(data['anonymous_user_id']) == 16
    assert 'ValueError: boom' in data['stack_trace']
    assert data['timestamp']