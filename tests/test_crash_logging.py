import json
import crash_logging
from main import create_app


def test_crash_logged(tmp_path, monkeypatch):
    log_file = tmp_path / 'crash.log'
    monkeypatch.setattr(crash_logging, 'CRASH_LOG_FILE', str(log_file))

    app = create_app()
    app.config['TESTING'] = True

    @app.route('/cause_error')
    def cause_error():
        raise ValueError('boom')

    with app.test_client() as client:
        resp = client.get('/cause_error')
        assert resp.status_code == 500

    lines = log_file.read_text().splitlines()
    assert lines, 'log should contain one entry'
    data = json.loads(lines[0])
    assert data['endpoint'] == 'cause_error'
    assert data['path'] == '/cause_error'