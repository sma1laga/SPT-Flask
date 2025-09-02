import json
import pytest
from main import create_app
from pages.fourier_page import compute_fourier
from pages.convolution import compute_convolution
from pages.autocorrelation import compute_autocorrelation
import analytics



def test_compute_fourier_invalid_function_returns_error():
    res = compute_fourier('invalid(', 0.0)
    assert 'error' in res
    assert 'Error evaluating function' in res['error']


def test_compute_convolution_invalid_function_returns_error():
    res = compute_convolution('rect(t)', 'invalid(')
    assert 'error' in res
    assert 'Error evaluating Function 2' in res['error']


def test_compute_autocorrelation_invalid_function_returns_error():
    res = compute_autocorrelation('invalid(')
    assert 'error' in res
    assert 'Error evaluating Function 1' in res['error']


def test_plot_function_update_invalid_expression():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        payload = {'func1': 'sin('}
        resp = client.post('/plot_function/update', data=json.dumps(payload), content_type='application/json')
        assert resp.status_code == 400
        data = resp.get_json()
        assert 'error' in data


def test_country_logged(tmp_path, monkeypatch):

    log_file = tmp_path / 'log.jsonl'
    monkeypatch.setattr(analytics, 'LOG_FILE', str(log_file))
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        client.post('/analytics/country', json={'country': 'Testland'})
        resp = client.get('/info/about')
        assert resp.status_code == 200
    data = json.loads(log_file.read_text().splitlines()[0])
    assert data['path'] == '/info/about'
    assert data['country'] == 'Testland'
    assert data['method'] == 'GET'


def test_country_defaults_to_unknown(tmp_path, monkeypatch):
    log_file = tmp_path / 'log.jsonl'
    monkeypatch.setattr(analytics, 'LOG_FILE', str(log_file))
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        resp = client.get('/info/about')
        assert resp.status_code == 200
    data = json.loads(log_file.read_text().splitlines()[0])
    assert data['country'] == 'Unknown'