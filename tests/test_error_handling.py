import json
import pytest
from main import create_app
from pages.fourier_page import compute_fourier
from pages.convolution import compute_convolution
import analytics
from unittest.mock import Mock
import requests


def test_compute_fourier_invalid_function_returns_error():
    res = compute_fourier('invalid(', 0.0)
    assert 'error' in res
    assert 'Error evaluating function' in res['error']


def test_compute_convolution_invalid_function_returns_error():
    res = compute_convolution('rect(t)', 'invalid(')
    assert 'error' in res
    assert 'Error evaluating Function 2' in res['error']


def test_plot_function_update_invalid_expression(monkeypatch):
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        payload = {'func1': 'sin('}
        resp = client.post('/plot_function/update', data=json.dumps(payload), content_type='application/json')
        assert resp.status_code == 400
        data = resp.get_json()
        assert 'error' in data


def test_get_country_success(monkeypatch):
    fake_resp = Mock(ok=True, text='Canada')
    monkeypatch.setattr(analytics.requests, 'get', lambda url, timeout=1: fake_resp)
    assert analytics._get_country('1.2.3.4') == 'Canada'


def test_get_country_failure(monkeypatch):
    def raise_exc(url, timeout=1):
        raise requests.RequestException
    monkeypatch.setattr(analytics.requests, 'get', raise_exc)
    assert analytics._get_country('1.2.3.4') == 'Unknown'


def test_analytics_logging(tmp_path, monkeypatch):
    log_file = tmp_path / 'log.jsonl'
    monkeypatch.setattr(analytics, 'LOG_FILE', str(log_file))
    monkeypatch.setattr(analytics, '_get_country', lambda ip: 'Testland')
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        resp = client.get('/info/about')
        assert resp.status_code == 200
    data = json.loads(log_file.read_text().splitlines()[0])
    assert data['path'] == '/info/about'
    assert data['country'] == 'Testland'
    assert data['method'] == 'GET'