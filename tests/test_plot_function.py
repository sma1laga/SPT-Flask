import json
from main import create_app
import pytest

@pytest.fixture(scope='module')
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_plot_function_update_basic(client):
    payload = {'func1': 'sin(t)'}
    resp = client.post('/plot_function/update', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 't1' in data and 'y1' in data
    assert len(data['t1']) == len(data['y1'])