import pytest
from main import create_app

@pytest.fixture(scope='module')
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

def test_fourier_page_contains_script(client):
    resp = client.get('/fourier/')
    assert resp.status_code == 200
    assert b'fourier_compute.js' in resp.data