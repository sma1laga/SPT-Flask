import pytest
from main import create_app

@pytest.fixture(scope='module')
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

def test_convolution_page_contains_script(client):
    resp = client.get('/convolution/')
    assert resp.status_code == 200
    assert b'convolution_compute.js' in resp.data