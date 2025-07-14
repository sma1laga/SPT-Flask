import pytest
from main import create_app

@pytest.fixture(scope='module')
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def check_route(client, url):
    resp = client.get(url)
    assert resp.status_code == 200


@pytest.mark.parametrize('url', [
    '/',
    '/plot_function/',
    '/fourier/',
    '/convolution/',
    '/autocorrelation/',
    '/block_diagram/',
    '/discrete/direct_plot/',
    '/inverse_z/',
    '/info/about'
])
def test_basic_pages(client, url):
    check_route(client, url)
