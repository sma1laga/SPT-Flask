import io
import pytest
from main import create_app
from pages.image_filter import MAX_FILE_SIZE


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_rejects_large_image(client):
    data = {
        'img_choice': 'upload',
        'image_file': (io.BytesIO(b'x' * (MAX_FILE_SIZE + 1)), 'test.png'),
    }
    resp = client.post('/image_filter/', data=data, content_type='multipart/form-data')
    assert b'File too large' in resp.data


def test_rejects_non_image(client):
    data = {
        'img_choice': 'upload',
        'image_file': (io.BytesIO(b'notanimage'), 'bad.png'),
    }
    resp = client.post('/image_filter/', data=data, content_type='multipart/form-data')
    assert b'Invalid image file' in resp.data