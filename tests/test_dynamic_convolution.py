import json
from main import create_app


def test_dynamic_convolution_exp_iwt():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        payload = {"func1": "exp_iwt(t)", "func2": "rect(t)"}
        resp = client.post('/convolution/dynamic/data',
                           data=json.dumps(payload),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 't' in data and 'y_conv' in data
        assert len(data['t']) == len(data['y_conv'])