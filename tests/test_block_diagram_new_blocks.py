import pytest
from pages.block_diagram.services import compile_diagram
from main import create_app



def test_zero_pole_block():
    graph = {
        "nodes": [
            {"id": 1, "type": "Input", "label": "X(s)", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {"kind": "impulse"}},
            {"id": 2, "type": "ZeroPole", "label": "Z/P", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {"zeros": "-1", "poles": "-2", "k": 3}},
            {"id": 3, "type": "Output", "label": "Y(s)", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {}},
        ],
        "edges": [
            {"from": 1, "to": 2, "sign": "+"},
            {"from": 2, "to": 3, "sign": "+"},
        ],
        "domain": "s",
    }

    res = compile_diagram(graph, domain="s")
    assert res["loop_tf"]["num"] == [3.0, 3.0]
    assert res["loop_tf"]["den"] == [1.0, 2.0]


def test_saturation_and_scope_block():
    graph = {
        "nodes": [
            {"id": 1, "type": "Input", "label": "X(s)", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {"kind": "impulse"}},
            {"id": 2, "type": "Gain", "label": "K", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {"k": 2}},
            {"id": 3, "type": "Saturation", "label": "Sat", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {"lower": -1, "upper": 1}},
            {"id": 4, "type": "Scope", "label": "Scope", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {}},
            {"id": 5, "type": "Output", "label": "Y(s)", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {}},
        ],
        "edges": [
            {"from": 1, "to": 2, "sign": "+"},
            {"from": 2, "to": 3, "sign": "+"},
            {"from": 3, "to": 5, "sign": "+"},
            {"from": 3, "to": 4, "sign": "+"},
        ],
        "domain": "s",
    }

    res = compile_diagram(graph, domain="s")
    assert res["saturation"] == {"lower": -1.0, "upper": 1.0}
    assert res["scopes"]["4"]["num"] == [2.0]
    assert res["scopes"]["4"]["den"] == [1.0]

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        payload = {
            "num": res["output_tf"]["num"],
            "den": res["output_tf"]["den"],
            "saturation": res["saturation"],
        }
        resp = client.post("/block_diagram/simulate", json=payload)
        assert resp.status_code == 200
        sim = resp.get_json()
        assert max(sim["y"]) == pytest.approx(1.0)
        assert min(sim["y"]) == pytest.approx(1.0)


def test_delay_block():
    graph = {
        "nodes": [
            {"id": 1, "type": "Input", "label": "X(s)", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {"kind": "impulse"}},
            {"id": 2, "type": "Delay", "label": "Delay", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {"tau": 1}},
            {"id": 3, "type": "Output", "label": "Y(s)", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {}},
        ],
        "edges": [
            {"from": 1, "to": 2, "sign": "+"},
            {"from": 2, "to": 3, "sign": "+"},
        ],
        "domain": "s",
    }

    res = compile_diagram(graph, domain="s")
    assert res["loop_tf"]["num"] == [-1.0, 2.0]
    assert res["loop_tf"]["den"] == [1.0, 2.0]
