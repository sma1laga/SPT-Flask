import pytest
from pages.block_diagram.services import compile_diagram


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
