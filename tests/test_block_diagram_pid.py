import pytest
from pages.block_diagram.services import compile_diagram

def test_compile_diagram_pid():
    graph = {
        "nodes": [
            {"id": 1, "type": "Input", "label": "X(s)", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {"kind": "impulse"}},
            {"id": 2, "type": "PID", "label": "PID", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {"kp": 1, "ki": 2, "kd": 3}},
            {"id": 3, "type": "TF", "label": "G(s)", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {"num": "1", "den": "s+1"}},
            {"id": 4, "type": "Output", "label": "Y(s)", "x": 0, "y": 0, "w": 0, "h": 0,
             "params": {}}
        ],
        "edges": [
            {"from": 1, "to": 2, "sign": "+"},
            {"from": 2, "to": 3, "sign": "+"},
            {"from": 3, "to": 4, "sign": "+"}
        ],
        "domain": "s"
    }

    res = compile_diagram(graph, domain="s")
    assert res["loop_tf"]["num"] == [3.0, 1.0, 2.0]
    assert res["loop_tf"]["den"] == [1.0, 1.0, 0.0]