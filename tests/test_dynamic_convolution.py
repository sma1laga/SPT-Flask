"""Contract tests for the dynamic convolution endpoint.

The page offers a fixed dropdown built from ``utils.dynamic_convolution.functions``
while the endpoint evaluates the expression against a separate allowlist. Those
two drifted apart once already -- ``exp_iwt`` was dropped from the evaluation
context while a test still asked for it -- so the list itself is now the test
data.
"""

import json

import numpy as np
import pytest

from main import create_app
from utils.dynamic_convolution import functions


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def convolve(client, func1, func2):
    resp = client.post(
        "/convolution/dynamic/data",
        data=json.dumps({"func1": func1, "func2": func2}),
        content_type="application/json",
    )
    return resp


@pytest.mark.parametrize("expression", [expr for _, expr in functions],
                         ids=[expr for _, expr in functions])
def test_every_offered_function_can_be_evaluated(client, expression):
    """Anything selectable in the dropdown has to survive the evaluator."""
    resp = convolve(client, expression, "rect(t)")
    assert resp.status_code == 200, resp.get_json()


def test_response_arrays_line_up(client):
    data = convolve(client, "rect(t)", "rect(t)").get_json()
    assert set(data) >= {"t", "y1", "y2", "y_conv"}
    assert len(data["t"]) == len(data["y1"]) == len(data["y2"]) == len(data["y_conv"])
    assert len(data["t"]) > 100


def test_rect_convolved_with_rect_is_a_triangle(client):
    data = convolve(client, "rect(t)", "rect(t)").get_json()
    t = np.array(data["t"])
    y = np.array(data["y_conv"])
    assert np.interp(0.0, t, y) == pytest.approx(1.0, abs=0.01)
    assert np.interp(0.5, t, y) == pytest.approx(0.5, abs=0.01)
    assert np.interp(-0.5, t, y) == pytest.approx(0.5, abs=0.01)
    assert np.interp(1.2, t, y) == pytest.approx(0.0, abs=0.01)


def test_convolving_with_a_shifted_delta_shifts_the_signal(client):
    """x(t) * delta(t - 2) = x(t - 2)."""
    data = convolve(client, "rect(t)", "delta(t-2)").get_json()
    t = np.array(data["t"])
    y = np.array(data["y_conv"])
    assert np.interp(2.0, t, y) == pytest.approx(1.0, abs=0.02)
    assert np.interp(0.0, t, y) == pytest.approx(0.0, abs=0.02)


def test_unknown_function_is_rejected(client):
    resp = convolve(client, "foo(t)", "rect(t)")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_syntax_error_is_rejected(client):
    resp = convolve(client, "rect(t)+", "rect(t)")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_empty_input_is_treated_as_zero(client):
    resp = convolve(client, "", "rect(t)")
    assert resp.status_code == 200
    assert not np.any(np.array(resp.get_json()["y_conv"]))
