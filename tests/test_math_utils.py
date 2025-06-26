import numpy as np
import utils.math_utils as m
import pytest


def test_rect_tri_step():
    t = np.array([-1.0, -0.5, 0.0, 0.4, 1.2])
    assert np.allclose(m.rect(t), [0, 0, 1, 1, 0])
    assert np.allclose(m.tri(t), [0, 0.5, 1, 0.6, 0])
    assert np.allclose(m.step(t), [0, 0, 1, 1, 1])


def test_delta_properties():
    t = np.array([-0.001, 0.0, 0.001])
    d = m.delta(t)
    assert d[1] == pytest.approx(np.max(d))
    assert d[0] == pytest.approx(d[2])


def test_delta_n():
    n = np.array([-1, 0, 1])
    d = m.delta_n(n)
    assert np.array_equal(d, [0.0, 1.0, 0.0])