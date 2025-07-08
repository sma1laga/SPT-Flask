import numpy as np
from pages.inverse_z import _parse_poly, _inverse_z_expr, _impulse_response


def test_simple_impulse_response():
    num = _parse_poly('[1, -0.5]')
    den = _parse_poly('[1, -0.3]')
    seq = _impulse_response(num, den, n_samples=3)

    assert seq[0] == 1
    assert len(seq) == 3

def test_roc_behaviour():
    num = _parse_poly('[1]')
    den = _parse_poly('[1, -0.5]')
    expr_left = _inverse_z_expr(num, den, roc=0.1)
    assert 'Heaviside(-k - 1)' in str(expr_left)
    expr_right = _inverse_z_expr(num, den, roc=2)
    assert 'Heaviside(k)' in str(expr_right)