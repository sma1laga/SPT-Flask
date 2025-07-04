import numpy as np
from pages.inverse_z import _parse_poly, _inverse_z_expr, _impulse_response


def test_simple_impulse_response():
    num = _parse_poly('[1, -0.5]')
    den = _parse_poly('[1, -0.3]')
    expr = _inverse_z_expr(num, den)
    # first coefficient should be 1
    seq = _impulse_response(num, den, N=3)
    assert np.isclose(seq[0], 1.0)
    assert len(seq) == 3