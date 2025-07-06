from pages.inverse_laplace import _parse_poly, _inverse_laplace_expr


def test_dirac_delta_from_constant_tf():
    num = _parse_poly('1')
    den = _parse_poly('1')
    expr = _inverse_laplace_expr(num, den)
    assert str(expr) == 'DiracDelta(t)'


def test_first_order_response():
    num = _parse_poly('1')
    den = _parse_poly('s+1')
    expr = _inverse_laplace_expr(num, den)
    assert str(expr) == 'exp(-t)*Heaviside(t)'