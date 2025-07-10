import numpy as np
import sympy as sp
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
    expr_left = _inverse_z_expr(num, den, roc=0.1, roc_type="inside")
    assert 'Heaviside(-k - 1)' in str(expr_left)
    expr_right = _inverse_z_expr(num, den, roc=2, roc_type="outside")
    assert 'Heaviside(k)' in str(expr_right)

def test_left_boundary_case():
    num = _parse_poly('[1, 0]')
    den = _parse_poly('[2, -3, 1]')
    expr = _inverse_z_expr(num, den, roc=0.5, roc_type="inside")
    ltx = sp.latex(expr).replace('\\theta', '\\varepsilon')
    cleaned = (
        ltx.replace('\\left', '')
        .replace('\\right', '')
        .replace('(- k - 1)', '[-k-1]')
        .replace('\\frac{2^{1 - k} \\varepsilon[-k-1]}{2}', '(\\frac12)^k \\varepsilon[-k-1]')
        .replace('(\\frac12)^k \\varepsilon[-k-1] - \\varepsilon[-k-1]', '-\\varepsilon[-k-1] + (\\frac12)^k \\varepsilon[-k-1]')
        .replace('(\\frac12)^k', '\\left(\\frac12\\right)^k')
    )
    assert cleaned in {
        r"-\varepsilon[-k-1] + \left(\frac12\right)^k \varepsilon[-k-1]",
        r"- \varepsilon[-k-1] + 2^{- k} \varepsilon[-k-1]",
    }