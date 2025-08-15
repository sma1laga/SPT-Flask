import numpy as np
import sympy as sp

CONSTANTS = { # constants allowed for simplification
    "e": sp.E,
    "pi": sp.pi,
}


def int_if_close(val: complex | float | int, tol: float = 1e-8) -> int | complex | float | sp.Integer:
    """Return an int/SymPy Integer if ``val`` is within ``tol`` of an integer."""
    if isinstance(val, complex): # clip real or imaginary part to numerical errors
        if abs(val.imag) < tol:
            val = val.real
        elif abs(val.real) < tol:
            val = val.imag * 1j
        else:
            return val
    if isinstance(val, (float, np.floating, sp.Float)):
        if abs(val - round(val)) < tol:
            return int(round(val))
    return val

def rm_keys(d: dict, keys: list) -> dict:
    """Remove keys from a dictionary."""
    if not isinstance(keys, list):
        keys = [keys]
    return {k: v for k, v in d.items() if k not in keys}

def expr2float(expr, th:int=20, complex_polar:bool=False):
    """Convert sympy numerical expressions to float if str(expr) is longer than threshold."""
    if len(str(expr)) > th:
        if complex_polar and (sp.im(expr) != 0):
            return int_if_close(float(sp.Abs(expr))) * sp.exp(sp.I * int_if_close(float(sp.arg(expr))))
        else:
            return int_if_close(complex(expr))
    return expr

def render_number(expr, th_float=20, complex_polar:bool=False):
    """Render a SymPy expression as an exact representation if it is short enough."""
    if isinstance(expr, sp.Expr):
        expr = sp.parse_expr(str(expr).replace("DiracDelta(0)", "1"))
    expr_out = sp.nsimplify(sp.sympify(int_if_close(expr)), CONSTANTS.values()) # render exact number
    if complex_polar and (sp.im(expr_out) != 0):
        expr_out = sp.Abs(expr_out) * sp.exp(sp.I * sp.arg(expr_out)) # convert to polar form
    expr_out = expr2float(expr_out, th_float, complex_polar) # decide if float or exact
    return expr_out

def coeffs_to_poly(coeffs, symbol: sp.Symbol) -> sp.Expr:
    """Asc → poly:  [cn-1,cn-2,...,c1,c0]  →  cn-1 x^(n-1) + cn-2 x^(n-2) + ... + c1 x + c0"""
    # check if exact coefficients can be used
    coeffs = [render_number(c) for c in coeffs]
    deg = len(coeffs) - 1
    return sum(
        c * symbol**(deg-i)
        for i, c in enumerate(coeffs)
    )

def factor_poly(expr: sp.Expr, symbol: sp.Symbol) -> sp.Expr:
    """Factorize a polynomial expression. Represent roots as floats if str(root_expression)>th_float."""
    roots = sp.roots(expr, symbol)
    c0 = expr.as_poly(symbol).LC()
    expr_factorized = c0 * sp.prod([(symbol - expr2float(r)) ** m for r, m in roots.items()])
    return expr_factorized

def factor_rational(num_expr: sp.Expr, den_expr: sp.Expr, symbol: sp.Symbol) -> sp.Expr:
    """Factor a rational expression."""
    num_fac = factor_poly(num_expr, symbol)
    den_fac = factor_poly(den_expr, symbol)
    expr_factorized = num_fac / den_fac
    return expr_factorized
