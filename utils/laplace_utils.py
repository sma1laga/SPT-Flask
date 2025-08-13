from __future__ import annotations

import logging
import re

import numpy as np
import sympy as sp
from scipy.signal import residue
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
from sympy.printing.latex import LatexPrinter
from sympy.utilities.lambdify import lambdify

logger = logging.getLogger(__name__)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

CONSTANTS = {
    "e": sp.E,
    "pi": sp.pi,
}

SYMBOLS_ALLOWED = {
    "s": sp.Symbol("s", complex=True),
    "j": sp.I,
    "Add": sp.Add,
    "Mul": sp.Mul,
    "Pow": sp.Pow,
    "Integer": sp.Integer,
    "Rational": sp.Rational,
    "Float": sp.Float,
    "factorial": sp.factorial,
}
SYMBOLS_ALLOWED.update(CONSTANTS)

def _int_if_close(val: complex | float | int, tol: float = 1e-8) -> int | complex | float | sp.Integer:
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


def parse_poly(txt: str) -> list[sp.Expr]:
    """Safely parse `txt` into a coefficient vector containing sympy expressions."""
    txt = txt.strip()
    if len(txt) > 200:
        raise ValueError("Input too long")
    if txt.startswith("["):
        items = txt.strip("[]")
        parts = [p.strip() for p in items.split(",") if p.strip()]
        coeffs = [
            parse_expr(
                p,
                evaluate=False,
                local_dict=SYMBOLS_ALLOWED,
                global_dict={},
                transformations=_TRANSFORMS,
            )
            for p in parts
        ]
        return coeffs

    expr = parse_expr(
        txt,
        evaluate=False,
        local_dict=SYMBOLS_ALLOWED,
        global_dict={},
        transformations=_TRANSFORMS,
    )
    poly = expr.as_poly(sp.symbols("s", complex=True))
    return poly.all_coeffs()

def _expr2float(expr, th:int=20):
    """Convert sympy numerical expressions to float if str(expr) is longer than threshold."""
    if len(str(expr)) > th:
        return _int_if_close(complex(expr))
    return expr

def coeffs_to_poly(coeffs, symbol: sp.Symbol=None) -> sp.Expr:
    """Convert coefficient vector `coeffs` to a SymPy polynomial in `s` if symbol is not specified."""
    s = symbol or sp.symbols("s", complex=True)
    # check if exact coefficients can be used
    coeffs = [sp.nsimplify(sp.sympify(_int_if_close(c)), CONSTANTS.values()) for c in coeffs]
    coeffs = [_expr2float(c) for c in coeffs]
    deg = len(coeffs) - 1
    return sum(sp.sympify(coeffs[k]) * s ** (deg - k) for k in range(len(coeffs)))

def factor_poly(expr: sp.Expr, symbol: sp.Symbol=None) -> sp.Expr:
    """Factorize a polynomial expression. Represent roots as floats if str(root_expression)>th_float."""
    s = symbol or sp.symbols("s", complex=True)
    roots = sp.roots(expr, s)
    c0 = expr.as_poly(s).LC()
    expr_factorized = c0 * sp.prod([(s - _expr2float(r)) ** m for r, m in roots.items()])
    return expr_factorized

def factor_rational(num_expr: sp.Expr, den_expr: sp.Expr, symbol: sp.Symbol=None) -> sp.Expr:
    """Factor a rational expression."""
    s = symbol or sp.symbols("s", complex=True)
    num_fac = factor_poly(num_expr, s)
    den_fac = factor_poly(den_expr, s)
    expr_factorized = num_fac / den_fac
    return expr_factorized

def inverse_laplace_expr(num, den) -> sp.Expr:
    """Symbolic inverse Laplace transform of ``num/den``."""
    t = sp.symbols("t", real=True)
    s = sp.symbols("s", complex=True)

    r, p, kvals = residue(num, den, tol=1e-4)
    expr_pfd = 0
    if len(kvals) > 0:
        expr_pfd = coeffs_to_poly(kvals)
    pi_buf = np.nan
    pole_count = 1
    for ri, pi in zip(r, p):
        if np.isclose(pi, pi_buf):
            pole_count += 1
        else:
            pole_count = 1
        if not np.isclose(ri, 0, atol=1e-8):
            ri_expr = sp.nsimplify(sp.sympify(_int_if_close(ri)), CONSTANTS.values())
            pi_expr = sp.nsimplify(sp.sympify(_int_if_close(pi)), CONSTANTS.values())
            # avoid cumbersome expressions
            ri_expr = _expr2float(ri_expr)
            pi_expr = _expr2float(pi_expr)
            expr_pfd += (ri_expr / (s - pi_expr) ** pole_count)
        pi_buf = pi
    expr = sp.inverse_laplace_transform(
        coeffs_to_poly(num) / coeffs_to_poly(den), s, t
    )
    expr = sp.simplify(expr)
    return expr, expr_pfd

def step_response_expr(num, den) -> tuple[sp.Expr, None]:
    """Symbolic step response of System with `H(s)=num/den`."""
    s = sp.Symbol("s", complex=True)
    t = sp.Symbol("t", real=True)
    Y_s = coeffs_to_poly(num) / coeffs_to_poly(den) / s
    expr = sp.inverse_laplace_transform(Y_s, s, t)
    return sp.simplify(expr), None

class CustomLatexPrinter(LatexPrinter):
    def __init__(self, settings=None):
        settings = settings or {}
        if "imaginary_unit" not in settings:
            settings["imaginary_unit"] = "rj"
        super().__init__(settings)
    
    def _print(self, expr, **kwargs):
        # catch Euler constant
        if expr is sp.E:
            return r"\mathrm{e}"
        # usual printing
        return super()._print(expr, **kwargs)
    
    def _print_Heaviside(self, expr, exp=None):
        pargs = ', '.join(self._print(arg) for arg in expr.pargs)
        tex = r"\varepsilon(%s)" % pargs
        if exp:
            tex = r"(%s)^{%s}" % (tex, exp)
        return tex

    def _print_Exp1(self, expr, exp=None):
        return r"\mathrm{e}"
    
    def _print_ExpBase(self, expr, exp=None):
        tex = r"\mathrm{e}^{%s}" % self._print(expr.args[0])
        return self._do_exponent(tex, exp)

def pretty_latex(expr: sp.Expr, simplify_expr: bool=True) -> str:
    if simplify_expr:
        expr = sp.simplify(expr)
    latex = CustomLatexPrinter().doprint(expr)

    # Collapse repeated factors and powers of \varepsilon(t)
    latex = re.sub(r"(\\left)?\((\\varepsilon\(t\))(\\right)?\)\^(\{)?(\d+)(\})?", r"\2", latex)
    latex = re.sub(r"(\\varepsilon\(t\)\s*)+", r"\1", latex)

    # Remove stray \cdot before \varepsilon(t)
    latex = latex.replace(r"\cdot \\varepsilon(t)", r"\varepsilon(t)")
    def mround(match):
        return "{:.4f}".format(float(match.group()))
    latex = re.sub(r"(\d+\.\d{5,})", mround, latex)

    return latex

def eval_expression(expr: sp.Expr, in_vals: np.ndarray, in_symbol: sp.Symbol) -> np.ndarray:
    """Evaluate a SymPy expression with given input values."""
    # match anything like DiracDelta(...)
    expr_wo_dirac = re.sub(r"DiracDelta\(([^)]+)(\))*\)(\s+|$){1}", "0 ", str(expr))
    expr_wo_dirac = sp.parse_expr(expr_wo_dirac)
    f = lambdify(in_symbol, expr_wo_dirac)
    f_eval = f(in_vals)
    if not isinstance(f_eval, np.ndarray):
        f_eval = f_eval * np.ones(len(in_vals), dtype=type(f_eval))
    return f_eval
