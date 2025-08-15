from __future__ import annotations

import logging
import re

import numpy as np
import sympy as sp
from scipy.signal import residue
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application, convert_xor
)
from sympy.printing.latex import LatexPrinter
from sympy.utilities.lambdify import lambdify

import utils.sympy_utils as sp_utils

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
    # trig functions
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
}
SYMBOLS_ALLOWED.update(CONSTANTS)

def parse_input(txt: str) -> sp.Expr:
    """Safely parse `txt` into a SymPy expression."""
    txt = txt.strip()
    if len(txt) > 100:
        raise ValueError("Input too long")
    if txt.startswith("["): # polynomial from coefficient list
        items = txt.strip("[]")
        parts = [p.strip() for p in items.split(",") if p.strip()]
        coeffs = [
            sp.parse_expr(
                p,
                evaluate=False,
                local_dict=sp_utils.rm_keys(SYMBOLS_ALLOWED, ["s"]), # without s
                global_dict={},
                transformations=_TRANSFORMS,
            )
            for p in parts
        ]
        return sp_utils.coeffs_to_poly(coeffs, sp.Symbol("s", complex=True))

    # any other expression
    expr = sp.parse_expr(
        txt,
        evaluate=False,
        local_dict=SYMBOLS_ALLOWED,
        global_dict={},
        transformations=_TRANSFORMS,
    )
    return expr

def inverse_laplace_expr(num, den) -> sp.Expr:
    """Symbolic inverse Laplace transform of ``num/den``."""
    t = sp.symbols("t", real=True)
    s = sp.symbols("s", complex=True)

    r, p, kvals = residue(num, den, tol=1e-4)
    expr_pfd = 0
    if len(kvals) > 0:
        expr_pfd = sp_utils.coeffs_to_poly(kvals, s)
    pi_buf = np.nan
    pole_count = 1
    for ri, pi in zip(r, p):
        if np.isclose(pi, pi_buf):
            pole_count += 1
        else:
            pole_count = 1
        if not np.isclose(ri, 0, atol=1e-8):
            ri_expr = sp_utils.render_number(ri)
            pi_expr = sp_utils.render_number(pi)
            expr_pfd += (ri_expr / (s - pi_expr) ** pole_count)
        pi_buf = pi
    expr = sp.inverse_laplace_transform(
        sp_utils.coeffs_to_poly(num, s) / sp_utils.coeffs_to_poly(den, s), s, t
    )
    expr = sp.simplify(expr)
    return expr, expr_pfd

def step_response_expr(num, den) -> tuple[sp.Expr, None]:
    """Symbolic step response of System with `H(s)=num/den`."""
    s = sp.Symbol("s", complex=True)
    t = sp.Symbol("t", real=True)
    Y_s = sp_utils.coeffs_to_poly(num, s) / sp_utils.coeffs_to_poly(den, s) / s
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

    def doprint(self, expr, **kwargs):
        """Override doprint to handle DiracDelta and Heaviside functions."""
        if kwargs.get("simplify_expr", True):
            expr = sp.simplify(expr)
        latex = super().doprint(expr)

        # Collapse repeated factors and powers of \varepsilon(t)
        latex = re.sub(r"(\\left)?\((\\varepsilon\(t\))(\\right)?\)\^(\{)?(\d+)(\})?", r"\2", latex)
        latex = re.sub(r"(\\varepsilon\(t\)\s*)+", r"\1", latex)

        # Remove stray \cdot before \varepsilon(t)
        latex = latex.replace(r"\cdot \\varepsilon(t)", r"\varepsilon(t)")
        def mround(match):
            return str(np.round(float(match.group()), 4))
        latex = re.sub(r"(\d+\.\d{5,})", mround, latex)
        return latex

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
