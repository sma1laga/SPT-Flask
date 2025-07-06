from __future__ import annotations

import ast
import logging
from typing import Tuple

import numpy as np
import sympy as sp
from scipy.signal import lti, impulse, residue
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application,
)

logger = logging.getLogger(__name__)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
)


def _int_if_close(val: complex | float | int, tol: float = 1e-12) -> int | complex | float | sp.Integer:
    """Return an int/SymPy Integer if ``val`` is within ``tol`` of an integer."""
    if isinstance(val, complex):
        if abs(val.imag) < tol:
            val = val.real
        else:
            return val
    if isinstance(val, (float, np.floating, sp.Float)):
        if abs(val - round(val)) < tol:
            return int(round(val))
    return val


def parse_poly(txt: str) -> np.ndarray:
    """Safely parse ``txt`` into a coefficient vector."""
    txt = txt.strip()
    if len(txt) > 200:
        raise ValueError("Input too long")
    if txt.startswith("["):
        try:
            coeffs = np.asarray(ast.literal_eval(txt), dtype=complex)
        except Exception:
            items = txt.strip("[]")
            parts = [p.strip() for p in items.split(",") if p.strip()]
            allowed = {
                "s": sp.Symbol("s"),
                "I": sp.I,
                "Add": sp.Add,
                "Mul": sp.Mul,
                "Pow": sp.Pow,
                "Integer": sp.Integer,
                "Rational": sp.Rational,
                "Float": sp.Float,
            }
            vals = [
                complex(
                    parse_expr(
                        p.replace("j", "I"),
                        evaluate=False,
                        local_dict=allowed,
                        global_dict={},
                        transformations=_TRANSFORMS,
                    )
                )
                for p in parts
            ]
            coeffs = np.asarray(vals, dtype=complex)
        if len(coeffs) - 1 > 50:
            raise ValueError("Polynomial degree too high")
        return coeffs

    allowed = {
        "s": sp.Symbol("s"),
        "I": sp.I,
        "Add": sp.Add,
        "Mul": sp.Mul,
        "Pow": sp.Pow,
        "Integer": sp.Integer,
        "Rational": sp.Rational,
        "Float": sp.Float,
    }
    expr = parse_expr(
        txt.replace("j", "I"),
        evaluate=False,
        local_dict=allowed,
        global_dict={},
        transformations=_TRANSFORMS,
    )
    poly = sp.Poly(sp.expand(expr), allowed["s"])
    if poly.degree() > 50:
        raise ValueError("Polynomial degree too high")
    coeffs = poly.all_coeffs()[::-1]
    return np.asarray([complex(c) for c in coeffs], dtype=complex)


def coeffs_to_poly(coeffs: np.ndarray) -> sp.Expr:
    """Convert coefficient vector ``coeffs`` to a SymPy polynomial."""
    s = sp.symbols("s")
    deg = len(coeffs) - 1
    return sum(sp.sympify(_int_if_close(coeffs[k])) * s ** (deg - k) for k in range(len(coeffs)))


def poly_long_division(num: np.ndarray, den: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return quotient and remainder coefficient vectors of ``num/den``."""
    s = sp.symbols("s")
    num_poly = sp.Poly(coeffs_to_poly(num), s)
    den_poly = sp.Poly(coeffs_to_poly(den), s)
    q_poly, r_poly = sp.div(num_poly, den_poly)
    q_coeffs = np.asarray([complex(c) for c in q_poly.all_coeffs()[::-1]], dtype=complex)
    if q_coeffs.size == 0:
        q_coeffs = np.asarray([0], dtype=complex)
    r_coeffs = np.asarray([complex(c) for c in r_poly.all_coeffs()[::-1]], dtype=complex)
    if r_coeffs.size == 0:
        r_coeffs = np.asarray([0], dtype=complex)
    return q_coeffs, r_coeffs


def inverse_laplace_expr(num: np.ndarray, den: np.ndarray, causal: bool = True) -> sp.Expr:
    """Symbolic inverse Laplace transform of ``num/den``."""
    t = sp.symbols("t")
    try:
        r, p, kvals = residue(num, den)
    except Exception:
        logger.exception("Residue failed")
        expr = sp.inverse_laplace_transform(
            coeffs_to_poly(num) / coeffs_to_poly(den), sp.symbols("s"), t
        )
        if causal:
            expr *= sp.Heaviside(t)
        return sp.simplify(expr)

    expr = 0
    kvals_rev = kvals[::-1]
    for order, ki in enumerate(kvals_rev):
        if not np.isclose(ki, 0):
            delta = sp.DiracDelta(t)
            if order > 0:
                delta = delta.diff(t, order)
            expr += sp.sympify(_int_if_close(ki)) * delta

    for ri, pi in zip(r, p):
        if not np.isclose(ri, 0):
            term = sp.sympify(_int_if_close(ri)) * sp.exp(pi * t)
            if causal:
                term *= sp.Heaviside(t)
            expr += term

    return sp.simplify(expr)


def impulse_response(num: np.ndarray, den: np.ndarray, N: int = 10, dt: float = 1.0) -> list:
    """Sampled impulse response using SciPy's ``impulse``."""
    sys = lti(num, den)
    t_vals = np.arange(0, N * dt, dt, dtype=float)
    t, y = impulse(sys, T=t_vals)
    seq = []
    for v in y:
        c = complex(v)
        if abs(c.imag) < 1e-12:
            c = float(c.real)
            if c.is_integer():
                c = int(c)
        c = _int_if_close(c)
        seq.append(c)
    return seq