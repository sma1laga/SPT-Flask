"""Bode / Nyquist plotter blueprint.

The frequency response is derived from the poles and zeros of ``H(s) = num(s)/den(s)``:

* the exact magnitude comes from a direct evaluation of the polynomials (with a
  root-product fallback where that overflows or hits a pole on the imaginary axis),
* the exact phase is assembled factor by factor so that the 360-degree branch is fixed
  by the transfer function itself (negative gains, right-half-plane roots, integrators
  and undamped pairs all land on the textbook branch, and ``np.unwrap`` is not needed),
* the straight-line (asymptotic) approximations use the classical Bode rules: corner at
  |root|, magnitude slopes of +-20 dB/decade per root, phase ramps of +-45 deg/decade
  between 0.1*|root| and 10*|root| that start at the low-frequency phase of the factor.
"""

from __future__ import annotations

import ast
import csv
import re
from ast import literal_eval
from dataclasses import dataclass
from io import BytesIO, StringIO
from itertools import zip_longest
from typing import Optional

import control
import numpy as np
import sympy as sp
from flask import Blueprint, Response, render_template, request
from markupsafe import escape
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from utils.eval_helpers import validate_expression_safety

_COMPLEX_COEFF_TOL = 1e-9
# Roots with |r| below this are treated as roots at the origin; roots with |Re r| below
# this fraction of |r| are treated as lying on the imaginary axis.  np.roots scatters an
# m-fold root by roughly eps**(1/m), so _snap_roots first moves clusters of nearly equal
# roots whose mean lies within that scatter of the axis (or origin) exactly onto it.
_ROOT_TOL = 1e-9
# Relative distance below which roots are considered one cluster (a repeated root).
_CLUSTER_RTOL = 1e-3
# Relative tolerance for merging corner frequencies of repeated roots (np.roots scatters
# a quadruple root by roughly 2e-4).
_CORNER_MERGE_RTOL = 1e-3
_MAX_DEGREE = 100
_MAX_EXPONENT = 1000
_MAX_INPUT_LENGTH = 1000
_N_FREQ = 500

bode_plot_bp = Blueprint('bode_plot', __name__, template_folder='templates')

_S = sp.symbols('s', complex=True)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

_ALLOWED_NAMES = {'s', 'S', 'j', 'J', 'I', 'pi', 'E'}
_ALLOWED_CALLS = {'sqrt', 'exp'}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

_PARSE_LOCALS = {
    's': _S,
    'S': _S,
    'j': sp.I,
    'J': sp.I,
    'I': sp.I,
    'pi': sp.pi,
    'E': sp.E,
    'sqrt': sp.sqrt,
    'exp': sp.exp,
}
# Only the names the standard sympy transformations emit; no builtins, no ``from sympy
# import *``.  Everything the user may reference is in _PARSE_LOCALS.
_PARSE_GLOBALS = {
    'Symbol': sp.Symbol,
    'Integer': sp.Integer,
    'Float': sp.Float,
    'Rational': sp.Rational,
    'I': sp.I,
    'Mul': sp.Mul,
    'Add': sp.Add,
    'Pow': sp.Pow,
}

_INPUT_HELP = (
    "Use a coefficient list such as [1, 2, 3] or a factorized expression such as "
    "(s+3+2*j)(s+3-2*j); only numbers, s, j, + - * / ^ and parentheses are allowed."
)


@dataclass
class ParsedPolynomial:
    """A user polynomial: float/complex coefficients (highest power first), its roots,
    the LaTeX/plain-text forms used for display and the sympy expression (if any)."""

    coeffs: list
    roots: np.ndarray
    latex: str
    text: str
    display: Optional[str]
    expr: Optional[sp.Expr]

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1


def _strip_leading_zeros(coeffs):
    coeffs = list(coeffs)
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs.pop(0)
    return coeffs


def _normalize_coefficients(coeffs, tol=_COMPLEX_COEFF_TOL):
    """Normalize coefficient types and strip negligible imaginary parts."""
    normalized = []
    has_complex_part = False

    for coeff in coeffs:
        coeff_complex = complex(coeff)
        if abs(coeff_complex.imag) <= tol:
            normalized.append(float(coeff_complex.real))
        else:
            normalized.append(coeff_complex)
            has_complex_part = True

    return normalized, has_complex_part


def _check_finite(coeffs):
    if not all(np.isfinite(c) for c in coeffs):
        raise ValueError("Coefficients must be finite numbers (no inf or nan).")


_NUMBER_RE = r'\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?'


def _normalize_expression_text(expr_str: str) -> str:
    """Turn the user's notation into plain Python syntax (``^`` -> ``**``, implicit products)."""
    text = expr_str.strip().replace('^', '**')
    # number followed by a letter or parenthesis: 2s, 2(s+1), 1e3s  (e/E are excluded from
    # the lookahead so that the exponent of 1e5 is never split off by backtracking)
    text = re.sub(rf'({_NUMBER_RE})\s*(?=[A-DF-Za-df-z(])', r'\1*', text)
    # closing parenthesis followed by a letter, number or parenthesis: (s+1)(s+2), (s+1)s
    text = re.sub(r'\)\s*(?=[A-Za-z0-9(.])', ')*', text)
    # single-letter variable followed by a parenthesis: s(s+1)
    text = re.sub(r'(?<![A-Za-z_])([sSjJI])\s*\(', r'\1*(', text)
    return text


def _check_expression_ast(text: str) -> None:
    """Reject anything that is not an arithmetic expression in s (defense against code
    execution through sympy's parser, which evaluates Python code)."""
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as exc:
        raise ValueError(f"Could not read the expression near position {exc.offset}. {_INPUT_HELP}")

    def literal_number(node):
        """Value of a (possibly signed) numeric literal, else None."""
        sign = 1
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
            if isinstance(node.op, ast.USub):
                sign = -sign
            node = node.operand
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return sign * node.value
        return None

    def check(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            check(node.left)
            exponent = literal_number(node.right)
            if exponent is None:
                raise ValueError("The exponent must be a plain number, e.g. (s+1)^3.")
            if abs(exponent) > _MAX_EXPONENT:
                raise ValueError(f"Exponent is too large. Please use an absolute value <= {_MAX_EXPONENT}.")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            check(node.left)
            check(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
            check(node.operand)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)) \
                and not isinstance(node.value, bool):
            pass
        elif isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES:
                raise ValueError(f"Unknown symbol '{node.id}'. {_INPUT_HELP}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _ALLOWED_CALLS and not node.keywords:
            for arg in node.args:
                check(arg)
        else:
            raise ValueError(f"Unsupported expression element. {_INPUT_HELP}")

    check(tree.body)


def _parse_sympy(text: str, evaluate: bool) -> sp.Expr:
    return parse_expr(
        text,
        local_dict=_PARSE_LOCALS,
        global_dict=dict(_PARSE_GLOBALS),
        transformations=standard_transformations,
        evaluate=evaluate,
    )


def _sympy_to_complex(value) -> complex:
    try:
        return complex(sp.N(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Coefficients must be finite numbers (no inf or nan).") from exc


def _degree_bound(expr: sp.Expr) -> int:
    """Upper bound of the polynomial degree in s of an unexpanded expression.

    Computed structurally (no expansion), so that the degree cap can be enforced before the
    potentially very expensive sp.expand."""
    if expr == _S:
        return 1
    if not expr.has(_S):
        return 0
    if expr.is_Add:
        return max(_degree_bound(arg) for arg in expr.args)
    if expr.is_Mul:
        return sum(_degree_bound(arg) for arg in expr.args)
    if expr.is_Pow:
        base, exponent = expr.args
        if exponent.is_Integer and exponent >= 0:
            return int(exponent) * _degree_bound(base)
        raise ValueError(
            "The expression must be a polynomial in s (no division by s, fractional powers, exp(s) etc.). "
            "Enter numerator and denominator separately."
        )
    raise ValueError(
        "The expression must be a polynomial in s (no division by s, fractional powers, exp(s) etc.). "
        "Enter numerator and denominator separately."
    )


def _snap_roots(roots) -> np.ndarray:
    """Move clusters of nearly equal roots that sit within numerical scatter of the
    imaginary axis (or the origin) exactly onto it.

    np.roots returns an m-fold root as a cluster of m values scattered by about
    eps**(1/m); half of a repeated imaginary-axis pair would otherwise land in the right
    half-plane and be reported as unstable."""
    roots = np.asarray(roots, dtype=complex).copy()
    n = roots.size
    if n == 0:
        return roots
    eps = np.finfo(float).eps
    assigned = np.zeros(n, dtype=bool)
    for i in range(n):
        if assigned[i]:
            continue
        scale = max(1.0, abs(roots[i]))
        members = np.where(~assigned & (np.abs(roots - roots[i]) <= _CLUSTER_RTOL * scale))[0]
        assigned[members] = True
        m = members.size
        mean = roots[members].mean()
        scale = max(1.0, abs(mean))
        tol = max(_ROOT_TOL, 10.0 * eps ** (1.0 / m)) * scale
        if abs(mean) <= tol:
            roots[members] = 0.0
        elif abs(mean.real) <= tol:
            roots[members] = 1j * roots[members].imag
    return roots


def _polynomial_roots(coeffs) -> np.ndarray:
    """Roots of a polynomial given by float/complex coefficients (highest power first).

    Exact sympy roots are used for small polynomials with 'nice' coefficients so that
    repeated roots such as (s+1)^3 come out clean; otherwise np.roots is used."""
    coeffs = _strip_leading_zeros(coeffs)
    degree = len(coeffs) - 1
    if degree <= 0:
        return np.array([], dtype=complex)
    if degree <= 8:
        try:
            exact = [sp.nsimplify(c, rational=True) if not isinstance(c, complex)
                     else sp.nsimplify(c.real, rational=True) + sp.I * sp.nsimplify(c.imag, rational=True)
                     for c in coeffs]
            poly = sp.Poly(exact, _S)
            found = sp.roots(poly, multiple=True)
            if len(found) == degree:
                return _snap_roots([complex(sp.N(r, 17)) for r in found])
        except Exception:
            pass
    with np.errstate(all='ignore'):
        return _snap_roots(np.roots(coeffs))


def _checked_roots(roots) -> np.ndarray:
    roots = np.asarray(roots, dtype=complex)
    if not np.all(np.isfinite(roots)):
        raise ValueError(
            "The coefficients span too many orders of magnitude; the roots cannot be represented."
        )
    return roots


def _roots_from_factors(expr: sp.Expr, degree: int) -> Optional[np.ndarray]:
    """Roots of a product expression, factor by factor, so multiplicities stay exact."""
    roots = []
    for factor in sp.Mul.make_args(expr):
        base, power = factor.as_base_exp()
        if not (power.is_Integer and power > 0):
            return None
        try:
            poly = sp.Poly(sp.expand(base), _S)
        except sp.PolynomialError:
            return None
        if poly.degree() <= 0:
            continue
        base_coeffs, _ = _normalize_coefficients([_sympy_to_complex(c) for c in poly.all_coeffs()])
        base_roots = _polynomial_roots(base_coeffs)
        roots.extend(list(base_roots) * int(power))
    if len(roots) != degree:
        return None
    return np.asarray(roots, dtype=complex)


def _latex_from_expression(text: str, expr_evaluated: sp.Expr) -> str:
    """Render the factorized input close to how the user typed it."""
    settings = dict(imaginary_unit='j', min=-3, max=4)
    rendered = None
    try:
        unevaluated = _parse_sympy(text, evaluate=False)
        candidate = sp.latex(unevaluated, order='none', **settings)
        if candidate and r'\text' not in candidate:
            rendered = candidate
    except Exception:
        rendered = None
    if rendered is None:
        rendered = sp.latex(expr_evaluated, **settings)
    # 1.0 \cdot 10^{5}  ->  10^{5}
    return re.sub(r'(?<![\d.])1\.0 \\cdot 10\^', r'10^', rendered)


def _text_from_expression(expr: sp.Expr) -> str:
    return sp.sstr(expr).replace('**', '^').replace('*I', 'j').replace('I*', 'j').replace('I', 'j')


def parse_polynomial(expr_str: str) -> ParsedPolynomial:
    """Parse a coefficient list ``[1, 2]`` or a factorized expression ``(s+1)(s+2)``."""
    if expr_str is None or not str(expr_str).strip():
        raise ValueError(f"Please enter a polynomial. {_INPUT_HELP}")
    expr_str = str(expr_str).strip()
    if len(expr_str) > _MAX_INPUT_LENGTH:
        raise ValueError(f"The input is too long (more than {_MAX_INPUT_LENGTH} characters).")

    if expr_str.startswith('['):
        try:
            coeffs = literal_eval(expr_str)
            if not (isinstance(coeffs, list) and coeffs
                    and all(isinstance(x, (int, float, complex)) and not isinstance(x, bool) for x in coeffs)):
                raise ValueError
        except Exception:
            raise ValueError("Coefficients must be entered as a non-empty list of numbers, e.g. [1, 1+10j, 1-10j].")
        coeffs, _ = _normalize_coefficients(coeffs)
        _check_finite(coeffs)
        coeffs = _strip_leading_zeros(coeffs)
        if len(coeffs) - 1 > _MAX_DEGREE:
            raise ValueError(f"Polynomial degree must not exceed {_MAX_DEGREE}.")
        if all(c == 0 for c in coeffs):
            raise ValueError("The polynomial is identically zero.")
        return ParsedPolynomial(
            coeffs=coeffs,
            roots=_checked_roots(_polynomial_roots(coeffs)),
            latex=format_polynomial(coeffs),
            text=_polynomial_text(coeffs),
            display=None,
            expr=None,
        )

    text = _normalize_expression_text(expr_str)
    _check_expression_ast(text)
    validate_expression_safety(text)
    try:
        expr = _parse_sympy(text, evaluate=True)
    except Exception as exc:
        raise ValueError(f"Could not parse the expression: {exc}. {_INPUT_HELP}")
    if expr.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        raise ValueError("Coefficients must be finite numbers (no inf or nan).")
    # Bound the degree structurally before expanding: expanding (s^5+...+1)^1000 would
    # take minutes, the bound is instant.
    if _degree_bound(expr) > _MAX_DEGREE:
        raise ValueError(f"Polynomial degree must not exceed {_MAX_DEGREE}.")
    try:
        poly = sp.Poly(sp.expand(expr), _S)
    except sp.PolynomialError:
        raise ValueError(
            "The expression must be a polynomial in s (no division by s, fractional powers, exp(s) etc.). "
            "Enter numerator and denominator separately."
        )
    if poly.degree() > _MAX_DEGREE:
        raise ValueError(f"Polynomial degree must not exceed {_MAX_DEGREE}.")
    coeffs = [_sympy_to_complex(c) for c in poly.all_coeffs()]
    coeffs, _ = _normalize_coefficients(coeffs)
    _check_finite(coeffs)
    coeffs = _strip_leading_zeros(coeffs)
    if all(c == 0 for c in coeffs):
        raise ValueError("The polynomial is identically zero.")
    roots = _roots_from_factors(expr, len(coeffs) - 1)
    if roots is None:
        roots = _polynomial_roots(coeffs)
    roots = _checked_roots(roots)
    return ParsedPolynomial(
        coeffs=coeffs,
        roots=roots,
        latex=_latex_from_expression(text, expr),
        text=_text_from_expression(expr),
        display=expr_str,
        expr=expr,
    )


def parse_poly_input(expr_str):
    """Backwards-compatible wrapper: returns ``(coefficients, display_string)``."""
    parsed = parse_polynomial(expr_str)
    return parsed.coeffs, parsed.display


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_real_latex(value: float, zero_tol: float = 1e-12) -> str:
    """Return a LaTeX-friendly string for a real number."""
    if value is None or np.isnan(value):
        return r"\mathrm{NaN}"
    if np.isinf(value):
        return r"-\infty" if value < 0 else r"\infty"
    if abs(value) <= zero_tol:
        return "0"

    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    mantissa_str, exponent_str = f"{magnitude:.3e}".split('e')
    exponent = int(exponent_str)
    scaled = float(mantissa_str)

    if exponent >= 3 or exponent <= -3:
        # Avoid printing 1.0 x 10^n when the mantissa is effectively one.
        if abs(scaled - 1) < 1e-9:
            mantissa_part = ""
        else:
            mantissa_part = f"{scaled:.3g}\\times"
        return f"{sign}{mantissa_part}10^{{{exponent}}}"

    return f"{sign}{magnitude:.6g}"


def _format_complex_latex_body(val: complex) -> str:
    """LaTeX body (without delimiters) of a complex number with a relative tolerance."""
    real_part, imag_part = val.real, val.imag
    if not (np.isfinite(real_part) and np.isfinite(imag_part)):
        return r"\infty"
    scale = max(abs(real_part), abs(imag_part))
    if scale < 1e-12:
        return "0"

    real_str = None
    if abs(real_part) >= 1e-9 * scale:
        real_str = _format_real_latex(real_part)

    imag_str = None
    if abs(imag_part) >= 1e-9 * scale:
        imag_abs = _format_real_latex(abs(imag_part))
        if imag_abs in {"0", "1"}:
            imag_abs = ""
        imag_unit = "\\mathrm{j}"
        if real_str is None:
            sign = "-" if imag_part < 0 else ""
            imag_str = f"{sign}{imag_abs}{imag_unit}" if imag_abs else f"{sign}{imag_unit}"
        else:
            sign = "-" if imag_part < 0 else "+"
            imag_str = f"{sign} {imag_abs}{imag_unit}" if imag_abs else f"{sign} {imag_unit}"

    if real_str is None and imag_str is None:
        return "0"
    if real_str is None:
        return imag_str
    if imag_str is None:
        return real_str
    return f"{real_str} {imag_str}"


def format_complex(val: complex) -> str:
    """Format a complex number as an inline LaTeX string."""
    return rf"\({_format_complex_latex_body(complex(val))}\)"


def format_polynomial(coeffs, var="s"):
    """Format a list of coefficients into a LaTeX polynomial string."""
    coeffs = _strip_leading_zeros(list(coeffs))
    terms = []
    n = len(coeffs)
    for i, coeff in enumerate(coeffs):
        power = n - i - 1
        if coeff == 0:
            continue
        if isinstance(coeff, complex):
            coeff_str = f"({_format_complex_latex_body(coeff)})"
            negative = False
        else:
            negative = coeff < 0
            magnitude = _format_real_latex(abs(coeff), zero_tol=0.0)
            if power > 0 and magnitude == "1":
                coeff_str = ""
            else:
                coeff_str = magnitude
        if power == 0:
            term = coeff_str or "1"
        elif power == 1:
            term = f"{coeff_str}{var}"
        else:
            term = f"{coeff_str}{var}^{{{power}}}"
        terms.append((negative, term))
    if not terms:
        return "0"
    negative, term = terms[0]
    poly_str = ("-" if negative else "") + term
    for negative, term in terms[1:]:
        poly_str += (" - " if negative else " + ") + term
    return poly_str


def _polynomial_text(coeffs, var="s") -> str:
    """Plain-text polynomial (for the PNG export header)."""
    coeffs = _strip_leading_zeros(list(coeffs))
    terms = []
    n = len(coeffs)
    for i, coeff in enumerate(coeffs):
        power = n - i - 1
        if coeff == 0:
            continue
        if isinstance(coeff, complex):
            coeff_str = f"({coeff.real:g}{coeff.imag:+g}j)"
            negative = False
        else:
            negative = coeff < 0
            magnitude = f"{abs(coeff):g}"
            coeff_str = "" if (power > 0 and magnitude == "1") else magnitude
        if power == 0:
            term = coeff_str or "1"
        elif power == 1:
            term = f"{coeff_str}{var}"
        else:
            term = f"{coeff_str}{var}^{power}"
        terms.append((negative, term))
    if not terms:
        return "0"
    negative, term = terms[0]
    text = ("-" if negative else "") + term
    for negative, term in terms[1:]:
        text += (" - " if negative else " + ") + term
    return text


def _format_real_text(value: float, unit: Optional[str] = None) -> str:
    """Format a real value into a concise human-readable string."""
    if value is None or not np.isfinite(value):
        return "—"

    abs_val = abs(value)
    if abs_val != 0 and (abs_val >= 1e4 or abs_val <= 1e-3):
        formatted = f"{value:.3e}"
    else:
        formatted = f"{value:.4f}".rstrip('0').rstrip('.')

    return f"{formatted} {unit}" if unit else formatted


def _format_latex_real(value: float, unit: Optional[str] = None) -> str:
    """Return a LaTeX inline string for a real number with an optional unit."""
    if value is None or not np.isfinite(value):
        return "—"
    content = _format_real_latex(value)
    if unit:
        return rf"\({content}\;{unit}\)"
    return rf"\({content}\)"


def _analysis_item(title: str, value: str, detail: str, level: str = "info") -> dict:
    return {"title": title, "value": value, "detail": detail, "level": level}


# ---------------------------------------------------------------------------
# Root classification and per-factor response
# ---------------------------------------------------------------------------

def _root_kind(root) -> str:
    """Classify a root as 'origin', 'axis' (imaginary axis), 'lhp' or 'rhp'."""
    root = complex(root)
    mag = abs(root)
    if mag <= _ROOT_TOL:
        return 'origin'
    if abs(root.real) <= _ROOT_TOL * max(1.0, mag):
        return 'axis'
    return 'lhp' if root.real < 0 else 'rhp'


def _factor_phase_exact(root, w):
    """Continuous phase (deg) of the factor (jw - root) over w > 0.

    The branch is continuous in w and tends to +90 deg as w -> inf (the factor behaves
    like jw at high frequency), which is the classical Bode convention."""
    root = complex(root)
    a, b = root.real, root.imag
    kind = _root_kind(root)
    if kind == 'origin':
        return np.full_like(w, 90.0, dtype=float)
    if kind == 'axis':
        return np.where(w >= b, 90.0, -90.0)
    if kind == 'lhp':
        return np.degrees(np.arctan2(w - b, -a))
    return 180.0 - np.degrees(np.arctan2(w - b, a))


def _factor_phase_start(root) -> float:
    """Low-frequency limit (w -> 0+) of _factor_phase_exact on the same branch."""
    root = complex(root)
    a, b = root.real, root.imag
    kind = _root_kind(root)
    if kind == 'origin':
        return 90.0
    if kind == 'axis':
        return -90.0 if b > 0 else 90.0
    if kind == 'lhp':
        return float(np.degrees(np.arctan2(-b, -a)))
    return float(180.0 + np.degrees(np.arctan2(b, a)))


def _factor_phase_straight(root, w):
    """Straight-line phase approximation of the factor (jw - root): a ramp (linear in
    log w) from the low-frequency phase to +90 deg between |root|/10 and 10*|root|."""
    kind = _root_kind(root)
    if kind in ('origin', 'axis'):
        return _factor_phase_exact(root, w)
    corner = abs(complex(root))
    start = _factor_phase_start(root)
    t = np.clip((np.log10(w / corner) + 1.0) / 2.0, 0.0, 1.0)
    return start + (90.0 - start) * t


def _factor_magnitude_exact_db(root, w):
    with np.errstate(divide='ignore'):
        return 20.0 * np.log10(np.abs(1j * w - complex(root)))


def _factor_magnitude_straight_db(root, w):
    """Asymptotic |jw - root| in dB: |root| below the corner, w above it."""
    if _root_kind(root) == 'origin':
        return 20.0 * np.log10(w)
    corner = abs(complex(root))
    return 20.0 * np.log10(corner) + 20.0 * np.maximum(0.0, np.log10(w / corner))


def _wrap_deg(x):
    """Wrap angles (deg) into (-180, 180]."""
    return -((-np.asarray(x, dtype=float) + 180.0) % 360.0) + 180.0


def _low_frequency_phase_slope(zeros, poles) -> float:
    """d(phase)/dw of H(jw) at w -> 0+ (rad per rad/s), from the factor decomposition.

    Each factor (jw - r) contributes -Re(r)/|r|^2; zeros add, poles subtract.  Roots at
    the origin and on the imaginary axis contribute nothing (constant, or a step)."""
    slope = 0.0
    for root, sign in [(z, 1.0) for z in zeros] + [(p, -1.0) for p in poles]:
        root = complex(root)
        if _root_kind(root) in ('lhp', 'rhp'):
            slope += sign * (-root.real) / abs(root) ** 2
    return slope


def _corner_frequencies(poles, zeros):
    """Sorted unique corner frequencies |root| of the finite, non-origin roots."""
    raw_values = []
    for root in np.concatenate([np.asarray(poles, dtype=complex), np.asarray(zeros, dtype=complex)]):
        if np.isfinite(root) and _root_kind(root) != 'origin':
            raw_values.append(float(abs(root)))

    if not raw_values:
        return []

    raw_values.sort()
    deduped = []
    for value in raw_values:
        if not deduped or not np.isclose(value, deduped[-1], rtol=_CORNER_MERGE_RTOL, atol=0.0):
            deduped.append(value)
    return deduped


def _evaluate_transfer(num, den, s_values):
    """Safely evaluate ``num/den`` at the complex points ``s_values`` (NaN where den = 0)."""
    s_array = np.atleast_1d(np.asarray(s_values, dtype=complex))
    with np.errstate(all='ignore'):
        num_vals = np.polyval(num, s_array)
        den_vals = np.polyval(den, s_array)
        response = np.full(s_array.shape, complex(np.nan, np.nan), dtype=complex)
        valid = np.isfinite(den_vals) & np.isfinite(num_vals) & (den_vals != 0)
        response[valid] = num_vals[valid] / den_vals[valid]
    if np.isscalar(s_values):
        return response.item()
    return response


def _make_control_transfer_function(num, den):
    """Build a python-control transfer function only for effectively real polynomials."""
    real_num, num_has_complex = _normalize_coefficients(num)
    real_den, den_has_complex = _normalize_coefficients(den)
    if num_has_complex or den_has_complex:
        return None
    try:
        return control.TransferFunction(real_num, real_den)
    except Exception:
        return None


def _frequency_response(num, den, w, zeros=None, poles=None):
    """Exact and straight-line magnitude (dB) and phase (deg) of num/den at w (> 0)."""
    num = _strip_leading_zeros(num)
    den = _strip_leading_zeros(den)
    w = np.asarray(w, dtype=float)
    zeros = np.asarray(_polynomial_roots(num) if zeros is None else zeros, dtype=complex)
    poles = np.asarray(_polynomial_roots(den) if poles is None else poles, dtype=complex)
    gain = complex(num[0]) / complex(den[0])

    # --- exact magnitude: direct evaluation, root product where that is not finite ---
    H = _evaluate_transfer(num, den, 1j * w)
    with np.errstate(divide='ignore', invalid='ignore'):
        magnitude_db = 20.0 * np.log10(np.abs(H))
    product_db = np.full_like(w, 20.0 * np.log10(abs(gain)), dtype=float)
    for z in zeros:
        product_db += _factor_magnitude_exact_db(z, w)
    for p in poles:
        product_db -= _factor_magnitude_exact_db(p, w)
    finite = np.isfinite(magnitude_db)
    magnitude_db = np.where(finite, magnitude_db, product_db)

    # --- phase: branch from the factor decomposition, values from the evaluation ---
    gain_phase = float(np.degrees(np.angle(gain)))
    exact_factor = np.full_like(w, gain_phase, dtype=float)
    straight = np.full_like(w, gain_phase, dtype=float)
    dc_raw = gain_phase
    for z in zeros:
        exact_factor += _factor_phase_exact(z, w)
        straight += _factor_phase_straight(z, w)
        dc_raw += _factor_phase_start(z)
    for p in poles:
        exact_factor -= _factor_phase_exact(p, w)
        straight -= _factor_phase_straight(p, w)
        dc_raw -= _factor_phase_start(p)

    with np.errstate(invalid='ignore'):
        correction = _wrap_deg(np.degrees(np.angle(H)) - exact_factor)
    # No correction where H is not finite or exactly zero (a sample on a jw-axis zero):
    # np.angle carries no information there and the factor phase is the right value.
    usable = np.isfinite(H) & (np.abs(H) > 0) & np.isfinite(correction)
    correction = np.where(usable, correction, 0.0)
    exact = exact_factor + correction

    # Normalise the low-frequency phase into [-180, 180].  On a +-180 tie prefer -180
    # when the phase rises at low frequency or the system contains integrators, else +180.
    # The direction is taken from the analytic slope d(phase)/dw at w -> 0+, so the branch
    # does not depend on the plotted frequency window.
    dc_norm = dc_raw - 360.0 * np.round(dc_raw / 360.0)
    if abs(abs(dc_norm) - 180.0) < 1e-6:
        integrators = sum(1 for p in poles if _root_kind(p) == 'origin')
        rises = _low_frequency_phase_slope(zeros, poles) > 0
        dc_norm = -180.0 if (integrators > 0 or rises) else 180.0
    shift = dc_norm - dc_raw
    exact += shift
    straight += shift

    # --- straight-line magnitude ---
    straight_mag = np.full_like(w, 20.0 * np.log10(abs(gain)), dtype=float)
    for z in zeros:
        straight_mag += _factor_magnitude_straight_db(z, w)
    for p in poles:
        straight_mag -= _factor_magnitude_straight_db(p, w)

    return {
        'omega': w,
        'response': H,
        'magnitude_db': magnitude_db,
        'phase_deg': exact,
        'magnitude_straight_db': straight_mag,
        'phase_straight_deg': straight,
        'phase_dc_deg': float(dc_norm),
        'zeros': zeros,
        'poles': poles,
    }


# Backwards-compatible helpers (used by tests / other callers).
def _make_straight_magnitude_approximation(num, den, w, magnitude_db=None, zeros=None, poles=None):
    if len(w) == 0:
        return []
    return _frequency_response(num, den, w, zeros=zeros, poles=poles)['magnitude_straight_db'].tolist()


def _make_straight_phase_approximation(num, den, w, zeros=None, poles=None):
    if len(w) == 0:
        return []
    return _frequency_response(num, den, w, zeros=zeros, poles=poles)['phase_straight_deg'].tolist()


# ---------------------------------------------------------------------------
# Margins, bandwidth, frequency range
# ---------------------------------------------------------------------------

def _finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _phase_passes_minus_180_at(num, den, zeros, poles, freq) -> bool:
    """True if the (continuous-branch) phase steps across -180 deg (mod 360) at ``freq``."""
    w = np.array([freq * (1 - 1e-9), freq * (1 + 1e-9)])
    phase = _frequency_response(num, den, w, zeros=zeros, poles=poles)['phase_deg']
    lo, hi = min(phase), max(phase)
    k_lo = np.ceil((lo + 180.0) / 360.0)
    k_hi = np.floor((hi + 180.0) / 360.0)
    return k_lo <= k_hi


def _compute_margins(num, den, zeros=None, poles=None) -> dict:
    """Gain/phase margins via python-control.

    ``control.margin`` returns ``(gm, pm, wcg, wcp)`` where ``wcg`` is the *phase*
    crossover frequency (phase = -180 deg, where the gain margin is read) and ``wcp`` is
    the *gain* crossover frequency (|H| = 0 dB, where the phase margin is read).

    Two cases python-control does not report are handled here: a phase crossover at
    w = 0 (negative DC gain / odd number of RHP poles) is flagged with
    ``phase_crossover_at_dc``, and a -180 deg passage at an undamped pole pair, where
    |H| -> inf and the gain margin is zero (-inf dB), is flagged with ``gain_margin_zero``."""
    result = {
        'available': False,
        'gain_margin_db': None,
        'gain_margin_infinite': False,
        'gain_margin_zero': False,
        'phase_margin_deg': None,
        'phase_margin_infinite': False,
        'phase_crossover_freq': None,
        'phase_crossover_at_dc': False,
        'gain_crossover_freq': None,
    }
    sys = _make_control_transfer_function(num, den)
    if sys is None:
        return result
    try:
        gm, pm, wcg, wcp = control.margin(sys)
    except Exception:
        return result
    result['available'] = True
    gm = _finite_or_none(gm) if not (gm is not None and np.isinf(gm)) else np.inf
    if gm is not None and np.isinf(gm):
        result['gain_margin_infinite'] = True
    elif gm is not None and gm > 0:
        result['gain_margin_db'] = 20.0 * np.log10(gm)
    pm_val = pm
    if pm_val is not None and np.isinf(pm_val):
        result['phase_margin_infinite'] = True
    else:
        result['phase_margin_deg'] = _finite_or_none(pm_val)
    wcg = _finite_or_none(wcg)
    if wcg is not None and wcg <= 0:
        result['phase_crossover_at_dc'] = True
        wcg = None
    result['phase_crossover_freq'] = wcg
    result['gain_crossover_freq'] = _finite_or_none(wcp)

    # Undamped pole pairs: python-control drops the -180 deg passage where |H| -> inf.
    if result['gain_margin_infinite'] and poles is not None:
        zeros = np.asarray([] if zeros is None else zeros, dtype=complex)
        axis_freqs = sorted({abs(complex(p).imag) for p in poles if _root_kind(p) == 'axis'})
        for freq in axis_freqs:
            if freq > 0 and _phase_passes_minus_180_at(num, den, zeros, poles, freq):
                result['gain_margin_infinite'] = False
                result['gain_margin_zero'] = True
                result['phase_crossover_freq'] = float(freq)
                break
    return result


def _compute_bandwidth(num, den, zeros, poles) -> dict:
    """-3 dB bandwidth: lowest w where |H(jw)| falls below |H(0)|/sqrt(2).

    Undefined for systems with poles or zeros at the origin (no finite, non-zero DC gain)."""
    num = _strip_leading_zeros(num)
    den = _strip_leading_zeros(den)
    if den[-1] == 0 or num[-1] == 0:
        return {'value': None, 'status': 'undefined'}
    dc_gain = abs(complex(num[-1]) / complex(den[-1]))
    threshold = dc_gain / np.sqrt(2.0)

    corners = _corner_frequencies(poles, zeros)
    if corners:
        w_lo, w_hi = corners[0] / 1e3, corners[-1] * 1e3
    else:
        w_lo, w_hi = 1e-3, 1e3
    grids = [np.logspace(np.log10(w_lo), np.log10(w_hi), 4000)]
    # Dense sub-grids around every corner so that narrow notches of lightly damped zero
    # pairs are not stepped over.
    for corner in corners:
        grids.append(np.logspace(np.log10(corner) - 0.05, np.log10(corner) + 0.05, 2000))
    grid = np.unique(np.concatenate(grids))
    mag = np.abs(_evaluate_transfer(num, den, 1j * grid))
    below = np.where(np.isfinite(mag) & (mag < threshold))[0]
    if below.size == 0:
        return {'value': None, 'status': 'infinite'}
    idx = int(below[0])
    if idx == 0:
        return {'value': float(grid[0]), 'status': 'value'}
    lo, hi = np.log10(grid[idx - 1]), np.log10(grid[idx])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if abs(_evaluate_transfer(num, den, 1j * 10 ** mid)) < threshold:
            hi = mid
        else:
            lo = mid
    return {'value': float(10 ** hi), 'status': 'value'}


def _parse_range_override(override):
    if override is None:
        return None
    try:
        w_min, w_max = float(override[0]), float(override[1])
    except (TypeError, ValueError):
        return None
    if np.isfinite(w_min) and np.isfinite(w_max) and w_min > 0 and w_max > w_min:
        return w_min, w_max
    return None


def _make_freq_vector(num, den, override=None, extra_freqs=None, zeros=None, poles=None):
    """Return the frequency vector for the Bode plot.

    If ``override`` is a valid ``(w_min, w_max)`` pair it is used directly.  Otherwise the
    range spans two decades below and above the extreme corner frequencies (at least
    three decades in total), widened so that every frequency in ``extra_freqs`` (crossover
    frequencies, bandwidth) is inside the range with a decade of padding."""
    rng = _parse_range_override(override)
    if rng is not None:
        return np.logspace(np.log10(rng[0]), np.log10(rng[1]), _N_FREQ)

    zeros = _polynomial_roots(num) if zeros is None else np.asarray(zeros, dtype=complex)
    poles = _polynomial_roots(den) if poles is None else np.asarray(poles, dtype=complex)
    corners = np.asarray(_corner_frequencies(poles, zeros), dtype=float)

    if corners.size:
        w_min = corners.min() / 100
        w_max = corners.max() * 100
    else:
        w_min, w_max = 1e-2, 1e2

    span_decades = np.log10(w_max) - np.log10(w_min)
    if span_decades < 1.5:
        center = np.median(corners) if corners.size else np.sqrt(w_min * w_max)
        w_min = min(w_min, center / 10)
        w_max = max(w_max, center * 10)
        span_decades = np.log10(w_max) - np.log10(w_min)

    min_total_span = 3.0
    if span_decades < min_total_span:
        center = np.median(corners) if corners.size else np.sqrt(w_min * w_max)
        center_log = np.log10(center)
        half_span = min_total_span / 2
        w_min = min(10 ** (center_log - half_span), w_min)
        w_max = max(10 ** (center_log + half_span), w_max)

    for freq in (extra_freqs or []):
        freq = _finite_or_none(freq)
        if freq is not None and freq > 0:
            w_min = min(w_min, freq / 10)
            w_max = max(w_max, freq * 10)

    return np.logspace(np.log10(w_min), np.log10(w_max), _N_FREQ)


# ---------------------------------------------------------------------------
# Full analysis shared by the page and the download routes
# ---------------------------------------------------------------------------

def _json_list(values):
    """Convert an array to a JSON-safe list (non-finite values become null)."""
    out = []
    for v in np.asarray(values, dtype=float).tolist():
        out.append(v if np.isfinite(v) else None)
    return out


@dataclass
class BodeAnalysis:
    numerator: ParsedPolynomial
    denominator: ParsedPolynomial
    response: dict
    margins: dict
    bandwidth: dict
    warnings: list
    corner_frequencies: list

    @property
    def num(self):
        return self.numerator.coeffs

    @property
    def den(self):
        return self.denominator.coeffs

    @property
    def function_latex(self) -> str:
        return f"H(s) = \\frac{{{self.numerator.latex}}}{{{self.denominator.latex}}}"

    @property
    def function_text(self) -> str:
        return f"H(s) = ({self.numerator.text}) / ({self.denominator.text})"

    def phase_level_at(self, freq, target=-180.0):
        """The branch of ``target`` (mod 360) closest to the plotted phase at ``freq``."""
        freq = _finite_or_none(freq)
        if freq is None or freq <= 0:
            return None
        w = self.response['omega']
        phase = self.response['phase_deg']
        value = float(np.interp(np.log10(freq), np.log10(w), phase))
        return target + 360.0 * round((value - target) / 360.0)

    def bode_payload(self) -> dict:
        r = self.response
        m = self.margins
        return {
            'omega': _json_list(r['omega']),
            'magnitude_db': _json_list(r['magnitude_db']),
            'phase_deg': _json_list(r['phase_deg']),
            'magnitude_straight_db': _json_list(r['magnitude_straight_db']),
            'phase_straight_deg': _json_list(r['phase_straight_deg']),
            'phase_dc_deg': r['phase_dc_deg'],
            'margins_available': m['available'],
            'gain_margin_db': _finite_or_none(m['gain_margin_db']),
            'gain_margin_infinite': bool(m['gain_margin_infinite']),
            'gain_margin_zero': bool(m['gain_margin_zero']),
            'phase_margin_deg': _finite_or_none(m['phase_margin_deg']),
            'phase_margin_infinite': bool(m['phase_margin_infinite']),
            'phase_crossover_freq': m['phase_crossover_freq'],
            'phase_crossover_at_dc': bool(m['phase_crossover_at_dc']),
            'gain_crossover_freq': m['gain_crossover_freq'],
            'phase_crossover_level_deg': self.phase_level_at(m['phase_crossover_freq']),
            'bandwidth': self.bandwidth['value'],
            'bandwidth_status': self.bandwidth['status'],
            'corner_frequencies': self.corner_frequencies,
        }


def analyse_transfer_function(num_str: str, den_str: str, rng=None) -> BodeAnalysis:
    """Parse both polynomials and compute everything the Bode page needs.

    Raises ``ValueError`` with a user-facing message for invalid input."""
    try:
        numerator = parse_polynomial(num_str)
    except ValueError as exc:
        raise ValueError(f"Error parsing numerator: {exc}")
    try:
        denominator = parse_polynomial(den_str)
    except ValueError as exc:
        raise ValueError(f"Error parsing denominator: {exc}")

    num, den = numerator.coeffs, denominator.coeffs
    zeros, poles = numerator.roots, denominator.roots

    margins = _compute_margins(num, den, zeros=zeros, poles=poles)
    bandwidth = _compute_bandwidth(num, den, zeros, poles)
    extra = [margins['phase_crossover_freq'], margins['gain_crossover_freq'], bandwidth['value']]
    w = _make_freq_vector(num, den, rng, extra_freqs=extra, zeros=zeros, poles=poles)
    response = _frequency_response(num, den, w, zeros=zeros, poles=poles)

    warnings = []
    if numerator.degree > denominator.degree:
        warnings.append(
            "Non-proper transfer function (deg(num) > deg(den)): the magnitude keeps rising at high frequency."
        )
    pole_kinds = [_root_kind(p) for p in poles]
    zero_kinds = [_root_kind(z) for z in zeros]
    if 'rhp' in pole_kinds:
        warnings.append("System has right-half-plane pole(s): open-loop unstable.")
    if 'axis' in pole_kinds:
        warnings.append("System has pole(s) on the imaginary axis: marginally stable (undamped resonance).")
    if 'rhp' in zero_kinds:
        warnings.append("Non-minimum-phase zero(s) detected (RHP zero): the phase lags more than the magnitude suggests.")
    if not margins['available']:
        warnings.append(
            "Gain margin and phase margin are only available for transfer functions with real coefficients."
        )

    return BodeAnalysis(
        numerator=numerator,
        denominator=denominator,
        response=response,
        margins=margins,
        bandwidth=bandwidth,
        warnings=warnings,
        corner_frequencies=_corner_frequencies(poles, zeros),
    )


# ---------------------------------------------------------------------------
# Nyquist
# ---------------------------------------------------------------------------

def _bisect_imag_zero(num, den, w_lo, w_hi):
    """Frequency in (w_lo, w_hi) where Im L(jw) changes sign (bisection in log w)."""
    f_lo = float(np.imag(_evaluate_transfer(num, den, 1j * w_lo)))
    lo, hi = np.log10(w_lo), np.log10(w_hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = float(np.imag(_evaluate_transfer(num, den, 1j * 10 ** mid)))
        if not np.isfinite(f_mid):
            return None
        if f_mid == 0:
            return 10 ** mid
        if (f_mid < 0) == (f_lo < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 10 ** (0.5 * (lo + hi))


def _real_axis_crossings(num, den, w, resp, skip_freqs=()):
    """Real-axis crossings of the locus for w > 0: sign changes of Im L(jw), refined by
    bisection.  Intervals containing a pole on the imaginary axis (``skip_freqs``) are
    ignored, because there the sign flips while |L| passes through infinity."""
    crossings = []
    imag_vals = np.imag(resp)
    real_vals = np.real(resp)
    n = len(w)
    skip_freqs = [float(f) for f in skip_freqs if np.isfinite(f) and f > 0]

    def interval_has_axis_pole(a, b):
        return any(a * (1 - 1e-9) <= f <= b * (1 + 1e-9) for f in skip_freqs)

    i = 1  # skip the w = 0 sample: every real-coefficient system is real there
    while i < n - 1:
        y1 = imag_vals[i]
        if not np.isfinite(resp[i]):
            i += 1
            continue
        if y1 == 0:
            # A run of samples exactly on the axis: report the run once if the sign changes across it.
            j = i
            while j < n - 1 and np.isfinite(resp[j + 1]) and imag_vals[j + 1] == 0:
                j += 1
            before = imag_vals[i - 1] if np.isfinite(resp[i - 1]) else 0.0
            after = imag_vals[j + 1] if j + 1 < n and np.isfinite(resp[j + 1]) else 0.0
            if before * after < 0 and not interval_has_axis_pole(w[i - 1], w[min(j + 1, n - 1)]):
                mid = (i + j) // 2
                crossings.append((float(w[mid]), float(real_vals[mid])))
            i = j + 1
            continue
        y2 = imag_vals[i + 1]
        if np.isfinite(resp[i + 1]) and y1 * y2 < 0 and not interval_has_axis_pole(w[i], w[i + 1]):
            w_cross = _bisect_imag_zero(num, den, w[i], w[i + 1])
            if w_cross is not None:
                value = _evaluate_transfer(num, den, 1j * w_cross)
                if np.isfinite(value):
                    crossings.append((float(w_cross), float(np.real(value))))
        i += 1
    return crossings


def _count_encirclements(num, den, w_positive, crossings=()):
    """Counter-clockwise encirclements of -1 by L(jw) for w from -inf to +inf.

    Returns ``None`` when the count cannot be trusted: non-finite samples (poles on the
    imaginary axis need the indented contour), a locus through -1, or a non-integer
    winding number.  The grid is refined around the real-axis crossings, where the locus
    may pass close to -1 between two samples."""
    grids = [np.asarray(w_positive, dtype=float)]
    for w_cross, _ in crossings:
        if w_cross > 0:
            grids.append(np.logspace(np.log10(w_cross) - 0.02, np.log10(w_cross) + 0.02, 2000))
            grids.append(np.array([w_cross]))
    w_pos = np.unique(np.concatenate(grids))
    w_full = np.concatenate([-w_pos[:0:-1], w_pos])
    resp = _evaluate_transfer(num, den, 1j * w_full) + 1.0
    if not np.all(np.isfinite(resp)):
        return None
    if np.min(np.abs(resp)) < 1e-9:
        return None  # locus passes through the critical point
    angles = np.unwrap(np.angle(resp))
    turns = (angles[-1] - angles[0]) / (2 * np.pi)
    if not np.isfinite(turns) or abs(turns - round(turns)) > 0.1:
        return None
    return int(round(turns))


def _make_nyquist_data(num, den, poles, gm_db, pm, zeros=None, w=None, margins=None):
    """Compute Nyquist contour samples and qualitative analysis information."""
    num = _strip_leading_zeros(num)
    den = _strip_leading_zeros(den)
    poles = np.asarray(poles, dtype=complex)
    if w is None:
        w = _make_freq_vector(num, den, zeros=zeros, poles=poles)
    w_positive = np.concatenate(([0.0], np.asarray(w, dtype=float)))
    resp_positive = _evaluate_transfer(num, den, 1j * w_positive)

    # Mirror the contour for negative frequencies (from -w_max down to -w_1, without 0).
    w_negative = -w_positive[:0:-1]
    resp_negative = _evaluate_transfer(num, den, 1j * w_negative)

    nyquist_data = {
        "positive": {
            "frequencies": w_positive.tolist(),
            "real": _json_list(np.real(resp_positive)),
            "imag": _json_list(np.imag(resp_positive)),
        },
        "negative": {
            "frequencies": w_negative.tolist(),
            "real": _json_list(np.real(resp_negative)),
            "imag": _json_list(np.imag(resp_negative)),
        },
        "critical_point": {"real": -1.0, "imag": 0.0},
    }

    def _complex_dict(value: complex, frequency: float) -> dict:
        if value is None or not np.isfinite(value):
            return {"frequency": float(frequency), "real": None, "imag": None}
        return {"frequency": float(frequency), "real": float(np.real(value)), "imag": float(np.imag(value))}

    nyquist_data["low_freq"] = _complex_dict(resp_positive[0], w_positive[0])
    nyquist_data["high_freq"] = _complex_dict(resp_positive[-1], w_positive[-1])

    finite_resp = resp_positive[np.isfinite(resp_positive)]
    distances = np.abs(finite_resp + 1) if finite_resp.size else np.array([])
    d_min = float(distances.min()) if distances.size else None
    nyquist_data["min_distance"] = d_min

    analysis_items = []

    # Open-loop pole information
    pole_kinds = [_root_kind(p) for p in poles]
    rhp_poles = pole_kinds.count('rhp')
    axis_poles = pole_kinds.count('axis') + pole_kinds.count('origin')
    axis_pole_freqs = sorted({abs(complex(p).imag) for p in poles if _root_kind(p) == 'axis'})
    if rhp_poles == 0 and axis_poles == 0:
        analysis_items.append(
            _analysis_item(
                "Open-loop pole distribution",
                "Stable (no RHP poles)",
                r"All open-loop poles lie in the left half-plane (P = 0). By the Nyquist criterion Z = P − N, the unity-feedback loop is stable exactly when the locus does not encircle \((-1,0\mathrm{j})\).",
                level="success",
            )
        )
    elif rhp_poles == 0:
        analysis_items.append(
            _analysis_item(
                "Open-loop pole distribution",
                f"No RHP poles, {axis_poles} on the imaginary axis",
                (
                    r"P = 0 poles in the open right half-plane, but {a} pole(s) lie on the imaginary axis (marginally stable open loop). The Nyquist contour must be indented around them, which adds large arcs to the locus; closed-loop stability still requires no net encirclement of \((-1,0\mathrm{{j}})\)."
                ).format(a=axis_poles),
                level="warning",
            )
        )
    else:
        extra = f" and {axis_poles} pole(s) on the imaginary axis" if axis_poles else ""
        analysis_items.append(
            _analysis_item(
                "Open-loop pole distribution",
                f"{rhp_poles} pole(s) in RHP",
                (
                    r"P = {p} open-loop pole(s) in the right half-plane{extra}. By the Nyquist criterion Z = P − N, closed-loop stability requires the locus to encircle \((-1,0\mathrm{{j}})\) exactly {p} time(s) counter-clockwise."
                ).format(p=rhp_poles, extra=extra),
                level="warning",
            )
        )

    # Minimum distance to -1+j0 (critical point)
    if d_min is not None:
        idx_min = int(np.nanargmin(np.where(np.isfinite(resp_positive), np.abs(resp_positive + 1), np.nan)))
        w_at_min = w_positive[idx_min]
        analysis_items.append(
            _analysis_item(
                "Distance to critical point",
                _format_latex_real(d_min),
                (
                    r"The Nyquist locus comes closest to the critical point at {freq} with \(|L(\mathrm{{j}}\omega)+1| = {distance}\) (the inverse of the peak sensitivity, sampled)."
                ).format(freq=_format_latex_real(w_at_min, 'rad/s'), distance=_format_real_latex(d_min)),
                level="info" if d_min > 0.2 else "warning",
            )
        )

    # Real-axis crossings for w > 0
    all_real = bool(finite_resp.size) and bool(np.all(np.abs(np.imag(finite_resp)) <= 1e-12 * np.maximum(1.0, np.abs(finite_resp))))
    crossings = [] if all_real else _real_axis_crossings(num, den, w_positive, resp_positive, axis_pole_freqs)
    crossings.sort(key=lambda pair: pair[0])
    nyquist_data["real_axis_crossings"] = [{"frequency": rc, "real": rr} for rc, rr in crossings]
    dc_value = resp_positive[0]
    starts_on_negative_axis = bool(np.isfinite(dc_value)) and float(np.real(dc_value)) < 0 \
        and abs(float(np.imag(dc_value))) <= 1e-12 * max(1.0, abs(dc_value))
    if all_real:
        detail = r"\(L(\mathrm{j}\omega)\) is real for every frequency, so the locus runs along the real axis (a pure gain, or a ratio of even polynomials)."
    elif crossings:
        crossings.sort(key=lambda pair: pair[0])
        parts = []
        for rc, rr in crossings:
            if rr < -1:
                parts.append(
                    f"{_format_latex_real(rc, 'rad/s')} → {_format_latex_real(rr)} (left of −1: |L| > 1 at −180°, gain margin {_format_latex_real(-20 * np.log10(abs(rr)), 'dB')})"
                )
            elif rr < 0:
                parts.append(
                    f"{_format_latex_real(rc, 'rad/s')} → {_format_latex_real(rr)} (between −1 and 0: gain margin {_format_latex_real(-20 * np.log10(abs(rr)), 'dB')})"
                )
            else:
                parts.append(f"{_format_latex_real(rc, 'rad/s')} → {_format_latex_real(rr)} (positive real axis)")
        detail = (
            r"Real-axis crossings (\(\Im\{{L(\mathrm{{j}}\omega)\}}=0\), \(\omega>0\)): {points}. A crossing to the left of \(-1\) means the loop gain exceeds 1 where the phase is \(-180^\circ\) (negative gain margin in dB); for an open-loop stable plant the closed loop is then unstable."
        ).format(points="; ".join(parts))
    elif margins is not None and margins.get('gain_margin_zero'):
        detail = (
            r"The locus does not cross the real axis at a finite point for \(\omega>0\); the phase passes \(-180^\circ\) at the undamped pole ({freq}) where \(|L(\mathrm{{j}}\omega)| \to \infty\), so the gain margin is zero (\(-\infty\) dB)."
        ).format(freq=_format_latex_real(margins.get('phase_crossover_freq'), 'rad/s'))
    elif starts_on_negative_axis:
        l0 = float(np.real(dc_value))
        detail = (
            r"The locus does not cross the real axis for \(\omega>0\), but it starts on the negative real axis at \(L(0) = {value}\) (phase \(\pm180^\circ\) at \(\omega \to 0\)); the corresponding gain margin is {gm}."
        ).format(value=_format_real_latex(l0), gm=_format_latex_real(-20 * np.log10(abs(l0)), 'dB'))
    elif margins is None or margins.get('gain_margin_infinite'):
        detail = r"The Nyquist locus does not cross the real axis for positive frequencies, so the phase never reaches \(\pm180^\circ\) and the gain margin is infinite."
    else:
        detail = r"The Nyquist locus does not cross the real axis for positive frequencies."
    analysis_items.append(
        _analysis_item(
            "Real-axis intercepts",
            "on the real axis" if all_real else f"{len(crossings)} crossing(s)",
            detail,
            level="info",
        )
    )

    # Unity-feedback closed-loop pole estimate and encirclement count
    closed_loop_den = _strip_leading_zeros(np.polyadd(den, num).tolist())
    if all(c == 0 for c in closed_loop_den):
        analysis_items.append(
            _analysis_item(
                "Unity-feedback verdict",
                "Ill-posed loop",
                r"\(1 + L(s)\) is identically zero, so the unity-feedback closed loop is not defined.",
                level="warning",
            )
        )
    else:
        closed_loop_poles = _polynomial_roots(closed_loop_den)
        unstable_closed = sum(1 for p in closed_loop_poles if _root_kind(p) in ('rhp',))
        marginal_closed = sum(1 for p in closed_loop_poles if _root_kind(p) in ('axis', 'origin'))
        encirclements = None
        if not axis_poles and marginal_closed == 0:
            encirclements = _count_encirclements(num, den, w_positive, crossings)
        if encirclements is not None and rhp_poles - encirclements != unstable_closed:
            # The sampled winding number disagrees with the closed-loop poles: do not print
            # a contradictory statement.
            encirclements = None
        if encirclements is None:
            if axis_poles:
                enc_text = "The encirclement count is not evaluated numerically because the open loop has poles on the imaginary axis (the contour must be indented around them)."
            elif marginal_closed:
                enc_text = "The locus passes through the critical point −1 itself (a closed-loop pole lies on the imaginary axis), so the encirclement count is not defined."
            else:
                enc_text = "The encirclement count could not be determined reliably from the sampled locus (it passes very close to −1)."
        else:
            direction = "counter-clockwise" if encirclements > 0 else "clockwise"
            enc_text = (
                f"Sampled locus: N = {abs(encirclements)} {direction} encirclement(s) of −1"
                if encirclements else "Sampled locus: no net encirclement of −1"
            ) + f", so Z = P − N = {rhp_poles - encirclements} closed-loop RHP pole(s)."
        if unstable_closed == 0 and marginal_closed == 0:
            cl_value = "Predicted stable"
            cl_detail = "All closed-loop poles (unity feedback) lie in the left half-plane. " + enc_text
            level = "success"
        elif unstable_closed == 0:
            cl_value = "Marginally stable"
            cl_detail = f"{marginal_closed} closed-loop pole(s) lie on the imaginary axis for unity feedback. " + enc_text
            level = "warning"
        else:
            cl_value = "Predicted unstable"
            cl_detail = f"{unstable_closed} closed-loop pole(s) fall in the right half-plane for unity feedback. " + enc_text
            level = "warning"
        analysis_items.append(_analysis_item("Unity-feedback verdict", cl_value, cl_detail, level=level))

    # Gain/phase margin recap
    if margins is not None:
        if margins['gain_margin_infinite']:
            gm_text = "∞"
        elif margins.get('gain_margin_zero'):
            gm_text = r"\(-\infty\;dB\)"
        elif margins['gain_margin_db'] is not None:
            gm_text = _format_latex_real(margins['gain_margin_db'], 'dB')
            if margins.get('phase_crossover_at_dc'):
                gm_text += r" (at \(\omega \to 0\))"
        else:
            gm_text = "—"
        pm_text = "∞" if margins['phase_margin_infinite'] else (
            _format_latex_real(margins['phase_margin_deg'], r'^\circ') if margins['phase_margin_deg'] is not None else "—")
    else:
        gm_text = _format_latex_real(gm_db, 'dB') if gm_db is not None else "—"
        pm_text = _format_latex_real(pm, r'^\circ') if pm is not None else "—"
    analysis_items.append(
        _analysis_item(
            "Classical margins",
            f"GM: {gm_text} / PM: {pm_text}",
            r"Margins read from the Bode plot: the gain margin is the distance of the negative real-axis crossing from \(-1\), the phase margin the angle at which the locus crosses the unit circle.",
            level="info",
        )
    )

    return nyquist_data, analysis_items


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

_PAGE_TITLE = "Bode Plot Calculator Online – Phase Margin & Gain Margin"
_META_DESCRIPTION = (
    "Compute Bode magnitude/phase and stability margins from a transfer function. Includes "
    "pole-zero map, bandwidth, exports, and optional Nyquist plot view."
)
_DEFAULT_NUM = "(s+2)"
_DEFAULT_DEN = "(s+10)(s+0.1)"


def _empty_context(**overrides):
    context = dict(
        bode_data=None,
        error=None,
        warning=None,
        default_num=_DEFAULT_NUM,
        default_den=_DEFAULT_DEN,
        function_str=None,
        function_text=None,
        zeros=None,
        poles=None,
        pz_pairs=None,
        gm=None,
        pm=None,
        wg=None,
        wp=None,
        pz_plot=None,
        nyquist_data=None,
        nyquist_analysis=None,
        show_nyquist=False,
        active_action='bode',
        page_title=_PAGE_TITLE,
        meta_description=_META_DESCRIPTION,
    )
    context.update(overrides)
    return context


def _range_from_request(source):
    w_min_str = source.get('w_min')
    w_max_str = source.get('w_max')
    if w_min_str and w_max_str:
        return (w_min_str, w_max_str)
    return None


@bode_plot_bp.route('/', methods=['GET', 'POST'])
def bode_plot():
    if request.method == 'POST':
        submit_action = request.form.get('submit_action', 'bode') or 'bode'
    else:
        submit_action = request.args.get('view', 'bode') or 'bode'
    show_nyquist = submit_action == 'nyquist'

    if request.method == 'GET':
        return render_template(
            "bode_plot.html",
            **_empty_context(show_nyquist=show_nyquist, active_action=submit_action),
        )

    user_num = request.form.get('numerator', _DEFAULT_NUM)
    user_den = request.form.get('denominator', _DEFAULT_DEN)
    rng = _range_from_request(request.form)

    try:
        analysis = analyse_transfer_function(user_num, user_den, rng)
    except ValueError as exc:
        return render_template(
            "bode_plot.html",
            **_empty_context(
                error=str(exc),
                default_num=user_num,
                default_den=user_den,
                show_nyquist=show_nyquist,
                active_action=submit_action,
            ),
        )

    zeros, poles = analysis.response['zeros'], analysis.response['poles']
    pz_plot = {
        "zeros": [{"re": float(np.real(z)), "im": float(np.imag(z))} for z in zeros if np.isfinite(z)],
        "poles": [{"re": float(np.real(p)), "im": float(np.imag(p))} for p in poles if np.isfinite(p)],
    }
    zero_list = [format_complex(z) for z in zeros]
    pole_list = [format_complex(p) for p in poles]
    pz_pairs = list(zip_longest(zero_list, pole_list, fillvalue=""))

    nyquist_plot = None
    nyquist_analysis = None
    if show_nyquist:
        nyquist_plot, nyquist_analysis = _make_nyquist_data(
            analysis.num, analysis.den, poles,
            analysis.margins['gain_margin_db'], analysis.margins['phase_margin_deg'],
            zeros=zeros, w=analysis.response['omega'], margins=analysis.margins,
        )

    m = analysis.margins
    return render_template(
        "bode_plot.html",
        **_empty_context(
            bode_data=analysis.bode_payload(),
            warning=("\n".join(analysis.warnings) if analysis.warnings else None),
            default_num=user_num,
            default_den=user_den,
            function_str=analysis.function_latex,
            function_text=analysis.function_text,
            zeros=zero_list,
            poles=pole_list,
            pz_pairs=pz_pairs,
            gm=m['gain_margin_db'],
            pm=m['phase_margin_deg'],
            wg=m['phase_crossover_freq'],
            wp=m['gain_crossover_freq'],
            pz_plot=pz_plot,
            nyquist_data=nyquist_plot,
            nyquist_analysis=nyquist_analysis,
            show_nyquist=show_nyquist,
            active_action=submit_action,
        ),
    )


def _error_response(exc, status=400):
    return Response(f"Error parsing inputs: {escape(str(exc))}", status=status, mimetype='text/plain')


def _fmt_csv(value):
    return f"{value:.6g}" if value is not None and np.isfinite(value) else ""


@bode_plot_bp.route('/download_csv')
def download_csv():
    try:
        analysis = analyse_transfer_function(
            request.args.get('numerator', ''),
            request.args.get('denominator', ''),
            _range_from_request(request.args),
        )
    except ValueError as exc:
        return _error_response(exc)

    r = analysis.response
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow([
        'Frequency (rad/s)', 'Magnitude (dB)', 'Phase (deg)',
        'Magnitude asymptote (dB)', 'Phase asymptote (deg)',
    ])
    for row in zip(r['omega'], r['magnitude_db'], r['phase_deg'], r['magnitude_straight_db'], r['phase_straight_deg']):
        cw.writerow([_fmt_csv(v) for v in row])
    output = si.getvalue().encode('utf-8')

    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename="bode_data.csv"'},
    )


@bode_plot_bp.route('/download_png')
def download_png():
    try:
        analysis = analyse_transfer_function(
            request.args.get('numerator', ''),
            request.args.get('denominator', ''),
            _range_from_request(request.args),
        )
    except ValueError as exc:
        return _error_response(exc)

    r = analysis.response
    w = r['omega']

    # Object-oriented matplotlib API: no global pyplot state, safe for concurrent requests.
    fig = Figure(figsize=(8, 6), layout="constrained")
    FigureCanvasAgg(fig)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.semilogx(w, r['magnitude_db'], label="Exact")
    ax1.semilogx(w, r['magnitude_straight_db'], linestyle='--', alpha=0.7, label="Asymptotes")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend(loc='best')
    ax2.semilogx(w, r['phase_deg'], label="Exact")
    ax2.semilogx(w, r['phase_straight_deg'], linestyle='--', alpha=0.7, label="Asymptotes")
    ax2.set_ylabel("Phase (°)")
    ax2.set_xlabel("Frequency (rad/s)")
    ax2.grid(True, which='both', linestyle='--')
    try:
        fig.suptitle(f"${analysis.function_latex}$", y=0.98)
        fig.canvas.draw()
    except Exception:
        fig.suptitle(analysis.function_text, y=0.98)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300)

    return Response(
        buf.getvalue(),
        mimetype='image/png',
        headers={'Content-Disposition': 'attachment; filename="bode_plot.png"'},
    )
