from flask import Blueprint, render_template, request
import numpy as np
import sympy as sp
from scipy.signal import lti, impulse, residue
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
import ast
import re


def _int_if_close(val, tol=1e-12):
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

inverse_laplace_bp = Blueprint('inverse_laplace', __name__)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor
)

def _parse_poly(txt: str) -> np.ndarray:
    """Parse ``txt`` into a coefficient vector for :func:`scipy.signal.residue`."""
    txt = txt.strip()
    if txt.startswith('['):
        try:
            return np.asarray(ast.literal_eval(txt), dtype=complex)
        except Exception:
            items = txt.strip('[]')
            parts = [p.strip() for p in items.split(',') if p.strip()]
            vals = [complex(parse_expr(p.replace('j', 'I'), transformations=_TRANSFORMS))
                    for p in parts]
            return np.asarray(vals, dtype=complex)

    s = sp.symbols('s')
    expr = parse_expr(
        txt.replace('j', 'I'),
        local_dict={'s': s},
        transformations=_TRANSFORMS,
        evaluate=False
    )
    coeffs = sp.Poly(sp.expand(expr), s).all_coeffs()  # descending s powers
    coeffs = coeffs[::-1]
    return np.asarray([complex(c) for c in coeffs], dtype=complex)

def _coeffs_to_poly(coeffs):
    s = sp.symbols('s')
    deg = len(coeffs) - 1
    return sum(
        sp.sympify(_int_if_close(coeffs[k])) * s**(deg - k)
        for k in range(len(coeffs))
    )

def _inverse_laplace_expr(num: np.ndarray, den: np.ndarray) -> sp.Expr:
    """Symbolic inverse Laplace transform of num/den."""
    t = sp.symbols('t')
    try:
        r, p, kvals = residue(num, den)
    except Exception:
        expr = sp.inverse_laplace_transform(_coeffs_to_poly(num)/_coeffs_to_poly(den), sp.symbols('s'), t)
        return sp.simplify(expr)

    expr = 0
    # direct polynomial terms -> Dirac delta derivatives
    kvals_rev = kvals[::-1]
    for order, ki in enumerate(kvals_rev):
        if not np.isclose(ki, 0):
            delta = sp.DiracDelta(t)
            if order > 0:
                delta = delta.diff(t, order)
            expr += sp.sympify(_int_if_close(ki)) * delta
            for ri, pi in zip(r, p):
                if not np.isclose(ri, 0):
                    expr += sp.sympify(_int_if_close(ri)) * sp.exp(pi*t) * sp.Heaviside(t)

    return sp.simplify(expr)

def _impulse_response(num, den, N=10):
    """Sampled impulse response using SciPy's ``impulse``."""
    sys = lti(num, den)
    t_vals = np.arange(N, dtype=float)
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

@inverse_laplace_bp.route('/', methods=['GET', 'POST'])
def inverse_laplace():
    num_txt = request.form.get('numerator', '[1]')
    den_txt = request.form.get('denominator', '[1, 1]')
    ht_ltx = None
    tf_ltx = None
    seq = None
    error = None
    if request.method == 'POST':
        try:
            num = _parse_poly(num_txt)
            den = _parse_poly(den_txt)

            num_expr = _coeffs_to_poly(num)
            den_expr = _coeffs_to_poly(den)
            tf_ltx = sp.latex(num_expr/den_expr)

            expr = _inverse_laplace_expr(num, den)
            ht_ltx = sp.latex(expr).replace('\\theta', '\\varepsilon')
            tf_ltx = re.sub(r'(\d)\.0(?!\d)', r'\1', tf_ltx)
            ht_ltx = re.sub(r'(\d)\.0(?!\d)', r'\1', ht_ltx)
            seq = _impulse_response(num, den, N=10)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return render_template(
        'inverse_laplace.html',
        default_num=num_txt,
        default_den=den_txt,
        tf_latex=tf_ltx,
        ht_latex=ht_ltx,
        seq=seq,
        error=error
    )