from flask import Blueprint, render_template, request
import numpy as np
import sympy as sp
from scipy.signal import lfilter
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
from scipy.signal import residuez, dlti, dimpulse
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

inverse_z_bp = Blueprint('inverse_z', __name__)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor
)

def _parse_poly(txt: str) -> np.ndarray:
    """Parse ``txt`` into a coefficient vector for :func:`scipy.signal.residuez`."""
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

    z = sp.symbols('z')
    expr = parse_expr(
        txt.replace('j', 'I'),
        local_dict={'z': z},
        transformations=_TRANSFORMS,
        evaluate=False
    )
    coeffs = sp.Poly(sp.expand(expr), z).all_coeffs()  # descending z+ powers
    coeffs = coeffs[::-1]
    return np.asarray([complex(c) for c in coeffs], dtype=complex)

def _coeffs_to_poly(coeffs):
    z = sp.symbols('z')
    deg = len(coeffs) - 1
    return sum(
        sp.sympify(_int_if_close(coeffs[k])) * z**(deg - k)
        for k in range(len(coeffs))
    )



def _inverse_z_expr(num: np.ndarray, den: np.ndarray) -> sp.Expr:
    """
    Symbolic inverse Z-transform of num/den.
    Falls back to a FIR-only formula when residuez chokes (len(num) > len(den)).
    """
    try:
        r, p, kvals = residuez(num, den)          # SciPy ≥1.10
    except Exception:                              # improper or SciPy bug
        k = sp.symbols('k')
        expr = 0
        for idx, c in enumerate(num):
            if not np.isclose(c, 0):
                expr += c * sp.DiracDelta(k - idx)
        return expr                                # FIR done — no poles
    # --- normal IIR path ---
    k = sp.symbols('k')
    expr = 0
    for i, ki in enumerate(kvals):
        if not np.isclose(ki, 0):
            expr += sp.sympify(_int_if_close(ki)) * sp.DiracDelta(k - i)
    for ri, pi in zip(r, p):
        if not np.isclose(ri, 0):
            expr += sp.sympify(_int_if_close(ri)) * (pi**k) * sp.Heaviside(k)

    return sp.simplify(expr)



def _impulse_response(num, den, N=10):
    """Impulse via direct filtering – robust for pure delays & improper H(z)."""
    x = np.zeros(N, dtype=complex)
    x[0] = 1                          # δ[k]
    y = lfilter(num, den, x)
    seq = []
    for v in y:
        c = complex(v)                # keep imaginary part if any
        if abs(c.imag) < 1e-12:
            c = float(c.real)
            if c.is_integer():
                c = int(c)
        c = _int_if_close(c)
        seq.append(c)
    return seq

@inverse_z_bp.route('/', methods=['GET', 'POST'])
def inverse_z():
    num_txt = request.form.get('numerator', '[1, 0]')
    den_txt = request.form.get('denominator', '[2, 1]')
    hz_ltx = None
    tf_ltx = None
    seq = None
    error = None
    if request.method == 'POST':
        try:
            num = _parse_poly(num_txt)
            den = _parse_poly(den_txt)

            z = sp.symbols('z')
            def _coeffs_to_poly(coeffs):
                z = sp.symbols('z')
                deg = len(coeffs) - 1
                return sum(sp.sympify(coeffs[k]) * z**(deg - k) for k in range(len(coeffs)))

            num_expr = _coeffs_to_poly(num)
            den_expr = _coeffs_to_poly(den)
            tf_ltx = sp.latex(num_expr/den_expr)

            expr = _inverse_z_expr(num, den)
            hz_ltx = sp.latex(expr).replace('\\theta', '\\varepsilon')
            tf_ltx = re.sub(r'(\d)\.0(?!\d)', r'\1', tf_ltx)
            hz_ltx = re.sub(r'(\d)\.0(?!\d)', r'\1', hz_ltx)
            seq = _impulse_response(num, den, N=10)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return render_template(
        'inverse_z.html',
        default_num=num_txt,
        default_den=den_txt,
        tf_latex=tf_ltx,
        hz_latex=hz_ltx,
        seq=seq,
        error=error
    )