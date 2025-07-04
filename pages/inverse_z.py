from flask import Blueprint, render_template, request
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
from scipy.signal import residuez, dlti, dimpulse
import ast

inverse_z_bp = Blueprint('inverse_z', __name__)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor
)

def _parse_poly(txt: str):
    txt = txt.strip()
    if txt.startswith('['):
        return np.asarray(ast.literal_eval(txt), dtype=complex)
    expr = parse_expr(
        txt.replace('j', 'I'),
        local_dict={'z': sp.symbols('z')},
        transformations=_TRANSFORMS,
        evaluate=False
    )
    coeffs = sp.Poly(sp.expand(expr), sp.symbols('z')).all_coeffs()
    return np.asarray([complex(c) for c in coeffs], dtype=complex)


def _inverse_z_expr(num, den):
    r, p, k = residuez(num, den)
    n = sp.symbols('n')
    expr = 0
    for i, ki in enumerate(k):
        if not np.isclose(ki, 0):
            expr += ki * sp.DiracDelta(n - i)
    for ri, pi in zip(r, p):
        if not np.isclose(ri, 0):
            expr += ri * (pi**n) * sp.Heaviside(n)
    return sp.simplify(expr)


def _impulse_response(num, den, N=10):
    sys = dlti(num, den, dt=1)
    _, h = dimpulse(sys, n=N)
    return [float(v) for v in np.squeeze(h)]


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
            num_expr = sp.Poly(num, z).as_expr()
            den_expr = sp.Poly(den, z).as_expr()
            tf_ltx = sp.latex(num_expr/den_expr)

            expr = _inverse_z_expr(num, den)
            hz_ltx = sp.latex(expr).replace('\\theta', '\\varepsilon')
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