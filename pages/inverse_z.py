from flask import Blueprint, render_template, request
import numpy as np
import sympy as sp
from scipy.signal import lfilter
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
from scipy.signal import residuez, dlti, dimpulse
try:
    from sympy.integrals.transforms import inverse_z_transform as _sympy_inverse_z
except Exception:  # pragma: no cover - optional
    _sympy_inverse_z = None
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

_PAIR_TABLE = {}  # key: (tuple(num), tuple(den)), value: sympy Expr
# TODO: populate externally

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



def _inverse_z_expr(num: np.ndarray, den: np.ndarray, roc: float | None = None) -> sp.Expr:
    """Symbolic inverse Z-transform of ``num/den``.

    Parameters
    ----------
    num, den : np.ndarray
        Coefficient arrays.
    roc : float or None, optional
        Region-of-convergence radius. ``None`` assumes causal sequence.
    """

    key = (tuple(num), tuple(den))
    if key in _PAIR_TABLE:
        return _PAIR_TABLE[key]

    z, k = sp.symbols('z k')
    H = _coeffs_to_poly(num) / _coeffs_to_poly(den)

    if _sympy_inverse_z is not None:
        try:
            expr = _sympy_inverse_z(H, z, k, noconds=True)
            return sp.simplify(expr)
        except NotImplementedError:
            raise
        except Exception:
            pass

    try:
        r, p, kvals = residuez(num, den)          # SciPy ≥1.10
    except Exception:                              # improper or SciPy bug
        k = sp.symbols('k')
        expr = 0
        for idx, c in enumerate(num):
            if not np.isclose(c, 0):
                expr += c * sp.DiracDelta(k - idx)
        return expr                                # FIR done — no poles

    expr = 0
    for i, ki in enumerate(kvals):
        if not np.isclose(ki, 0):
            expr += sp.sympify(_int_if_close(ki)) * sp.DiracDelta(k - i)
    for ri, pi in zip(r, p):
        if np.isclose(ri, 0):
            continue
        term = sp.sympify(_int_if_close(ri)) * (pi**k)
        if roc is None:
            term *= sp.Heaviside(k)
        else:
            if abs(pi) > roc:
                term *= sp.Heaviside(-k - 1)
            else:
                term *= sp.Heaviside(k)
        expr += term

    return sp.simplify(expr)



def _impulse_response(num, den, n_samples: int = 10):
    """Impulse via direct filtering – robust for pure delays & improper ``H(z)``."""
    x = np.zeros(n_samples, dtype=complex)
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
    roc_txt = request.form.get('roc_radius', '')
    hz_ltx = None
    tf_ltx = None
    seq = None
    error = None
    roc_used = 'auto'
    if request.method == 'POST':
        try:
            num = _parse_poly(num_txt)
            den = _parse_poly(den_txt)
            roc_val = None
            if roc_txt.strip():
                roc_val = float(roc_txt)
                roc_used = roc_val
            z = sp.symbols('z')
            def _coeffs_to_poly(coeffs):
                z = sp.symbols('z')
                deg = len(coeffs) - 1
                return sum(sp.sympify(coeffs[k]) * z**(deg - k) for k in range(len(coeffs)))

            num_expr = _coeffs_to_poly(num)
            den_expr = _coeffs_to_poly(den)
            tf_ltx = sp.latex(num_expr/den_expr)

            expr = _inverse_z_expr(num, den, roc=roc_val)
            hz_ltx = sp.latex(expr).replace('\\theta', '\\varepsilon')
            tf_ltx = re.sub(r'(\d)\.0(?!\d)', r'\1', tf_ltx)
            hz_ltx = re.sub(r'(\d)\.0(?!\d)', r'\1', hz_ltx)
            seq = _impulse_response(num, den, n_samples=10)
        except NotImplementedError as exc:
            error = f'SymPy not implemented: {exc}'
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return render_template(
        'inverse_z.html',
        default_num=num_txt,
        default_den=den_txt,
        default_roc=roc_txt,
        tf_latex=tf_ltx,
        hz_latex=hz_ltx,
        seq=seq,
        roc_used=roc_used,
        error=error
    )


if __name__ == "__main__":
    # simple smoke tests
    n = _parse_poly('[1]')
    d = _parse_poly('[1, -0.5]')
    expr_l = _inverse_z_expr(n, d, roc=0.1)
    assert 'Heaviside(-k - 1)' in str(expr_l)
    expr_r = _inverse_z_expr(n, d, roc=2)
    assert 'Heaviside(k)' in str(expr_r)
    seq = _impulse_response(n, d, n_samples=3)
    assert seq[0] == 1
    print('smoke tests passed')