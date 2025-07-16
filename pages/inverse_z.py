from flask import Blueprint, render_template, request
import numpy as np
import sympy as sp
from scipy.signal import lfilter
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
import ast, re

inverse_z_bp = Blueprint('inverse_z', __name__)

# ──────────────────────────────────────────────────────────────────────────────
# utility to snap floats to ints when very close
def _int_if_close(val, tol=1e-12):
    if isinstance(val, complex):
        if abs(val.imag) < tol:
            val = val.real
        else:
            return val
    if isinstance(val, (float, np.floating, sp.Float)):
        if abs(val - round(val)) < tol:
            return int(round(val))
    return val

# ──────────────────────────────────────────────────────────────────────────────
# parsing a user-entered polynomial (either "[a, b, c]" or "a*z^2 + b*z + c")
_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor
)

def _parse_poly(txt: str) -> np.ndarray:
    txt = txt.strip()
    if txt.startswith('['):
        try:
            return np.asarray(ast.literal_eval(txt), dtype=complex)
        except Exception:
            items = txt.strip('[]')
            parts = [p.strip() for p in items.split(',') if p.strip()]
            vals = [complex(parse_expr(p.replace('j','I'),
                                      transformations=_TRANSFORMS))
                    for p in parts]
            return np.asarray(vals, dtype=complex)
    # else treat as symbolic expression in z
    z = sp.symbols('z')
    expr = parse_expr(txt.replace('j','I'),
                      local_dict={'z':z},
                      transformations=_TRANSFORMS,
                      evaluate=False)
    coeffs = sp.Poly(sp.expand(expr), z).all_coeffs()   # descending powers
    coeffs = coeffs[::-1]                                # reverse to ascending
    return np.asarray([complex(c) for c in coeffs], dtype=complex)

def _coeffs_to_poly(coeffs):
    """Asc → poly:  [c0,c1,…]  →  c0 + c1 z + …"""
    z = sp.symbols('z')
    return sum(
        sp.sympify(_int_if_close(c))*z**i
        for i, c in enumerate(coeffs)
    )

# ──────────────────────────────────────────────────────────────────────────────
# direct‐filter impulse response (for display of first N samples)
def _impulse_response(num, den, n_samples=10, roc_type="outside"):
    if roc_type == "inside":
        return None
    x = np.zeros(n_samples, dtype=complex)
    x[0] = 1
    y = lfilter(num, den, x)
    seq = []
    for v in y:
        c = complex(v)
        if abs(c.imag) < 1e-12:
            c = float(c.real)
            if c.is_integer():
                c = int(c)
        seq.append(_int_if_close(c))
    return seq

# ──────────────────────────────────────────────────────────────────────────────
# the core: symbolic inverse Z-transform via partial‐fractions + table lookup
_Wz, _k = sp.symbols('z k', integer=True)

# ──────────────────────────────────────────────────────────────────────────────
# symbolic inverse Z-transform: partial-fractions → table lookup
_Wz, _k = sp.symbols('z k', integer=True)

def _inverse_z_expr(num, den, roc=None, roc_type="inside"):
    z, k = _Wz, _k

    # helper:  coeff list (asc)  →  polynomial in z
    def _poly(vals):
        return sum(sp.sympify(_int_if_close(c))*z**i for i,c in enumerate(vals))

    H = sp.simplify(_poly(num) / _poly(den))
    parts = sp.Add.make_args(sp.apart(H, z))       # simple pieces only

    expr = sp.Integer(0)

    for term in parts:
        # ----- δ[k+m] ---------------------------------------------------------
        if term.is_Pow and term.base == z:
            m = int(term.exp)
            C = sp.nsimplify(_int_if_close(term.as_coeff_mul(z)[0]))
            expr += C * sp.DiracDelta(k + m)
            continue

        # ----- polynomial term  P(z) = Σ c_m z^m  ----------------------------
        # apart() puts every polynomial monomial into one "term" (den==1)
        num_t, den_t = term.as_numer_denom()
        if den_t == 1:
            poly = sp.Poly(num_t, z)           # coefficients in DESC order
            for exp, coef in zip(poly.monoms(), poly.coeffs()):
                m = exp[0]                     # exponent of z
                C = sp.nsimplify(_int_if_close(coef))
                expr  += C * sp.DiracDelta(k + m)
            continue

        # ----- first-order rational piece ------------------------------------
        num_t, den_t = term.as_numer_denom()
        p_den = sp.Poly(den_t, z)
        if p_den.degree() != 1:
            raise ValueError(f"Only simple poles are supported: {term}")

        a1, a0 = p_den.all_coeffs()      # a1 z + a0
        a  = sp.nsimplify(-a0/a1)        # pole
        num_t = sp.nsimplify(num_t/a1)   #   …and scale so denom becomes z-a

        p_num = sp.Poly(num_t, z)

        if p_num.degree() == 0:          # ----  A / (z-a)
            A = p_num.all_coeffs()[0]
            if roc_type == "outside":    # causal
                expr += A * a**(k-1) * sp.Heaviside(k-1)
            else:                        # left-sided
                expr += -A * a**(k-1) * sp.Heaviside(-k-1)
                expr += -A/a * sp.DiracDelta(k)      # ●●● add this line
            continue

        if p_num.degree() == 1 and p_num.all_coeffs()[-1] == 0:
            B = p_num.all_coeffs()[0]    # ----  B z / (z-a)
            if roc_type == "outside":
                expr += B * a**k * sp.Heaviside(k)
            else:
                expr += -B * a**k * sp.Heaviside(-k-1)
            continue

        raise ValueError(f"Cannot invert term {term!r}")

    return sp.simplify(expr)


# ──────────────────────────────────────────────────────────────────────────────
@inverse_z_bp.route('/', methods=['GET','POST'])
def inverse_z():
    num_txt = request.form.get('numerator','[1,0]')
    den_txt = request.form.get('denominator','[1]')
    roc_txt = request.form.get('roc_radius','')
    roc_type = request.form.get('roc_type','inside')
    tf_ltx = hz_ltx = seq = error = None
    roc_used = 'auto'

    if request.method == 'POST':
        try:
            num = _parse_poly(num_txt)
            den = _parse_poly(den_txt)
            roc_val = None
            if roc_txt.strip():
                roc_val = float(roc_txt)
                roc_used = roc_val

            # LaTeX for H(z)
            num_expr = _coeffs_to_poly(num)
            den_expr = _coeffs_to_poly(den)
            tf_ltx = sp.latex(num_expr/den_expr)

            # our new inverse‐Z implementation
            expr = _inverse_z_expr(num, den,
                                   roc=roc_val,
                                   roc_type=roc_type)
            hz_ltx = sp.latex(expr).replace('\\theta','\\varepsilon')

            # clean up trailing “.0”
            tf_ltx = re.sub(r'(\d)\.0(?!\d)', r'\1', tf_ltx)
            hz_ltx = re.sub(r'(\d)\.0(?!\d)', r'\1', hz_ltx)

            # first‐10 samples
            seq = _impulse_response(num, den,
                                    n_samples=10,
                                    roc_type=roc_type)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    return render_template(
        'inverse_z.html',
        default_num=num_txt,
        default_den=den_txt,
        default_roc=roc_txt,
        tf_latex=tf_ltx,
        hz_latex=hz_ltx,
        seq=seq,
        roc_used=roc_used,
        roc_type=roc_type,
        error=error
    )

if __name__ == "__main__":
    # quick smoke‐test
    n = _parse_poly('[3]')
    d = _parse_poly('[ -2, 1 ]')   # 3/(z-2)
    print("X5:", _inverse_z_expr(n,d, roc=2, roc_type="outside"))
    # expect: 3*2^(k-1)*Heaviside(k-1)
