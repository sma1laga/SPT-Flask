from flask import Blueprint, render_template, request
import numpy as np
import sympy as sp
from scipy.signal import lfilter
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
from sympy.printing.latex import LatexPrinter, accepted_latex_functions
import ast

inverse_z_bp = Blueprint('inverse_z', __name__)

# ──────────────────────────────────────────────────────────────────────────────
# utility to snap floats to ints when very close
def _int_if_close(val, tol=1e-12):
    if isinstance(val, complex):
        if abs(val.imag) < tol:
            val = val.real
        elif abs(val.real) < tol:
            val = val.imag
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
    z = sp.symbols('z', complex=True)
    expr = parse_expr(txt.replace('j','I'),
                      local_dict={'z':z},
                      transformations=_TRANSFORMS,
                      evaluate=False)
    coeffs = sp.Poly(sp.expand(expr), z).all_coeffs()
    return np.asarray([complex(c) for c in coeffs], dtype=complex)

def _coeffs_to_poly(coeffs):
    """Asc → poly:  [cn-1,cn-2,...,c1,c0]  →  cn-1 z^(n-1) + cn-2 z^(n-2) + ... + c1 z + c0"""
    z = sp.symbols('z', complex=True)
    n = len(coeffs)
    return sum(
        sp.sympify(_int_if_close(c))*z**(n-1-i)
        for i, c in enumerate(coeffs)
    )

# ──────────────────────────────────────────────────────────────────────────────
# direct‐filter impulse response (for display of first N samples)
def _impulse_response(num, den, n_samples=10, roc_type="causal"):
    if roc_type == "anticausal":
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
        if not isinstance(c, int): # round to 3 decimal places
            c = round(c, 3)
        seq.append(_int_if_close(c))
    return seq

def _inverse_z_expr(num, den, roc_type="causal"):
    "symbolic inverse z-transform: partial-fractions → table lookup"
    z = sp.symbols('z', complex=True)

    # helper:  coeff list (asc)  →  polynomial in z
    def _poly(vals):
        n = len(vals)
        return sum(sp.sympify(_int_if_close(c))*z**(n-1-i) for i,c in enumerate(vals))
    
    H_by_z = sp.simplify(_poly(num) / (z * _poly(den)))
    parts = sp.Add.make_args(sp.apart(H_by_z, z))       # simple pieces only
    parts = [p * z for p in parts] # redo div by z

    expr = sp.Integer(0)
    k = sp.symbols('k', integer=True)
    # ROC radius
    R = -1 if roc_type == "causal" else np.inf
    origin_flag = False # true if origin is a pole and ROC is anticausal
    for term in parts:
        # ----- δ[k+m] ---------------------------------------------------------
        if term.is_Pow and term.base == z or term.is_number:
            m = 0 if term.is_number else int(term.exp)
            if m < 0:
                if roc_type == "causal":
                    R = max(R, 0)
                else:
                    origin_flag = True
            C = sp.nsimplify(term.as_coeff_mul(z)[0])
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
        p_den = sp.Poly(den_t, z)
        if p_den.degree() != 1:
            raise ValueError(f"Only single poles are supported: {term}")

        a1, a0 = p_den.all_coeffs()      # a1 z + a0
        a = sp.nsimplify(-a0/a1)        # pole
        if roc_type == "causal": # update ROC radius
            R = max(R, sp.Abs(a))
        else:
            R = min(R, sp.Abs(a))
        num_t = sp.nsimplify(num_t/a1)   #   …and scale so denom becomes z-a

        p_num = sp.Poly(num_t, z)

        if p_num.degree() == 0:          # ----  A / (z-a)
            A = p_num.all_coeffs()[0]
            if roc_type == "causal":    # causal
                expr += A * a**(k-1) * sp.Heaviside(k-1)
            else:                        # left-sided
                expr += -A * a**(k-1) * sp.Heaviside(-k)
            continue

        if p_num.degree() == 1 and p_num.all_coeffs()[-1] == 0:
            B = p_num.all_coeffs()[0]    # ----  B z / (z-a)
            if roc_type == "causal":
                expr += B * a**k * sp.Heaviside(k)
            else:
                expr += -B * a**k * sp.Heaviside(-k-1)
            continue

        raise ValueError(f"Cannot invert term {term!r}")
    parts_sum = sum([sp.nsimplify(p) for p in parts])
    return expr, parts_sum, R, origin_flag

def _roc_latex(roc_type:str, roc_radius, exclude_origin:bool) -> str:
    """Generate LaTeX string for the ROC description."""
    if roc_type == "causal":
        if roc_radius >= 0:
            return rf"\left|z\right| > {roc_radius}"
        else:
            return r"z\in \mathbb{C}"
    else:
        if roc_radius == np.inf:
            if exclude_origin:
                return r"z\in \mathbb{C} \setminus \{0\}"
            else:
                return r"z\in \mathbb{C}"
        elif exclude_origin:
            return r"z\in\{z|\left|z\right| < %s\} \setminus \{0\}" % roc_radius
        else:
            return rf"\left|z\right| < {roc_radius}"

class CustomLatexPrinter(LatexPrinter):
    def _print_Function(self, expr):
        name = expr.func.__name__
        pargs = ', '.join(self._print(arg) for arg in expr.args)
        if name in accepted_latex_functions:
            return rf"\{name}\left({pargs}\right)"
        return rf"{name}\left[{pargs}\right]"

    def _print_Heaviside(self, expr, exp=None):
        pargs = ', '.join(self._print(arg) for arg in expr.pargs)
        tex = r"\varepsilon\left[%s\right]" % pargs
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex
    
    def _print_DiracDelta(self, expr, exp=None):
        if len(expr.args) == 1 or expr.args[1] == 0:
            tex = r"\delta\left[%s\right]" % self._print(expr.args[0])
        else:
            tex = r"\delta^{\left( %s \right)}\left[ %s \right]" % (
                self._print(expr.args[1]), self._print(expr.args[0]))
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex


# ──────────────────────────────────────────────────────────────────────────────
@inverse_z_bp.route('/', methods=['GET','POST'])
def inverse_z():
    num_txt = request.form.get('numerator','[1,0]')
    den_txt = request.form.get('denominator','[1]')
    roc_type = request.form.get('roc_type','causal')
    sf_ltx = sf_parts_ltx = hk_ltx = roc_ltx = seq = error = None

    if request.method == 'POST':
        try:
            num = _parse_poly(num_txt)
            den = _parse_poly(den_txt)

            # LaTeX for H(z)
            num_expr = _coeffs_to_poly(num)
            den_expr = _coeffs_to_poly(den)
            sf_ltx = sp.latex(num_expr/den_expr)

            expr, parts, roc_radius, exclude_origin = _inverse_z_expr(num, den, roc_type)

            roc_ltx = _roc_latex(roc_type, roc_radius, exclude_origin)
            
            sf_parts_ltx = CustomLatexPrinter().doprint(parts)
            if sf_ltx == sf_parts_ltx:
                sf_parts_ltx = None
            hk_ltx = CustomLatexPrinter().doprint(expr)

            # first 10 samples
            seq = _impulse_response(num, den,
                                    n_samples=10,
                                    roc_type=roc_type)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    return render_template(
        'inverse_z.html',
        default_num=num_txt,
        default_den=den_txt,
        sf_latex=sf_ltx,
        sf_parts_latex=sf_parts_ltx,
        hk_latex=hk_ltx,
        seq=seq,
        roc_type=roc_type,
        roc_latex=roc_ltx,
        error=error
    )