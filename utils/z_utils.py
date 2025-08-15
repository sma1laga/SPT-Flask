import sympy as sp
import numpy as np
from scipy.signal import residue
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application, convert_xor
)
from sympy.printing.latex import LatexPrinter, accepted_latex_functions
import re

import utils.sympy_utils as sp_utils

CONSTANTS = {
    "e": sp.E,
    "pi": sp.pi,
}
SYMBOLS_ALLOWED = {
    "z": sp.Symbol("z", complex=True),
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

H0 = sp.Integer(1) # Value of Heaviside(0)

# ──────────────────────────────────────────────────────────────────────────────
# parsing a user-entered polynomial (either "[a, b, c]" or "a*z^2 + b*z + c")
_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor
)

def parse_input(txt: str) -> sp.Expr:
    txt = txt.strip()
    if len(txt) > 100:
        raise ValueError("Input too long")
    if txt.startswith('['): # polynomial from coefficient list
        items = txt.strip('[]')
        parts = [p.strip() for p in items.split(',') if p.strip()]
        coeffs = [
            sp.parse_expr(
                p,
                evaluate=False,
                local_dict=sp_utils.rm_keys(SYMBOLS_ALLOWED, ["z"]), # without z
                global_dict={},
                transformations=_TRANSFORMS,
            )
            for p in parts
        ]
        return sp_utils.coeffs_to_poly(coeffs, sp.Symbol("z", complex=True))
    
    # any other expression
    expr = sp.parse_expr(
        txt,
        evaluate=False,
        local_dict=SYMBOLS_ALLOWED,
        global_dict={},
        transformations=_TRANSFORMS,
    )
    return expr

def inverse_z_expr(num, den, roc_type="causal"):
    "symbolic inverse z-transform: partial-fractions → table lookup"
    z = sp.symbols('z', complex=True)
    k = sp.symbols('k', integer=True)
    # H(z)/z
    H_by_z = sp.simplify(sp_utils.coeffs_to_poly(num, z) / (z * sp_utils.coeffs_to_poly(den, z)))
    num_H_by_z, den_H_by_z = H_by_z.as_numer_denom()
    r, p, kvals = residue(
        [complex(c) for c in num_H_by_z.as_poly(z).all_coeffs()],
        [complex(c) for c in den_H_by_z.as_poly(z).all_coeffs()],
        tol=1e-4,
    )
    expr_pfd = sp.Integer(0)
    if kvals.size > 0:
        expr_pfd = sp_utils.coeffs_to_poly(kvals, z)
    expr_pfd_parts = list(sp.Add.make_args(expr_pfd))
    pi_buf = np.nan
    pole_count = 1
    expr_buf = sp.Integer(0)
    for ri, pi in zip(r, p):
        if np.isclose(pi, pi_buf):
            pole_count += 1
        else:
            pole_count = 1
        ri_expr = sp_utils.render_number(ri, complex_polar=True)
        pi_expr = sp_utils.render_number(pi, complex_polar=True)

        if pole_count == 3: # sum up all previous parts with same 
            term_2 = expr_pfd_parts[-1] * (z - pi_expr) ** 3
            term_1 = expr_pfd_parts[-2] * (z - pi_expr) ** 3
            expr_buf = sp.simplify(term_1 + term_2 + ri_expr) / (z - pi_expr) ** pole_count
            expr_pfd_parts = expr_pfd_parts[:-2] # remove last two parts
        else:
            expr_buf = ri_expr / (z - pi_expr) ** pole_count
        expr_pfd_parts.append(expr_buf)
        pi_buf = pi

    expr_pfd = sum([pf * z for pf in expr_pfd_parts]) # H(z)/z * z

    def _only_poles_at_zero(p: sp.Poly) -> bool:
        """Check if all poles of polynomial term are at zero."""
        # p should only have a non-zero coefficient at the highest exponent
        if p.degree() == 0:
            return True
        return all(c == 0 for c in p.all_coeffs()[1:])
    
    expr_k = sp.Integer(0)
    # determine ROC radius
    R = -1 if roc_type == "causal" else np.inf
    origin_flag = False # true if origin is a pole and ROC is anticausal
    poles = list(sp.roots(sp_utils.coeffs_to_poly(den, z), z))
    poles_abs = [sp.Abs(p) for p in poles]
    if roc_type == "causal":
        if len(poles) > 0:
            R = max(poles_abs)
    else: # anticausal
        if len(poles) > 0:
            R = min(poles_abs)
            if 0 in poles_abs:
                origin_flag = True
    
    for term in sp.Add.make_args(expr_pfd):
        num_t, den_t = term.as_numer_denom()
        # ----- c ⋅ δ[k ± m] ---------------------------------------------------------
        if _only_poles_at_zero(den_t.as_poly(z)):
            if den_t.is_number: # regular polynomial
                term_poly = term.as_poly(z)
                m = term_poly.degree()
                c = term_poly.LC()
            else: # m-degree pole at zero, form: c0 / (c1 z^m)
                m = - den_t.as_poly(z).degree()
                c = (num_t / den_t.as_poly(z).LC())
            expr_k += sp_utils.render_number(c) * sp.DiracDelta(k + m)
            continue
        p_num = num_t.as_poly(z)
        p_den = den_t.as_poly(z)
        # ----- first-order pole ------------------------------------------------
        if p_den.degree() == 1:
            # form: c0 z / (c1(z - a))
            a = list(sp.roots(p_den, z))[0]
            if roc_type == "causal": # update ROC radius
                R = max(R, sp.Abs(a))
            else:
                R = min(R, sp.Abs(a))
            assert 0 in sp.roots(p_num, z), "Numerator should have a root at zero"
            c = sp_utils.render_number(p_num.LC() / p_den.LC())
            a = sp_utils.render_number(a, complex_polar=True)
            if roc_type == "causal":
                expr_k += c * a**k * sp.Heaviside(k, H0)
            else:
                expr_k -= c * a**k * sp.Heaviside(-k-1, H0)
        # ----- second/third order pole ------------------------------------
        if p_den.degree() > 1:
            if p_den.degree() > 3:
                raise ValueError(f"At most triple poles are supported")
            # causal double/triple pole
            if p_den.degree() == 2:
                # form: (c0 z) / (c1(z - a)**2) to k a**k step[k]
                roots = sp.roots(p_den, z)
                assert len(roots) == 1, "Only double poles are supported"
                a = list(roots)[0]
                # c1 / c0 / a
                c = sp_utils.render_number(p_num.LC() / p_den.LC() / a)
                a = sp_utils.render_number(a, complex_polar=True)
                if roc_type == "causal":
                    expr_k += c * k * a**k * sp.Heaviside(k, H0)
                else:
                    expr_k -= c * k * a**k * sp.Heaviside(-k, H0)
            else: # deg == 3
                # form: (c0 z(z+a)) / (c1(z - a)**3) to c1/c0 * a k**2 a**k step[k]
                roots = sp.roots(p_den, z)
                assert len(roots) == 1, "Only triple poles are supported"
                a = list(roots)[0]
                # num roots should be 0 and -a
                roots_num = sp.roots(p_num, z)
                assert len(roots_num) == 2, "Numerator should have two roots"
                if -a not in roots_num:
                    raise ValueError(f"Numerator {num_t} not matching triple pole at {a}")
                c = sp_utils.render_number(p_num.LC() / p_den.LC() / a)
                a = sp_utils.render_number(a, complex_polar=True)
                if roc_type == "causal":
                    expr_k += c * k**2 * a**k * sp.Heaviside(k, H0)
                else:
                    expr_k -= c * k**2 * a**k * sp.Heaviside(-k, H0)

    expr_k = sp.simplify(expr_k)
    if R != np.inf:
        R = sp_utils.render_number(R)
    return expr_k, expr_pfd, R, origin_flag

def roc_latex(roc_type:str, roc_radius, exclude_origin:bool) -> str:
    """Generate LaTeX string for the ROC description."""
    ltx_printer = DiscreteLatexPrinter()
    roc_radius_ltx = ltx_printer.doprint(roc_radius, simplify_expr=False)
    if roc_type == "causal":
        if roc_radius >= 0:
            return rf"\left|z\right| > {roc_radius_ltx}"
        else:
            return r"z\in \mathbb{C}"
    else:
        if roc_radius == np.inf:
            if exclude_origin:
                return r"z\in \mathbb{C} \setminus \{0\}"
            else:
                return r"z\in \mathbb{C}"
        elif exclude_origin:
            return r"z\in\{z|\left|z\right| < %s\} \setminus \{0\}" % roc_radius_ltx
        else:
            return rf"\left|z\right| < {roc_radius_ltx}"

def eval_expression(expr: sp.Expr, in_vals: np.ndarray, in_symbol: sp.Symbol) -> list:
    """Evaluate a SymPy expression with given input values."""
    float_th = 10
    if expr.is_number: # constant functions
        return [sp_utils.render_number(expr, th_float=float_th)] * len(in_vals)
    # turn expr into a function without lambdify
    y = []
    for val in in_vals:
        out_val = expr.subs(in_symbol, val)
        y.append(sp_utils.render_number(out_val, th_float=float_th))
    return y


class DiscreteLatexPrinter(LatexPrinter):
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

        # Collapse repeated factors and powers of step[k]
        latex = re.sub(r"(\\left)?\((\\varepsilon\[k\])(\\right)?\)\^(\{)?(\d+)(\})?", r"\2", latex)
        latex = re.sub(r"(\\varepsilon\[k\]\s*)+", r"\1", latex)

        # Remove stray \cdot before step[k]
        latex = latex.replace(r"\cdot \varepsilon\[k\]", r"\varepsilon\[k\]")
        def mround(match):
            return str(np.round(float(match.group()), 4))
        latex = re.sub(r"(\d+\.\d{5,})", mround, latex)
        return latex

    def _print_Function(self, expr):
        name = expr.func.__name__
        pargs = ', '.join(self._print(arg) for arg in expr.args)
        inv_trig_table = [
            "asin", "acos", "atan",
            "acsc", "asec", "acot",
            "asinh", "acosh", "atanh",
            "acsch", "asech", "acoth",
        ]
        if name in accepted_latex_functions:
            return rf"\{name}\left({pargs}\right)"
        if name in inv_trig_table:
            return rf"\operatorname{{{name}}}\left({pargs}\right)"
        return rf"{name}\left[{pargs}\right]"
    
    def _print_Heaviside(self, expr, exp=None):
        tex = r"\varepsilon\left[%s\right]" % self._print(expr.args[0])
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
    
    def _print_Exp1(self, expr, exp=None):
        return r"\mathrm{e}"
    
    def _print_ExpBase(self, expr, exp=None):
        tex = r"\mathrm{e}^{%s}" % self._print(expr.args[0])
        return self._do_exponent(tex, exp)