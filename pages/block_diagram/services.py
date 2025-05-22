"""
High-level orchestration
────────────────────────
Wrap small helper functions in utils.py / 3rd-party control libs.
The heavy lifting lives in utils.py so you can unit-test it easily.
"""
from .utils import (
    sfg_to_coeffs,
    coeffs_to_state_space,
    coeffs_to_ode_str,
)
import sympy as sp
from sympy import Matrix
import re


def compile_diagram(graph_json: dict, *, domain: str = "s") -> dict:
    """
    Parameters
    ----------
    graph_json : dict
        { "nodes":[...], "edges":[...], "domain":"s"|"z", ... }
    domain : str
        Explicit override ("s" for Laplace, "z" for discrete)

    Returns
    -------
    dict  (ready for JSONify)
    """
    # 1) translate graph → polynomial coefficient lists
    num_coeffs, den_coeffs = sfg_to_coeffs(graph_json)

    # 2) build symbolic transfer function (for pretty printing, LaTeX, …)
    var = sp.symbols(domain)
    # build a symbolic FRACTION instead of calling TransferFunction
    G_expr = sp.Poly(num_coeffs, var) / sp.Poly(den_coeffs, var)
    G_simplified = sp.together(G_expr)

    # 3) canonical state-space (controllable canonical form)
    A, B, C, D = coeffs_to_state_space(num_coeffs, den_coeffs)

    # 4) fancy outputs --------------------------------------------------
    ode_string  = coeffs_to_ode_str(num_coeffs, den_coeffs, domain)

    # latex for TF, ODE, and SS
    tf_latex  = sp.latex(G_simplified)
    ode_latex = (
    ode_string
      .replace("d/dt", "D")        # derivative operator
      .replace(" · ", "\\,")       # thin space for multiplication
    )
    ode_latex = re.sub(r'\^(\d+)', r'^{\1}', ode_latex)

    A_lx = sp.latex(Matrix(A))
    B_lx = sp.latex(Matrix(B))
    C_lx = sp.latex(Matrix(C))
    D_lx = sp.latex(Matrix(D))
    ss_latex = (
        "\\dot{x}=" + A_lx + "x+" + B_lx + "u\\\\[6pt]"
        "y="        + C_lx + "x+" + D_lx + "u"
    )

    return {
         "transfer_function": {
             "num": num_coeffs,
             "den": den_coeffs,
             "str": str(G_simplified),
             "latex": tf_latex
             },
        "state_space": {
             "A": A.tolist(),
             "B": B.tolist(),
             "C": C.tolist(),
             "D": D.tolist(),
             "latex": ss_latex
             },
        "ode": ode_string,
        "ode_latex": ode_latex
     }