"""
High-level orchestration
────────────────────────
Wrap small helper functions in utils.py / 3rd-party control libs.
The heavy lifting lives in utils.py so you can unit-test it easily.
"""
from .utils import (
    build_sfg,
    mason_gain,
    coeffs_to_state_space,
    coeffs_to_ode_str,
)
import sympy as sp
from sympy import Matrix, Poly, symbols
import re
import control


def compile_diagram(graph_json: dict, *, domain: str = "s") -> dict:

    # 1) build the directed graph + locate source & sink
    G, src_id, dst_id, domain = build_sfg(graph_json)

    # 2) overall loop TF Y(s)/X(s)
    loop_tf_expr = mason_gain(G, src_id, dst_id)

    # 3) source TF X(s)
    from .utils import gain_expr
    X_expr = gain_expr({ "type":"Source", 
                         "params":next(n for n in graph_json["nodes"]
                                      if n["id"]==src_id)["params"] },
                       domain)

    # 4) output TF Y(s)
    Y_expr = sp.simplify( loop_tf_expr * X_expr )

    var = symbols(domain)
    def expr_to_coeffs(e):
        n,d = sp.fraction(e)
        return [float(c) for c in Poly(n,var).all_coeffs()],\
               [float(c) for c in Poly(d,var).all_coeffs()]

    # properness guard on loop TF only (optional for X or Y too)
    numL, denL = expr_to_coeffs(loop_tf_expr)
    num_deg = Poly(numL, var).degree(); den_deg = Poly(denL, var).degree()
    if num_deg > den_deg:
        raise ValueError("Non-proper loop-TF; adjust diagram.")

    # 5) assemble all three
    loop_num, loop_den = numL, denL
    in_num,  in_den  = expr_to_coeffs(X_expr)
    out_num, out_den = expr_to_coeffs(Y_expr)

    # 6) state-space of the loop TF
    A, B, C, D = coeffs_to_state_space(loop_num, loop_den)

    # 7) human-readable ODE of the loop
    ode_string = coeffs_to_ode_str(loop_num, loop_den, domain)


    # generate LaTeX for SS matrices and ODE
    from sympy import Matrix, latex

    # turn your numpy arrays into Sympy matrices
    A_mat = Matrix(A)
    B_mat = Matrix(B)
    C_mat = Matrix(C)
    D_mat = Matrix(D)

    # now call sympy.latex(...)
    ss_latex = (
        r"\dot x = " + latex(A_mat) + r" \,x + " + latex(B_mat) + r" \,u \\[6pt]" +
        r"y = "      + latex(C_mat) + r" \,x + " + latex(D_mat) + r" \,u"
    )

    ode_latex = coeffs_to_ode_str(loop_num, loop_den, domain)  # ideally wrap with proper braces

    return {
        "loop_tf":    { "num": loop_num, "den": loop_den,
                        "latex": sp.latex(loop_tf_expr) },
        "input_tf":   { "num": in_num,  "den": in_den,  
                        "latex": sp.latex(X_expr) },
        "output_tf":  { "num": out_num, "den": out_den,
                        "latex": sp.latex(Y_expr) },
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