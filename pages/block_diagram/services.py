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
    gain_expr,
    s,
    z,
)
import sympy as sp
from sympy import Matrix, Poly
import re
import control
import networkx as nx



def compile_diagram(graph_json: dict, *, domain: str = "s") -> dict:

    # 1) build the directed graph + locate source & sink
    G, src_id, dst_id, domain = build_sfg(graph_json)

    # 2) overall loop TF Y(s)/X(s)
    loop_tf_expr = mason_gain(G, src_id, dst_id)

    # 3) source TF X(s)
    src_node = next(n for n in graph_json["nodes"] if n["id"] == src_id)
    X_expr = gain_expr(src_node, domain)

    # 4) output TF H(s) and display-only Y(s)
    H_expr = sp.simplify(loop_tf_expr / X_expr)   # <-- divide out the source block
    Y_expr = sp.simplify(H_expr * X_expr)     # for LaTeX display only

    # Use the canonical symbol from utils (s or z) and coeff helper
    var = s if domain == "s" else z
    def expr_to_coeffs(e):
        n, d = sp.fraction(sp.together(e))
        return [float(c) for c in Poly(n, var).all_coeffs()], \
            [float(c) for c in Poly(d, var).all_coeffs()]


    numH, denH = expr_to_coeffs(H_expr)
    if len(numH) - 1 > len(denH) - 1:
        raise ValueError("Non-proper loop-TF; adjust diagram.")

    # For display
    loop_num, loop_den = expr_to_coeffs(loop_tf_expr)  # this is X(s)*H(s)
    in_num,  in_den    = expr_to_coeffs(X_expr)

    # For simulation (use H(s))
    out_num, out_den   = numH, denH


    # 6) state-space of the loop TF
    A, B, C, D = coeffs_to_state_space(out_num, out_den)

    # 7) human-readable ODE of the loop
    ode_string = coeffs_to_ode_str(out_num, out_den, domain)


    # generate LaTeX for SS matrices and ODE
    from sympy import Matrix, latex

    # turn your numpy arrays into Sympy matrices
    A_mat = Matrix(A)
    B_mat = Matrix(B)
    C_mat = Matrix(C)
    D_mat = Matrix(D)

    # now call sympy.latex
    ss_latex = (
        r"\dot x = " + latex(A_mat) + r" \,x + " + latex(B_mat) + r" \,u \\[6pt]" +
        r"y = "      + latex(C_mat) + r" \,x + " + latex(D_mat) + r" \,u"
    )

    ode_latex = coeffs_to_ode_str(loop_num, loop_den, domain)  # ideally wrap with proper braces

    # Detect a saturation block along a path from source to sink. The
    # scope block is linear (unity gain) so it does not influence the
    # transfer-function but the saturation limits are required fortime-domain simulation
    sat_node = None
    for n in graph_json.get("nodes", []):
        if n.get("type") == "Saturation":
            node_id = n["id"]
            if nx.has_path(G, src_id, node_id) and nx.has_path(G, node_id, dst_id):
                sat_node = n
                break

    saturation = None
    if sat_node:
        p = sat_node.get("params", {})
        try:
            lower = float(p.get("lower"))
        except (TypeError, ValueError):
            lower = None
        try:
            upper = float(p.get("upper"))
        except (TypeError, ValueError):
            upper = None
        saturation = {"lower": lower, "upper": upper}
    # compute transfer-functions for al Scope blocks
    scope_tfs = {}
    for n in graph_json.get("nodes", []):
        if n.get("type") == "Scope":
            sid = n["id"]
            if not nx.has_path(G, src_id, sid):
                continue
            # H_scope(s) = (source to scope) / X(s)
            scope_expr = sp.simplify(mason_gain(G, src_id, sid) / X_expr)
            sn, sd = expr_to_coeffs(scope_expr)
            if len(sn) - 1 > len(sd) - 1:
                continue
            scope_tfs[str(sid)] = {"num": sn, "den": sd, "latex": sp.latex(scope_expr)}

    return {
        "loop_tf":   {"num": loop_num, "den": loop_den, "latex": sp.latex(loop_tf_expr)},
        "input_tf":  {"num": in_num,   "den": in_den,   "latex": sp.latex(X_expr)},
        "output_tf": {"num": out_num,  "den": out_den,  "latex": sp.latex(H_expr)},  # H(s)
        "y_signal_latex": sp.latex(Y_expr),  # optional, for display only
        "state_space": {
             "A": A.tolist(),
             "B": B.tolist(),
             "C": C.tolist(),
             "D": D.tolist(),
             "latex": ss_latex
             },
        "ode": ode_string,
        "ode_latex": ode_latex,
        "saturation": saturation,
        "scopes": scope_tfs,
     }