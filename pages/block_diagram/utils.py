"""
Utility helpers for the Block-Diagram module
────────────────────────────────────────────
• parse_poly()              – user string → coefficient list
• graph_to_coeffs()         – diagram → (num, den)
• coeffs_to_state_space()   – lists → (A,B,C,D)
• coeffs_to_ode_str()       – pretty differential / difference equation
"""
from typing import List

import sympy as sp
import control  
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
from control.matlab import zpk2tf
import re

# ────────────────────────────────────────────────────────────────────────
def parse_poly(poly_str: str) -> List[float]:
    """
    Turn  's^2 + 2'  →  [1, 0, 2]
    Accepts variable  s  or  z.  Raises ValueError on syntax errors.
    """
    poly_str = poly_str.replace("^", "**")          # allow caret notation
    s, z = sp.symbols("s z", complex=True)
    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(poly_str, local_dict={"s": s, "z": z}, transformations=transformations)
    poly = sp.Poly(expr, s if "s" in poly_str else z)
    return [float(c) for c in poly.all_coeffs()]

def _parse_root_list(root_str: str) -> List[complex]:
    """Return list of roots from a comma/space separated string."""
    root_str = root_str.strip()
    if not root_str:
        return []
    parts = re.split(r"[\s,]+", root_str)
    roots: List[complex] = []
    for part in parts:
        if not part:
            continue
        # allow both 'j' and 'i' for imaginary unit
        part = part.replace("i", "j")
        roots.append(complex(sp.N(sp.sympify(part))))
    return roots



# ────────────────────────────────────────────────────────────────────────
def graph_to_coeffs(graph: dict):
    """
    • exactly **one** TF block allowed (for now)
    • any number of Gain blocks *in series* with it → multiply numerators
    • falls back to legacy System node if no TF present
    """
    tf_nodes = [n for n in graph["nodes"] if n["type"] == "TF" and
                n["params"].get("num") and n["params"].get("den")]

    if not tf_nodes:
        sys_node = next((n for n in graph["nodes"] if n["type"] == "System"), None)
        if not sys_node:
            raise ValueError("No transfer-function block found.")
        return sys_node["params"]["num"], sys_node["params"]["den"]

    if len(tf_nodes) > 1:
        raise NotImplementedError("Multiple TF blocks not supported yet.")

    # single TF
    num = parse_poly(tf_nodes[0]["params"]["num"])
    den = parse_poly(tf_nodes[0]["params"]["den"])

    # multiply by all Gain constants (series assumption)
    gain_nodes = [n for n in graph["nodes"] if n["type"] == "Gain" and
                  n["params"].get("k") is not None]
    k_total = 1.0
    for g in gain_nodes:
        k_total *= float(g["params"]["k"])
    num = [k_total * c for c in num]
    return num, den



# ────────────────────────────────────────────────────────────────────────
def coeffs_to_state_space(num: List[float], den: List[float]):
    """
    Use python-control to obtain the controllable canonical realisation.
    Returns the 4 numpy arrays (A,B,C,D).
    """
    sys = control.TransferFunction(num, den)
    ss = control.ss(sys)
    return ss.A, ss.B, ss.C, ss.D


# ────────────────────────────────────────────────────────────────────────
def coeffs_to_ode_str(num: List[float], den: List[float], domain: str = "s") -> str:
    """Return a textual differential/difference equation (non-latex)."""

    def poly_str(coeffs, var_symbol):
        deg = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            if c == 0:
                continue
            pwr = deg - i
            term = f"{c:g}"
            if pwr > 0:
                term += f"{var_symbol}"
                if pwr > 1:
                    term += f"^{pwr}"
            terms.append(term)
        return " + ".join(terms) if terms else "0"

    var = "D" if domain == "z" else "d/dt"
    left = poly_str(den, var) + " · y(t)"
    right = poly_str(num, var) + " · u(t)"
    return f"{left} = {right}"


def _format_coefficient(coef: float) -> sp.Expr:
    """Return a simplified SymPy expression for a numeric coefficient."""
    return sp.nsimplify(coef, rational=True)


def _term_sign_and_body(expr: sp.Expr, body: str, first: bool) -> str:
    """Combine a coefficient expression with its symbolic body."""
    if expr == 0:
        return ""

    sign = "-" if expr.is_negative else "+"
    magnitude = -expr if expr.is_negative else expr
    mag_str = "" if magnitude == 1 else sp.latex(magnitude)

    term_core = f"{mag_str}{body}" if mag_str else body
    if first:
        return term_core if sign == "+" else f"-{term_core}"
    return f" + {term_core}" if sign == "+" else f" - {term_core}"


def _derivative_symbol(var: str, order: int) -> str:
    """Return a variable marked with over-dot notation for its derivative."""

    if order == 0:
        return var

    combining_marks = {
        1: "\u0307",   # combining dot above → ẋ
        2: "\u0308",   # combining diaeresis → ẍ
    }

    mark = combining_marks.get(order)
    if mark is not None:
        return f"{var}{mark}"

    return rf"{var}^{{({order})}}"


def _difference_symbol(var: str, shift: int) -> str:
    index = "n" if shift == 0 else (f"n-{shift}" if shift > 0 else f"n+{-shift}")
    return rf"{var}[{index}]"


def _format_latex_sum(coeffs: List[float], *, domain: str, var: str) -> str:
    terms: List[str] = []
    if domain == "s":
        order = len(coeffs) - 1
        for idx, coef in enumerate(coeffs):
            expr = _format_coefficient(coef)
            if expr == 0:
                continue
            deriv_order = order - idx
            body = _derivative_symbol(var, deriv_order)
            term = _term_sign_and_body(expr, body, first=not terms)
            if term:
                terms.append(term)
    else:
        for shift, coef in enumerate(coeffs):
            expr = _format_coefficient(coef)
            if expr == 0:
                continue
            body = _difference_symbol(var, shift)
            term = _term_sign_and_body(expr, body, first=not terms)
            if term:
                terms.append(term)

    if not terms:
        return "0"
    result = "".join(terms)
    return result.strip()


def coeffs_to_ode_latex(num: List[float], den: List[float], domain: str = "s") -> str:
    """Return a LaTeX equation for the differential/difference form."""

    left = _format_latex_sum(den, domain=domain, var="y")
    right = _format_latex_sum(num, domain=domain, var="u")
    return rf"{left} = {right}"

# ────────────────────────────────────────────────────────────────────
#  Feed-forward chain helper  (Milestone M1)
# ────────────────────────────────────────────────────────────────────
import networkx as nx
import numpy as np
from control import TransferFunction

def tf_from_block(node):
    """Return a python-control TransferFunction for ONE block."""
    t = node["type"]
    p = node.get("params", {})

    if t == "TF":
        num  = parse_poly(p["num"])
        den  = parse_poly(p["den"])
        return TransferFunction(num, den)

    if t == "Gain":
        k = float(p.get("k", 1))
        return TransferFunction([k], [1])

    if t == "Integrator":        # 1/s
        return TransferFunction([1], [1, 0])
    
    if t == "Derivative":          #   s
        return TransferFunction([1, 0], [1])

    if t == "Delay":
        tau = float(p.get("tau", 0) or 0)
        if tau == 0:
            return TransferFunction([1], [1])
        try:
            order = int(p.get("pade_order", 1) or 1)
        except (TypeError, ValueError):
            order = 1
        order = max(order, 1)
        num, den = control.pade(tau, order)
        return TransferFunction(num, den)

    if t == "ZeroPole":
        zeros = _parse_root_list(p.get("zeros", ""))
        poles = _parse_root_list(p.get("poles", ""))
        k = float(p.get("k", 1) or 1)
        num, den = zpk2tf(zeros, poles, k)
        return TransferFunction(num, den)

    if t == "PID":
        kp = float(p.get("kp", 0) or 0)
        ki = float(p.get("ki", 0) or 0)
        kd = float(p.get("kd", 0) or 0)
        return TransferFunction([kd, kp, ki], [1, 0])

    if t == "Saturation" or t == "Scope":
        return TransferFunction([1], [1])

    # Input, Output, Adder – unity TF
    return TransferFunction([1], [1])


def chain_to_coeffs(graph: dict):
    """
    Multiply all blocks along the UNIQUE feed-forward path
    Input → … → Output.  Ignores branches & loops.
    """
    G = nx.DiGraph()
    for e in graph["edges"]:
        G.add_edge(e["from"], e["to"])

    # pick first Input and first Output block ids
    src = next(n["id"] for n in graph["nodes"] if n["type"] == "Input")
    dst = next(n["id"] for n in graph["nodes"] if n["type"] == "Output")

    try:
        path = nx.shortest_path(G, src, dst)
    except nx.NetworkXNoPath as exc:
        raise ValueError("No Input→Output path") from exc

    # multiply TFs along that path
    blocks = {n["id"]: n for n in graph["nodes"]}
    sys    = tf_from_block(blocks[path[0]])
    for blk_id in path[1:]:
        sys *= tf_from_block(blocks[blk_id])

    # return coefficient *lists* (python-control stores as np arrays)
    num = sys.num[0][0].tolist()
    den = sys.den[0][0].tolist()
    return num, den


# ───────── Mason-ready helpers  (add BELOW the existing imports) ───────
import re, itertools
import networkx as nx
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from control import tf2ss, TransferFunction

s, z = sp.symbols("s z", complex=True)

# ---- block → SymPy gain ----------------------------------------------
def gain_expr(node, domain="s", *, delay_model="pade"):
    t, p = node["type"], node.get("params", {})
    var  = s if domain == "s" else z

    if t == "TF":
        num = parse_expr(p["num"].replace("^","**"), local_dict={'s':s,'z':z})
        den = parse_expr(p["den"].replace("^","**"), local_dict={'s':s,'z':z})
        return num/den
    if t == "Gain":
        return sp.sympify(p.get("k", 1))
    if t == "Integrator":
        return 1/var
    if t == "Delay":
        tau = float(p.get("tau", 0) or 0)
        if tau == 0:
            return 1
        if delay_model == "exact":
            return sp.exp(-tau * var)
        try:
            order = int(p.get("pade_order", 1) or 1)
        except (TypeError, ValueError):
            order = 1
        order = max(order, 1)
        num, den = control.pade(tau, order)
        num_expr = sum(num[i] * var**(len(num)-1-i) for i in range(len(num)))
        den_expr = sum(den[i] * var**(len(den)-1-i) for i in range(len(den)))
        return num_expr/den_expr
    if t == "Derivative":
        return var
    if t == "ZeroPole":
        zeros = _parse_root_list(p.get("zeros", ""))
        poles = _parse_root_list(p.get("poles", ""))
        k = sp.sympify(p.get("k", 1) or 1)
        num_expr = sp.prod([var - z0 for z0 in zeros]) if zeros else 1
        den_expr = sp.prod([var - p0 for p0 in poles]) if poles else 1
        return k * num_expr / den_expr
    if t == "PID":
        kp = sp.sympify(p.get("kp", 0) or 0)
        ki = sp.sympify(p.get("ki", 0) or 0)
        kd = sp.sympify(p.get("kd", 0) or 0)
        return (kd*var**2 + kp*var + ki) / var
    if t in ("Mux", "Demux"):
        return 1
    if t in ("Source", "Input"):
        kind = p.get("kind", "step")
        if kind == "impulse": return 1
        if kind == "step":    return 1/var
        if kind == "custom":
            n = parse_expr(p["num"], local_dict={'s':s,'z':z})
            d = parse_expr(p["den"], local_dict={'s':s,'z':z})
            return n/d
    return 1  # Adder, Output, default
# ----------------------------------------------------------------------


# ---- build directed signal-flow graph --------------------------------
def build_sfg(graph: dict, *, delay_model="pade"):
    """Return (G, src_id, dst_id, domain)"""
    blocks = {n["id"]: n for n in graph["nodes"]}
    domain = graph.get("domain", "s")

    G = nx.DiGraph()
    for blk in blocks.values():
        G.add_node(blk["id"])

    for e in graph["edges"]:
        sign = 1 if e.get("sign", "+") == "+" else -1
        gain = gain_expr(blocks[e["from"]], domain, delay_model=delay_model) * sign
        G.add_edge(e["from"], e["to"], gain=gain)

    src = next(n["id"] for n in graph["nodes"]
               if n["type"] == "Input")
    dst = next(n["id"] for n in graph["nodes"] if n["type"] == "Output")
    return G, src, dst, domain
# ----------------------------------------------------------------------


# ---- Mason’s gain formula --------------------------------------------
def mason_gain(G: nx.DiGraph, src, dst):
    """Return SymPy expr of overall TF from src → dst."""


    def path_gain(path):
        gain = sp.Integer(1)
        for idx in range(len(path) - 1):
            gain *= G[path[idx]][path[idx + 1]]["gain"]
        return sp.simplify(gain)

    def loop_gain(loop):
        gain = sp.Integer(1)
        for idx in range(len(loop)):
            gain *= G[loop[idx]][loop[(idx + 1) % len(loop)]]["gain"]
        return sp.simplify(gain)

    all_paths = list(nx.all_simple_paths(G, src, dst))
    if not all_paths:
        raise ValueError("No forward path from source to sink in diagram.")

    loops = list(nx.simple_cycles(G))
    loop_info = [
        {
            "nodes": set(loop),
            "gain": loop_gain(loop),
        }
        for loop in loops
    ]

    def _non_touching_products(loops_subset):
        terms_by_order = {}
        n = len(loops_subset)
        for r in range(1, n + 1):
            products = []
            for combo in itertools.combinations(range(n), r):
                sets = [loops_subset[i]["nodes"] for i in combo]
                if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
                    continue
                prod = sp.Integer(1)
                for i in combo:
                    prod *= loops_subset[i]["gain"]
                products.append(sp.simplify(prod))
            if products:
                terms_by_order[r] = products
        return terms_by_order

    def _delta(loops_subset):
        delta = sp.Integer(1)
        combos = _non_touching_products(loops_subset)
        for order, products in combos.items():
            term_sum = sum(products)
            delta += ((-1) ** order) * term_sum
        return sp.simplify(delta)

    Δ = _delta(loop_info)
    if sp.simplify(Δ) == 0:
        raise ValueError("Ill-posed diagram: algebraic loop without dynamics.")

    total = sp.Integer(0)
    for path in all_paths:
        gain = path_gain(path)
        path_nodes = set(path)
        loops_disjoint = [loop for loop in loop_info if loop["nodes"].isdisjoint(path_nodes)]
        Δ_path = _delta(loops_disjoint)
        total += gain * Δ_path

    T = sp.simplify(total / Δ)
    return T
# ----------------------------------------------------------------------


# ---- new top-level helper, replaces chain_to_coeffs -------------------
def sfg_to_coeffs(graph: dict, *, delay_model="pade"):
    """Full SFG → (num, den) lists using Mason (handles ± adders & loops)."""
    G, src, dst, domain = build_sfg(graph, delay_model=delay_model)
    tf_expr = mason_gain(G, src, dst)

    var = s if domain == "s" else z
    num, den = sp.fraction(tf_expr)
    num = sp.Poly(num, var).all_coeffs()
    den = sp.Poly(den, var).all_coeffs()
    # convert SymPy numbers → float
    num = [float(c) for c in num]
    den = [float(c) for c in den]
    return num, den
# ----------------------------------------------------------------------
