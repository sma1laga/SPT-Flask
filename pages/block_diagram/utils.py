"""
Utility helpers for the Block-Diagram module
────────────────────────────────────────────
• parse_poly()              – user string → coefficient list
• graph_to_coeffs()         – diagram → (num, den)
• coeffs_to_state_space()   – lists → (A,B,C,D)
• coeffs_to_ode_str()       – pretty differential / difference equation
"""
from typing import List, Tuple

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
    """
    Format a human-readable differential   (domain='s')
    or difference equation                (domain='z').
    """
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
        num, den = control.pade(tau, 1)
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
import re, networkx as nx, sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from control import tf2ss, TransferFunction

s, z = sp.symbols("s z", complex=True)

# ---- block → SymPy gain ----------------------------------------------
def gain_expr(node, domain="s"):
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
        num, den = control.pade(tau, 1)
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
def build_sfg(graph: dict):
    """Return (G, src_id, dst_id, domain)"""
    blocks = {n["id"]: n for n in graph["nodes"]}
    domain = graph.get("domain", "s")

    G = nx.DiGraph()
    for blk in blocks.values():
        G.add_node(blk["id"])

    for e in graph["edges"]:
        sign = 1 if e.get("sign", "+") == "+" else -1
        gain = gain_expr(blocks[e["from"]], domain) * sign
        G.add_edge(e["from"], e["to"], gain=gain)

    src = next(n["id"] for n in graph["nodes"]
               if n["type"] == "Input")
    dst = next(n["id"] for n in graph["nodes"] if n["type"] == "Output")
    return G, src, dst, domain
# ----------------------------------------------------------------------


# ---- Mason’s gain formula --------------------------------------------
def mason_gain(G: nx.DiGraph, src, dst):
    """Return SymPy expr of overall TF from src → dst."""
    all_paths  = list(nx.all_simple_paths(G, src, dst))
    loops      = list(nx.simple_cycles(G))

    def path_gain(path):
        g = 1
        for i in range(len(path)-1):
            g *= G[path[i]][path[i+1]]["gain"]
        return sp.simplify(g)

    def loop_gain(loop):
        g = 1
        for i in range(len(loop)):
            g *= G[loop[i]][loop[(i+1) % len(loop)] ]["gain"]
        return sp.simplify(g)

    P = [path_gain(p) for p in all_paths]
    L = [loop_gain(l) for l in loops]

    Δ = 1 - sum(L)                          # first-order; non-touching loops ignored
    T = sum(P) / Δ
    return sp.simplify(T)
# ----------------------------------------------------------------------


# ---- new top-level helper, replaces chain_to_coeffs -------------------
def sfg_to_coeffs(graph: dict):
    """Full SFG → (num, den) lists using Mason (handles ± adders & loops)."""
    G, src, dst, domain = build_sfg(graph)
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
