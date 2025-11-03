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
import numpy as np



def compile_diagram(graph_json: dict, *, domain: str = "s") -> dict:

    # 1) build the directed graph + locate source & sink
    G, src_id, dst_id, domain = build_sfg(graph_json)
    G_exact, _, _, _ = build_sfg(graph_json, delay_model="exact")

    # 2) overall loop TF Y(s)/X(s)
    loop_tf_expr = mason_gain(G, src_id, dst_id)
    loop_tf_exact_expr = mason_gain(G_exact, src_id, dst_id)

    # 3) source TF X(s)
    src_node = next(n for n in graph_json["nodes"] if n["id"] == src_id)
    X_expr = gain_expr(src_node, domain)
    X_exact_expr = gain_expr(src_node, domain, delay_model="exact")

    # 4) output TF H(s) and display-only Y(s)
    H_unsimplified = sp.together(loop_tf_expr / X_expr)
    H_expr = sp.simplify(H_unsimplified)   # <-- divide out the source block
    Y_expr = sp.simplify(H_expr * X_expr)     # for LaTeX display only
    H_exact_expr = sp.simplify(loop_tf_exact_expr / X_exact_expr)

    # Use the canonical symbol from utils (s or z) and coeff helper
    var = s if domain == "s" else z
    def expr_to_coeffs(e):
        n, d = sp.fraction(sp.together(e))
        return [float(c) for c in Poly(n, var).all_coeffs()], \
            [float(c) for c in Poly(d, var).all_coeffs()]

    raw_num, raw_den = expr_to_coeffs(H_unsimplified)
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

    ode_latex = coeffs_to_ode_str(out_num, out_den, domain)  # ideally wrap with proper braces

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

    analysis = compute_analysis(
        out_num,
        out_den,
        raw_num=raw_num,
        raw_den=raw_den,
        freq_expr=H_exact_expr,
        domain=domain,
    )
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
        "analysis": analysis,
     }


def compute_analysis(num, den, raw_num=None, raw_den=None, freq_expr=None, domain="s"):
    """Return pole/zero, bode, Nyquist, Nichols and root-locus data."""

    try:
        sys = control.TransferFunction(num, den)
    except Exception:
        return {
            "pz": {"poles": [], "zeros": [], "stability": "unknown"},
            "bode": {},
            "nyquist": {},
            "nichols": {},
            "root_locus": {"branches": [], "k": [], "snapshots": [], "zeros": []},
        }

    analysis_num = raw_num if raw_num is not None else num
    analysis_den = raw_den if raw_den is not None else den
    try:
        analysis_sys = control.TransferFunction(analysis_num, analysis_den)
    except Exception:
        analysis_sys = sys

    def complex_pairs(values):
        return [
            {"re": float(np.real(v)), "im": float(np.imag(v))}
            for v in values
        ]

    poles = control.poles(analysis_sys)
    zeros = control.zeros(analysis_sys)

    if np.any(np.real(poles) > 1e-8):
        stability = "unstable"
    elif np.any(np.isclose(np.real(poles), 0.0, atol=1e-8)):
        stability = "marginal"
    else:
        stability = "stable"

    # shared logarithmic grid for frequency-domain plots
    w_min = 1e-3
    w_max = 1e3
    try:
        # attempt to broaden frequency range based on pole/zero magnitudes
        magnitudes = np.abs(np.concatenate([poles, zeros]))
        finite = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)]
        if finite.size:
            w_min = max(min(finite) / 100, 1e-4)
            w_max = max(max(finite) * 100, 1e1)
    except Exception:
        pass

    omega = np.logspace(np.log10(w_min), np.log10(w_max), 600)

    use_exact_delay = freq_expr is not None and domain == "s"
    response = None
    if use_exact_delay:
        try:
            var_symbol = s if domain == "s" else z
            freq_func = sp.lambdify(var_symbol, freq_expr, modules=["numpy"])
            s_vals = 1j * omega
            response = np.asarray(freq_func(s_vals), dtype=complex)
        except Exception:
            response = None
            use_exact_delay = False

    if response is None:
        mag, phase, _ = control.bode(sys, omega, plot=False)
        mag = np.squeeze(mag)
        phase = np.squeeze(phase)
        mag = np.atleast_1d(mag)
        phase = np.atleast_1d(phase)
        response = mag * np.exp(1j * phase)
    else:
        response = np.atleast_1d(response)
        mag = np.abs(response)
        phase = np.unwrap(np.angle(response))

    mag = np.asarray(mag, dtype=float)
    phase = np.asarray(phase, dtype=float)
    response = np.asarray(response, dtype=complex)

    mag_db = 20 * np.log10(np.maximum(mag, np.finfo(float).tiny))
    phase_deg = np.degrees(phase)

    def finite_or_none(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            return [finite_or_none(v) for v in value]
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    try:
        gm, pm, wg, wp = control.margin(sys)
    except Exception:
        gm = pm = wg = wp = None

    gm_db = None
    if gm is not None and gm > 0 and np.isfinite(gm):
        gm_db = 20 * np.log10(gm)

    try:
        bandwidth = control.bandwidth(sys)
        if np.isnan(bandwidth) or np.isinf(bandwidth):
            bandwidth = None
    except Exception:
        bandwidth = None

    # Nyquist curve using bode response
    conj_segment = np.conjugate(response[:-1][::-1])
    nyquist_curve = np.concatenate([response, conj_segment])

    # Nichols chart data (phase vs mag in dB)
    nichols_phase = phase_deg
    nichols_mag = mag_db

    # Root-locus data
    try:
        rlist, klist = control.root_locus(analysis_sys, plot=False)
        rlist = np.squeeze(rlist)
        if rlist.ndim == 1:
            rlist = rlist[:, np.newaxis]
    except Exception:
        rlist = np.empty((0, 0), dtype=complex)
        klist = np.array([])

    branches = []
    for col in range(rlist.shape[1]):
        branch_vals = []
        for val in rlist[:, col]:
            if np.isfinite(val):
                branch_vals.append((float(np.real(val)), float(np.imag(val))))
            else:
                branch_vals.append(None)
        xs = []
        ys = []
        for item in branch_vals:
            if item is None:
                xs.append(None)
                ys.append(None)
            else:
                xs.append(item[0])
                ys.append(item[1])
        branches.append({"x": xs, "y": ys})

    snapshots = []
    for row in range(rlist.shape[0]):
        poles_row = []
        for val in rlist[row, :] if rlist.size else []:
            if np.isfinite(val):
                poles_row.append({"re": float(np.real(val)), "im": float(np.imag(val))})
            else:
                poles_row.append(None)
        snapshots.append(poles_row)

    analysis = {
        "pz": {
            "poles": complex_pairs(poles),
            "zeros": complex_pairs(zeros),
            "stability": stability,
        },
        "bode": {
            "omega": omega.tolist(),
            "magnitude_db": mag_db.tolist(),
            "phase_deg": phase_deg.tolist(),
            "gain_margin_db": finite_or_none(gm_db),
            "phase_margin_deg": finite_or_none(pm),
            "gain_cross_freq": finite_or_none(wg),
            "phase_cross_freq": finite_or_none(wp),
            "bandwidth": finite_or_none(bandwidth),
        },
        "nyquist": {
            "real": nyquist_curve.real.tolist(),
            "imag": nyquist_curve.imag.tolist(),
        },
        "nichols": {
            "phase_deg": nichols_phase.tolist(),
            "magnitude_db": nichols_mag.tolist(),
        },
        "root_locus": {
            "branches": branches,
            "k": finite_or_none(klist.tolist()),
            "snapshots": snapshots,
            "zeros": complex_pairs(zeros),
        },
    }

    return analysis