"""
URL endpoints
─────────────
GET  /block_diagram/          → HTML page with the canvas
POST /block_diagram/compile   → JSON API: graph-in, TF/SS/ODE-out
"""
from flask import render_template, request, jsonify, current_app
from . import block_diagram_bp as bp 
from .services import compile_diagram
import control
import numpy as np
from pathlib import Path



@bp.route("/", methods=["GET"], endpoint="diagram_page")
def diagram_page():
    """Return the canvas UI plus the list of built-in example files."""
    static_root = Path(current_app.static_folder)        # <-- never None
    preload_dir = static_root / "blocks_preloaded"
    preloaded   = sorted(p.name for p in preload_dir.glob("*.bdiag"))
    return render_template("block_diagram.html",
                           preloaded_diagrams=preloaded)


@bp.route("/compile", methods=["POST"])
def compile_diagram_api():
    """Take a JSON graph → return TF, state-space and ODE/difference eqn."""
    graph = request.get_json(force=True)
    # future-proof: allow ?domain=z or ?domain=s in query string
    domain = request.args.get("domain", graph.get("domain", "s"))
    try:
        result = compile_diagram(graph, domain=domain)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@bp.route('/simulate', methods=['POST'])
def simulate():
    tf_json = request.get_json()
    num, den = tf_json["num"], tf_json["den"]
    sys = control.TransferFunction(num, den)
    t, y = control.step_response(sys, T=np.linspace(0, 10, 500))

    sat = tf_json.get("saturation")
    if sat:
        lower = sat.get("lower")
        upper = sat.get("upper")
        if lower is not None or upper is not None:
            # np.clip handles None by treating them as -/+inf
            lo = lower if lower is not None else -np.inf
            hi = upper if upper is not None else np.inf
            y = np.clip(y, lo, hi)

    return jsonify(time=t.tolist(), y=y.tolist())