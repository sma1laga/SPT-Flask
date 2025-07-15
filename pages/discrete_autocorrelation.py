from flask import Blueprint, render_template, request, jsonify
import re
from pages.discrete_convolution import compute_discrete_convolution


discrete_autocorrelation_bp = Blueprint(
    "discrete_autocorrelation",
    __name__,
    template_folder="templates"
)


@discrete_autocorrelation_bp.route("/", methods=["GET"])
def discrete_autocorrelation():
    return render_template("discrete/discrete_autocorrelation.html")


def _reverse_n(expr: str) -> str:
    """Return the expression with standalone k/n replaced by -k/-n."""
    expr = re.sub(r"\bk\b", "(-k)", expr)
    return re.sub(r"\bn\b", "(-n)", expr)


def compute_discrete_autocorrelation(func_str: str, ds: float = 1.0):
    """Compute discrete autocorrelation via convolution with reversed sequence."""
    return compute_discrete_convolution(func_str, _reverse_n(func_str), ds)


@discrete_autocorrelation_bp.route("/update", methods=["POST"])
def discrete_autocorrelation_update():
    data = request.get_json(force=True) or {}
    func_str = data.get("func", "").strip()
    try:
        ds = float(data.get("ds", 1.0))
        if ds <= 0:
            raise ValueError("Δk must be > 0")
    except Exception as e:
        return jsonify(error=f"Invalid Δk: {e}"), 400

    try:
        result = compute_discrete_autocorrelation(func_str, ds)
    except Exception as e:
        return jsonify(error=str(e)), 400

    return jsonify(result)