from flask import Blueprint, render_template
from pages.convolution import compute_convolution
import numpy as np
import re


autocorrelation_bp = Blueprint("autocorrelation", __name__)

@autocorrelation_bp.route("/", methods=["GET"])
def autocorrelation():
    return render_template("autocorrelation.html")



def _replace_exp(expr: str, func: str) -> str:
    """Replace exp_iwt with the provided real function name."""
    return re.sub(r"exp_iwt\s*\(([^)]*)\)", rf"{func}(\1)", expr)


def compute_autocorrelation(func_str: str):
    """
    Compute autocorrelation phi_xx(τ). If the input uses exp_iwt(...), split into
    cos/sin for Re/Im. Otherwise treat the input as purely real and set Im = 0
    Timereversal is numeric (3rd arg=True in compute_convolution).
    """
    uses_exp = re.search(r"exp_iwt\s*\(", func_str) is not None
    f_re = _replace_exp(func_str, "np.cos") if uses_exp else func_str
    f_im = _replace_exp(func_str, "np.sin") if uses_exp else "0"

    rr = compute_convolution(f_re, f_re, True)
    if isinstance(rr, dict) and rr.get("error"): return rr
    ii = compute_convolution(f_im, f_im, True)
    if isinstance(ii, dict) and ii.get("error"): return ii
    ri = compute_convolution(f_re, f_im, True)
    if isinstance(ri, dict) and ri.get("error"): return ri
    ir = compute_convolution(f_im, f_re, True)
    if isinstance(ir, dict) and ir.get("error"): return ir

    re_part = np.array(rr["y_conv"]) + np.array(ii["y_conv"])
    im_part = np.array(ir["y_conv"]) - np.array(ri["y_conv"])

    t1 = np.asarray(rr["t1"])
    dt = t1[1] - t1[0]
    n  = len(rr["y_conv"])
    mid = (n - 1) / 2
    tau = dt * (np.arange(n) - mid)

    return {
        "t": rr["t1"],
        "y_re": rr["y1"],   # samples of Re{x(t)}
        "y_im": ir["y1"],   # samples of Im{x(t} (zeros if uses_exp is False)
        "tau": tau.tolist(),
        "r_re": re_part.tolist(),
        "r_im": im_part.tolist(),
    }
