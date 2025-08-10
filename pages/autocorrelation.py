from flask import Blueprint, render_template
from pages.convolution import compute_convolution
import numpy as np
import re


autocorrelation_bp = Blueprint("autocorrelation", __name__)

@autocorrelation_bp.route("/", methods=["GET"])
def autocorrelation():
    return render_template("autocorrelation.html")


def _reverse_t(expr: str) -> str:
    """Return the expression with every standalone `t` replaced by `-t`."""
    return re.sub(r"\bt\b", "(-t)", expr)

def _replace_exp(expr: str, func: str) -> str:
    """Replace exp_iwt with the provided real function name."""
    return re.sub(r"exp_iwt\s*\(([^)]*)\)", rf"{func}(\1)", expr)


def compute_autocorrelation(func_str: str):
    """Compute continuous autocorrelation with conjugate symmetry handling.

    Evaluates real and imaginary components separately, mirrors the result to
    enforce symmetry around the origin and returns both parts for plotting.
    """
    f_re = _replace_exp(func_str, "np.cos")
    f_im = _replace_exp(func_str, "np.sin")

    rr = compute_convolution(f_re, _reverse_t(f_re))
    if isinstance(rr, dict) and rr.get("error"):
        return rr
    ii = compute_convolution(f_im, _reverse_t(f_im))
    ri = compute_convolution(f_re, _reverse_t(f_im))
    ir = compute_convolution(f_im, _reverse_t(f_re))

    re_part = np.array(rr["y_conv"]) + np.array(ii["y_conv"])
    im_part = np.array(ir["y_conv"]) - np.array(ri["y_conv"])

    n = len(re_part)
    re_part = 0.5*(re_part + re_part[::-1])
    im_part = 0.5*(im_part - im_part[::-1])

    tau = np.array(rr["t_conv"]) - rr["t_conv"][n//2]

    return {
        "t": rr["t1"],
        "y_re": rr["y1"],
        "y_im": ir["y1"],
        "tau": tau.tolist(),
        "r_re": re_part.tolist(),
        "r_im": im_part.tolist()
    }