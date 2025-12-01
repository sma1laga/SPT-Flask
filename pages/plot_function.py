# pages/plot_function.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
from functools import partial
from utils.math_utils import (
    rect, tri, step, cos, sin, sign, exp_iwt, inv_t, si
)
from utils.eval_helpers import error_data


# additional special functions kept local to this module
def delta_plotting(t, th=1e-3):
    """Returns array with at most 1 nonzero value where `t` is closest to 0 and `|t| < th` """
    delta_out = np.zeros_like(t)
    idx = np.argmin(np.abs(t))
    if np.abs(t[idx]) < th:
        delta_out[idx] = 1
    return delta_out

def arcsin(t):
    return np.arcsin(t)

def arccos(t):
    return np.arccos(t)

def arctan(t):
    return np.arctan(t)

def sinh(t):
    return np.sinh(t)

def cosh(t):
    return np.cosh(t)

def tanh(t):
    return np.tanh(t)

def gauss(t):
    """Standard normal PDF."""
    return np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)

def _adjust_t(t: np.ndarray, shift: float, width: float) -> np.ndarray:
    """Adjust time values based on shift and width (1/width * (t - shift))."""
    return (t - shift) / width

plot_function_bp = Blueprint("plot_function", __name__,
                             template_folder="templates")


@plot_function_bp.route("/", methods=["GET"])
def plot_function():
    return render_template("plot_function.html")


@plot_function_bp.route("/update", methods=["POST"])
def plot_function_update():
    data = request.get_json(force=True) or {}

    func1_str, func2_str = data.get("func1", "").strip(), data.get("func2", "").strip()

    if  func1_str == "heartcoded" or func2_str == "heartcoded" or func1_str == "tarik+lea" or func2_str == "tarik+lea" or func1_str == "lea+tarik":
        func1_str = "np.sqrt(1 - (np.abs(t) - 1)**2)"
        func2_str = "np.arccos(1 - np.abs(t)) - np.pi"

    # f₁ sliders ----------------------------------------------------
    s1  = float(data.get("shift1", 0))
    a1  = float(data.get("amp1",   1))
    w1  = float(data.get("width1", 1))

    # f₂ sliders ----------------------------------------------------
    s2  = float(data.get("shift2", 0))
    a2  = float(data.get("amp2",   1))
    w2  = float(data.get("width2", 1))

    _adjust_t1 = partial(_adjust_t, shift=s1, width=w1)
    _adjust_t2 = partial(_adjust_t, shift=s2, width=w2)

    # calculate signals for t = [(center - MAX_T), (center + MAX_T)]
    # plot around t = [(center - MAX_T/2), (center + MAX_T/2)]
    MAX_T = 10
    center = 0
    start = center - MAX_T
    end = center + MAX_T
    t = np.linspace(start, end, 4097)

    ns = dict(t=_adjust_t1(t), np=np, pi=np.pi, e=np.e, j=1j,
              rect=rect, tri=tri, step=step,
              cos=cos, sin=sin,
              sign=sign, delta=partial(delta_plotting, th=0.1*w1), exp_iwt=exp_iwt, inv_t=inv_t,
              si=si, exp=np.exp, sqrt=np.sqrt,
              arcsin=arcsin, arccos=arccos, arctan=arctan,
              sinh=sinh, cosh=cosh, tanh=tanh,
              gauss=gauss)

    try:
        y1 = a1 * eval(func1_str, ns) if func1_str else np.zeros_like(t)
    except Exception as e:
        return jsonify(error_data("Error in f₁(t): ", e)), 400
    if np.any(np.isinf(y1)):
        return jsonify({"error": "f₁(t) produced non-finite values"}), 400

    y2 = None
    if func2_str:
        try:
            ns["t"] = _adjust_t2(t)
            ns["delta"] = partial(delta_plotting, th=0.1*w2)
            y2 = a2 * eval(func2_str, ns)
        except Exception as e:
            return jsonify(error_data("Error in f₂(t): ", e)), 400
        if np.any(np.isinf(y2)):
            return jsonify({"error": "f₂(t) produced non-finite values"}), 400

    def to_json_list(arr):
        if np.iscomplexobj(arr):
            return {
                "real": arr.real.tolist(),
                "imag": arr.imag.tolist(),
            }
        lst = arr.tolist()
        cleaned = []
        for v in lst:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                cleaned.append(None)
            else:
                cleaned.append(v)
        return cleaned
    
    return jsonify({
        "t1": t.tolist(),
        "y1": to_json_list(y1),
        "t2": t.tolist() if y2 is not None else None,
        "y2": to_json_list(y2) if y2 is not None else None,
        "xrange": [center - MAX_T/2, center + MAX_T/2]
    })
