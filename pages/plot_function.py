# pages/plot_function.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
from utils.math_utils import (
    rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si
)

# additional special functions kept local to this module
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

plot_function_bp = Blueprint("plot_function", __name__,
                             template_folder="templates")


@plot_function_bp.route("/", methods=["GET"])
def plot_function():
    return render_template("plot_function.html")


@plot_function_bp.route("/update", methods=["POST"])
def plot_function_update():
    data = request.get_json(force=True) or {}

    func1_str, func2_str = data.get("func1", ""), data.get("func2", "")

    # f₁ sliders ----------------------------------------------------
    s1  = float(data.get("shift1", 0))
    a1  = float(data.get("amp1",   1))
    w1  = float(data.get("width1", 1))

    # f₂ sliders ----------------------------------------------------
    s2  = float(data.get("shift2", 0))
    a2  = float(data.get("amp2",   1))
    w2  = float(data.get("width2", 1))

    # initial broad grid to estimate a suitable centre
    t_broad = np.linspace(-100, 100, 8000)

    ns_broad = dict(t=t_broad, np=np,
                    rect=rect, tri=tri, step=step,
                    cos=cos, sin=sin,
                    sign=sign, delta=delta, exp_iwt=exp_iwt, inv_t=inv_t,
                    si=si, exp=np.exp,
                    arcsin=arcsin, arccos=arccos, arctan=arctan,
                    sinh=sinh, cosh=cosh, tanh=tanh,
                    gauss=gauss)

    # evaluate on broad grid
    try:
        y1_broad = eval(func1_str, ns_broad) if func1_str.strip() else np.zeros_like(t_broad)
    except Exception as e:
        return jsonify({"error": f"Error in f₁(t): {e}"}), 400
    if np.any(np.isinf(y1_broad)):
        return jsonify({"error": "f₁(t) produced non-finite values"}), 400
    
    y2_broad = None
    if func2_str.strip():
        try:
            y2_broad = eval(func2_str, ns_broad)
        except Exception as e:
            return jsonify({"error": f"Error in f₂(t): {e}"}), 400
        if np.any(np.isinf(y2_broad)):
            return jsonify({"error": "f₂(t) produced non-finite values"}), 400

    # apply transforms on broad grid for centre calculation
    t1_broad = t_broad * w1 + s1
    t2_broad = t_broad * w2 + s2 if y2_broad is not None else None

    def center_point(t_arr, y_arr):
        """Heuristic centre for plotting.

        For signals that start from (near) zero and then rise (e.g. step or
        ramp), the arithmetic centre of mass can drift far from the interesting
        region.  We first look for such a "starting" edge and fall back to the
        magnitude centre of mass if none is found.
        """
        if y_arr is None:
            return None, 0
        
        mag = np.abs(np.nan_to_num(y_arr, nan=0.0))
        total = np.sum(mag)
        if np.isnan(total):
            return None, 0
        if total == 0:
            return None, 0

        # detect a leading region of (near) zeros followed by activity.
        # require a reasonably long run of near-zero values to avoid
        # mistaking oscillatory functions for one-sided signals.
        thresh = mag.max() * 1e-3
        nz = np.where(mag > thresh)[0]
        min_run = max(10, len(t_arr) // 100)
        if len(nz) > 0 and nz[0] >= min_run and np.all(mag[:nz[0]] < thresh):
            return float(t_arr[nz[0]]), total

        return float(np.sum(t_arr * mag) / total), total

    c1, m1 = center_point(t1_broad, y1_broad)
    c2, m2 = center_point(t2_broad, y2_broad)

    if c1 is None and c2 is None:
        center = 0.0
    elif c2 is None:
        center = c1
    elif c1 is None:
        center = c2
    else:
        center = (c1 * m1 + c2 * m2) / (m1 + m2)

    # final grid centred around detected centre
    start = (center - 20 - s1) / w1
    end = (center + 20 - s1) / w1
    t = np.linspace(start, end, 4000)

    ns = dict(t=t, np=np,
              rect=rect, tri=tri, step=step,
              cos=cos, sin=sin,
              sign=sign, delta=delta, exp_iwt=exp_iwt, inv_t=inv_t,
              si=si, exp=np.exp,
              arcsin=arcsin, arccos=arccos, arctan=arctan,
              sinh=sinh, cosh=cosh, tanh=tanh,
              gauss=gauss)

    # evaluate again on final grid
    try:
        y1 = eval(func1_str, ns) if func1_str.strip() else np.zeros_like(t)
    except Exception as e:
        return jsonify({"error": f"Error in f₁(t): {e}"}), 400
    if np.any(np.isinf(y1)):
        return jsonify({"error": "f₁(t) produced non-finite values"}), 400

    y2 = None
    if func2_str.strip():
        try:
            y2 = eval(func2_str, ns)
        except Exception as e:
            return jsonify({"error": f"Error in f₂(t): {e}"}), 400
        if np.any(np.isinf(y2)):
            return jsonify({"error": "f₂(t) produced non-finite values"}), 400
        
    # apply separate transforms
    t1 = t * w1 + s1
    y1 = y1 * a1

    if y2 is not None:
        t2 = t * w2 + s2
        y2 = y2 * a2
    else:
        t2 = None
        
    if t2 is not None:
        x_min = float(min(t1.min(), t2.min()))
        x_max = float(max(t1.max(), t2.max()))
    else:
        x_min = float(t1.min())
        x_max = float(t1.max())
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
        "t1": t1.tolist(),
        "y1": to_json_list(y1),
        "t2": t2.tolist() if t2 is not None else None,
        "y2": to_json_list(y2) if y2 is not None else None,
        "xrange": [x_min, x_max]

    })
