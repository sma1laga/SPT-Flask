# pages/dft_page.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
import re
from utils.math_utils import (
    rect_N, tri_N, step, cos, sin, sign, delta_n, exp_iwt, inv_t, si
)
from utils.eval_helpers import safe_eval

discrete_fourier_bp = Blueprint("discrete_fourier", __name__,
                                template_folder="templates")
dft_bp = discrete_fourier_bp

# ---------------------------------------------------------------------------
# helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _rewrite_expr(expr: str) -> str:
    """Rewrite expression to match the discrete symbols/functions.

    Converts:
    - 'rect_4[...]' -> 'rect(..., 4)'
    - 'tri_3[...]' -> 'tri(..., 3)'
    """
    # e.g., replace "rect_5(anystr)" → "rect(anystr, 5)"
    # FIXME: fails for nested brackets, e.g., "rect_5[sin[k]])"
    pattern = r'rect_(.+)\[([^]]+)\]'
    repl = r'rect(\g<2>, \g<1>)'
    expr_new = re.sub(pattern, repl, expr)
    # e.g., replace "tri_5(anystr)" → "tri(anystr, 5)"
    pattern = r'tri_(.+)\[([^]]+)\]'
    repl = r'tri(\g<2>, \g<1>)'
    expr_new = re.sub(pattern, repl, expr_new)
    expr_new = expr_new.replace('[', '(').replace(']', ')')
    expr_new = expr_new.replace('^', '**')  # replace ^ with Python's **
    return expr_new

def make_sequence(func_str: str, L: int, M: int, shift: float,
                  amp: float, width: float) -> np.ndarray:
    """
    Build x[k] for k = 0 … L-1 with three transforms:
        k' = (k-shift) % L * 1/width
        x'[k] = amp · f[k']
    and pad to length M if necessary.
    """
    func_str = _rewrite_expr(func_str)

    k = np.arange(L, dtype=float)
    k_prime = ((k-shift) % L) * 1/width

    # safe evaluation sandbox
    ctx = {
        "k": k_prime, "n": k_prime, "L": L, "M": M,
        "pi": np.pi, "e": np.e, "i": 1j, "j": 1j,
        "rect": rect_N, "tri": tri_N, "step": step,
        "sin": sin, "cos": cos, "tan": np.tan,
        "exp": np.exp, "exp_iwt": exp_iwt, "log": np.log, "sqrt": np.sqrt,
        "abs": abs, "sign": sign, "delta": delta_n,
        "si": si, "inv_k": inv_t,
    }

    try:
        x = safe_eval(func_str, ctx) if func_str else np.zeros_like(k, dtype=float)
        if not isinstance(x, np.ndarray):
            x = np.ones(L, dtype=float) * x  # convert scalar to array
    except Exception as e:
        raise ValueError(f"Error: {e}")
    if len(x) < M:
        x = np.pad(x, (0, M - len(x)))
    return amp * x.astype(complex)

def compute_dft(x: np.ndarray, dft_len: int):
    L = x.size
    X = np.fft.fft(x, n=dft_len)
    X = np.where(np.isclose(np.abs(X), 0), 0, X)  # clip small magnitudes to zero
    X.imag = np.where(np.isclose(X.imag, 0), 0, X.imag)  # avoid noisy phase

    return {
        "k": np.arange(L).tolist(),
        "x_real": x.real.tolist(),
        "x_imag": x.imag.tolist(),
        "mu": np.arange(dft_len).tolist(),
        "mag": np.abs(X).tolist(),
        "phase": np.angle(X).tolist(),
        "label": f"L = {L},  M = {dft_len}"
    }

# ---------------------------------------------------------------------------
# routes --------------------------------------------------------------------
# ---------------------------------------------------------------------------
@discrete_fourier_bp.route("/", methods=["GET"], endpoint="show_dft")
def dft():
    return render_template(
        "dft.html",
        page_title = "DFT Plot Online – Discrete-Time Spectrum (Magnitude & Phase)",
        meta_description = "Compute the DFT/FFT of a discrete-time signal online. Plot magnitude and phase spectrum, explore frequency resolution and leakage, and export data.",
    )

@dft_bp.route("/update", methods=["POST"])
def update_dft():
    data = request.get_json(force=True) or {}
    try:
        L          = min(256, max(1, int(data.get("L", 8))))
        M          = min(256, max(1, int(data.get("M", 8))))
        func_str   = data.get("func", "sin[2*pi*k/L]").strip() or "0"
        shift      = float(data.get("shift", 0))
        amp        = float(data.get("amp",   1))
        width      = float(data.get("width", 1))

        x = make_sequence(func_str, L, M, shift, amp, width)
        x_dft = compute_dft(x, M)
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(x_dft)
