# pages/dft_page.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
import math

discrete_fourier_bp = Blueprint("discrete_fourier", __name__,
                                template_folder="templates")
dft_bp = discrete_fourier_bp

# ---------------------------------------------------------------------------
# helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------
def make_sequence(func_str: str, N: int, shift: float,
                  amp: float, width: float) -> np.ndarray:
    """
    Build x[n] for n = 0 … N-1 with three transforms:
        n' = n * width + shift
        x'[n] = amp · f(n')
    """
    n = np.arange(N, dtype=float)
    n_prime = n * width + shift

    # safe evaluation sandbox
    safe = {
        "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
        "abs": abs, "complex": complex,
        # simple window helpers
        "rect": lambda n,N: 1.0,
        "tri":  lambda n,N: (1.0 - abs((2*n)/(N-1) - 1)) if N > 1 else 1.0,
        "step": lambda n,N: 1.0 if n >= 0 else 0.0,
        "delta":lambda n,N: 1.0 if n == 0 else 0.0,
        "sign": lambda n,N: 0.0 if n == 0 else (1.0 if n > 0 else -1.0)
    }

    seq = []
    for k, n_val in enumerate(n_prime):
        local = {"n": n_val, "N": N} | safe
        try:
            y = eval(func_str, {"__builtins__": None}, local)
        except Exception as e:
            raise ValueError(f"Error at n={k}: {e}")
        seq.append(amp * complex(y))
    return np.asarray(seq, dtype=complex)

def compute_dft(x: np.ndarray, pad_factor: int):
    N = x.size
    M = max(1, pad_factor) * N
    X = np.fft.fft(x, n=M)

    k = np.arange(M)
    mag = np.abs(X)
    phase = np.angle(X)

    # normalise & clean tiny bins
    if mag.max() > 0:
        mag /= mag.max()
    phase[mag < 1e-6] = 0.0

    return {
        "n": np.arange(N).tolist(),
        "x_real": x.real.tolist(),
        "x_imag": x.imag.tolist(),
        "k": k.tolist(),
        "mag": mag.tolist(),
        "phase": phase.tolist(),
        "label": f"N = {N},  M = {M}"
    }

# ---------------------------------------------------------------------------
# routes --------------------------------------------------------------------
# ---------------------------------------------------------------------------
@discrete_fourier_bp.route("/", methods=["GET"], endpoint="show_dft")
def dft():
    return render_template("dft.html")

@dft_bp.route("/update", methods=["POST"])
def update_dft():
    data = request.get_json(force=True) or {}
    try:
        N          = max(1, int(data.get("N",          8)))
        func_str   = data.get("func", "sin(2*pi*n/N)").strip() or "0"
        padFactor  = max(1, int(data.get("padFactor", 10)))
        shift      = float(data.get("shift", 0))
        amp        = float(data.get("amp",   1))
        width      = float(data.get("width", 1))

        x = make_sequence(func_str, N, shift, amp, width)
        res = compute_dft(x, padFactor)
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(res)
