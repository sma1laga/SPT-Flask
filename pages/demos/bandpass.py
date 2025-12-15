# pages/demos/bandpass.py
from __future__ import annotations
import threading
from functools import lru_cache
import numpy as np
from flask import Blueprint, render_template, request, jsonify
import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt
from utils.img import fig_to_base64
from utils.math_utils import rect

demos_bandpass_bp = Blueprint(
    "demos_bandpass", __name__, template_folder="../../templates"
)

RC_PARAMS = {
    "axes.labelsize": 12,
    "font.size": 11,
    "mathtext.fontset": "cm",
}

# UI ranges
STEP = 0.5
MIN_DELTA, MAX_DELTA = 0.5, 10.0
MIN_W0,    MAX_W0    = 0, 30.0
DEFAULT_DELTA = 3.5
DEFAULT_W0    = 15.0


def ideal_bp_H(omega, delta_omega, omega0):
    'create an ideal band pass H(jw) in the frequency domain'
    if delta_omega == 0:
        return np.zeros_like(omega)
    return rect((omega + omega0) / delta_omega) + rect((omega - omega0) / delta_omega)

def envelope_ht(t, delta_omega):
    return (delta_omega / np.pi) * np.sinc((delta_omega * t) / (2.0 * np.pi))

# Precomputed grids
W_MIN, W_MAX, W_N = -40.0, 40.0, 1201
T_MIN, T_MAX, T_N =  -4.0,  4.0,  901
W_GRID = np.linspace(W_MIN, W_MAX, W_N)
T_GRID = np.linspace(T_MIN, T_MAX, T_N)

RENDER_LOCK = threading.Lock()
_LAST_IMG = None

@lru_cache(maxsize=1024)
def _render_img_cached(dw_q: int, w0_q: int) -> str:
    delta_omega = dw_q / 2.0  # step = 0.5
    omega0      = w0_q / 2.0

    H   = ideal_bp_H(W_GRID, delta_omega, omega0)
    env = envelope_ht(T_GRID, delta_omega)
    ht  = env * np.cos(omega0 * T_GRID)
    with plt.rc_context(RC_PARAMS):
        fig, (axF, axT) = plt.subplots(1, 2, figsize=(8.6, 3.0), layout="constrained")
        # Frequency domain
        axF.plot(W_GRID, H, lw=2)
        axF.set_xlim(W_MIN, W_MAX)
        # axF.set_ylim(-0.05, 1.05)
        axF.grid(True)
        axF.set_title("Frequency domain")
        axF.set_xlabel(r"$\omega$")
        axF.set_ylabel(r"$H(\mathrm{j}\omega)$")
        # Time domain
        axT.plot(T_GRID, ht,          color="C0", label=r"$h(t)$", lw=2)
        axT.plot(T_GRID, env, color="r", ls="--", label="Envelope", lw=1)
        axT.plot(T_GRID, -env, color="r", ls="--", lw=1)
        axT.set_xlim(T_MIN, T_MAX)
        axT.set_ylim(-3.5, 3.5)
        axT.grid(True)
        axT.set_title("Time domain")
        axT.set_xlabel(r"$t$")
        axT.set_ylabel(r"$h(t)$")
        axT.legend(loc="upper right")

        img = fig_to_base64(fig)
    return img

#  cache for default
try:
    _ = _render_img_cached(int(round(DEFAULT_DELTA * 2.0)), int(round(DEFAULT_W0 * 2.0)))
except Exception:
    pass

@demos_bandpass_bp.route("/", methods=["GET"])
def page():
    defaults = dict(
        delta_omega=DEFAULT_DELTA,
        omega0=DEFAULT_W0,
        step=STEP,
        min_delta=MIN_DELTA,
        max_delta=MAX_DELTA,
        min_omega0=MIN_W0,
        max_omega0=MAX_W0,
    )
    return render_template("demos/bandpass.html", defaults=defaults)

@demos_bandpass_bp.route("/compute", methods=["POST"])
def compute():
    global _LAST_IMG
    try:
        data = request.get_json(force=True) or {}
        delta_omega = float(data.get("delta_omega", DEFAULT_DELTA))
        omega0      = float(data.get("omega0",      DEFAULT_W0))

        delta_omega = float(np.clip(delta_omega, MIN_DELTA, MAX_DELTA))
        omega0      = float(np.clip(omega0,      MIN_W0,    MAX_W0))

        dw_q = int(round(delta_omega * 2.0))
        w0_q = int(round(omega0      * 2.0))

        with RENDER_LOCK:
            img = _render_img_cached(dw_q, w0_q)
            _LAST_IMG = img
        return jsonify(dict(image=img))
    except Exception:
        if _LAST_IMG is not None:
            return jsonify(dict(image=_LAST_IMG, note="served_last_good")), 200
        try:
            with RENDER_LOCK:
                img = _render_img_cached(int(round(DEFAULT_DELTA*2.0)), int(round(DEFAULT_W0*2.0)))
                _LAST_IMG = img
            return jsonify(dict(image=img, note="served_default")), 200
        except Exception:
            return jsonify(dict(error="render_failed")), 200
