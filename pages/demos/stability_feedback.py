# pages/demos/stability_feedback.py
from __future__ import annotations
import threading, io, base64
import math
from functools import lru_cache
import numpy as np
from flask import Blueprint, render_template, request, jsonify
import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt
from utils.img import fig_to_base64

RC_PARAMS = {
    "savefig.bbox": "tight",
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "font.size": 22,
    "mathtext.fontset": "cm",
}

stability_feedback_bp = Blueprint(
    "stability_feedback", __name__, template_folder="../../templates"
)

# UI defaults 
STEP = 0.5
DEFAULTS = dict(a=1, b=1, K=1)
BOUNDS   = dict(a=(-3, 3), b=(-2, 2), K=(-6, 6))

# concurrency controls
RENDER_LOCK = threading.Lock()   # for Matplotlib
META_LOCK = threading.Lock()   # for sequencing state
_LAST_IMG: str | None = None
_LATEST_SEQ = 0                  # highest seq number weave seen from any client

def _bg_for_stability(pole_real: float) -> str:
    return "#ccffcc" if pole_real < 0 else "#ffcccc"

def _draw_axes(ax):
    maxval = BOUNDS["a"][1] + BOUNDS["b"][1] * BOUNDS["K"][1] + 0.5
    # draw axes with arrows, size twice of normal arrow size
    arrowprops = dict(arrowstyle="-|>", lw=1.2, color="black", mutation_scale=40)
    ax.annotate("", xy=(maxval, 0), xytext=(-maxval, 0), arrowprops=arrowprops)
    ax.annotate("", xy=(0, maxval), xytext=(0, -maxval), arrowprops=arrowprops)
    ax.set_xlim(-maxval, maxval)
    ax.set_ylim(-maxval, maxval)
    ax.set_aspect("equal", adjustable="box")
    minor_ticks = np.arange(-int(maxval), int(maxval) + 1, 1)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.25)
    ax.set_xlabel(r"$\mathrm{Re}\{s\}$")
    ax.set_ylabel(r"$\mathrm{Im}\{s\}$")

def _fig_to_svg_data_url(fig) -> str:
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="svg")
        plt.close(fig)
        return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return fig_to_base64(fig)
def _coerce_float(value, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if math.isfinite(candidate) else default

def _coerce_int(value, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    return candidate

@lru_cache(maxsize=8192)
def _render_cached(a_q: int, b_q: int, K_q: int) -> str:
    a = a_q / 2.0
    b = b_q / 2.0
    K = K_q / 2.0
    p_H = a
    p_Q = a - K*b
    with plt.rc_context(RC_PARAMS):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
        
        ax1.set_title("$H(s)$")
        ax1.set_facecolor(_bg_for_stability(p_H))
        _draw_axes(ax1)
        ax1.plot([p_H],[0.0], marker="x", ms=30, mew=2.6, color="tab:red")

        ax2.set_title("$Q(s)$")
        ax2.set_facecolor(_bg_for_stability(p_Q))
        _draw_axes(ax2)
        ax2.plot([p_Q],[0.0], marker="x", ms=30, mew=2.6, color="tab:red")

        img = _fig_to_svg_data_url(fig)
    return img


# cache for defaults
try:
    _ = _render_cached(int(round(DEFAULTS["a"]*2)),
                       int(round(DEFAULTS["b"]*2)),
                       int(round(DEFAULTS["K"]*2)))
except Exception:
    pass

@stability_feedback_bp.route("/", methods=["GET"])
def page():
    return render_template(
        "demos/stability_feedback.html",
        defaults=dict(
            **DEFAULTS,
            step=STEP,
            min_a=BOUNDS["a"][0], max_a=BOUNDS["a"][1],
            min_b=BOUNDS["b"][0], max_b=BOUNDS["b"][1],
            min_K=BOUNDS["K"][0], max_K=BOUNDS["K"][1],
        )
    )

@stability_feedback_bp.route("/compute", methods=["POST"])
def compute():
    """Render only the newest request: older ones are dropped immediately."""
    global _LATEST_SEQ, _LAST_IMG
    data = request.get_json(force=True) or {}

    # Read inputs
    a = _coerce_float(data.get("a", DEFAULTS["a"]), DEFAULTS["a"])
    b = _coerce_float(data.get("b", DEFAULTS["b"]), DEFAULTS["b"])
    K = _coerce_float(data.get("K", DEFAULTS["K"]), DEFAULTS["K"])
    seq = _coerce_int(data.get("seq", 0), 0)  # monotonically increasing from client

    with META_LOCK:
        if seq > _LATEST_SEQ:
            _LATEST_SEQ = seq
        is_latest = (seq == _LATEST_SEQ)

    if not is_latest:
        return jsonify(dict(image=_LAST_IMG, seq=seq, note="superseded")), 200

    a = float(np.clip(a, *BOUNDS["a"]))
    b = float(np.clip(b, *BOUNDS["b"]))
    K = float(np.clip(K, *BOUNDS["K"]))
    a_q = int(round(a * 2.0))
    b_q = int(round(b * 2.0))
    K_q = int(round(K * 2.0))

    with META_LOCK:
        if seq != _LATEST_SEQ:
            return jsonify(dict(image=_LAST_IMG, seq=seq, note="preempted")), 200

    try:
        with RENDER_LOCK:
            img = _render_cached(a_q, b_q, K_q)
            _LAST_IMG = img
        return jsonify(dict(image=img, seq=seq)), 200
    except Exception:
        #  return last-good or default
        if _LAST_IMG is not None:
            return jsonify(dict(image=_LAST_IMG, seq=seq, note="served_last_good")), 200
        img = _render_cached(int(round(DEFAULTS["a"]*2)),
                             int(round(DEFAULTS["b"]*2)),
                             int(round(DEFAULTS["K"]*2)))
        _LAST_IMG = img
        return jsonify(dict(image=img, seq=seq, note="served_default")), 200
