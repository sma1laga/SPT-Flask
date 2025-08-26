# pages/demos/stability_feedback.py
from __future__ import annotations
import threading, io, base64
from functools import lru_cache
import numpy as np
from flask import Blueprint, render_template, request, jsonify
import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from utils.img import fig_to_base64

rcParams.update({
    "figure.dpi": 140,
    "savefig.bbox": "tight",
    "font.size": 13,
    "mathtext.fontset": "dejavusans",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
})

stability_feedback_bp = Blueprint(
    "stability_feedback", __name__, template_folder="../../templates"
)

# UI defaults 
STEP = 0.5
DEFAULTS = dict(a=1.5, b=2.0, K=2.0)
BOUNDS   = dict(a=(-10.0, 10.0), b=(-10.0, 10.0), K=(-10.0, 10.0))

# Plot window & ticks
XMIN, XMAX = -25.0, 25.0
YMIN, YMAX = -25.0, 25.0
XTICKS_MAJOR = np.arange(XMIN, XMAX + 1e-9, 5.0)
YTICKS_MAJOR = np.arange(YMIN, YMAX + 1e-9, 5.0)
XTICKS_MINOR = np.arange(XMIN, XMAX + 1e-9, 1.0)
YTICKS_MINOR = np.arange(YMIN, YMAX + 1e-9, 1.0)

# concurrency controls
RENDER_LOCK = threading.Lock()   # for Matplotlib 
META_LOCK   = threading.Lock()   # for sequencing state
_LAST_IMG: str | None = None
_LATEST_SEQ = 0                  # highest seq number weave seen from any client

def _bg_for_stability(pole_real: float) -> str:
    return "#E7F8E7" if pole_real < 0 else "#FBE1E1"

def _draw_axes(ax):
    ax.axhline(0, color="black", lw=1.2)
    ax.axvline(0, color="black", lw=1.2)
    ax.annotate("", xy=(XMAX, 0), xytext=(XMAX-1.5, 0),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.annotate("", xy=(0, YMAX), xytext=(0, YMAX-1.5),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(XTICKS_MAJOR); ax.set_yticks(YTICKS_MAJOR)
    ax.set_xticks(XTICKS_MINOR, minor=True); ax.set_yticks(YTICKS_MINOR, minor=True)
    ax.grid(True, which="major", linewidth=1.0, alpha=0.55)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.25)
    ax.set_xlabel(r"Re$\{s\}$"); ax.set_ylabel(r"Im$\{s\}$")

def _fig_to_svg_data_url(fig) -> str:
    try:
        rcParams["svg.fonttype"] = "none"
        buf = io.BytesIO()
        fig.savefig(buf, format="svg")
        return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return fig_to_base64(fig)

@lru_cache(maxsize=8192)
def _render_cached(a_q: int, b_q: int, K_q: int) -> str:
    a = a_q / 2.0; b = b_q / 2.0; K = K_q / 2.0
    p_H = a
    p_Q = a - K*b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20.0, 8.5))
    ax1.set_title(r"$H(s)=\dfrac{b}{s-a}$")
    ax1.set_facecolor(_bg_for_stability(p_H))
    _draw_axes(ax1)
    ax1.plot([p_H],[0.0], marker="x", ms=18, mew=2.6, color="crimson")
    ax1.text(p_H+1.2, 1.2, r"$\times\ \text{pole}$", color="crimson", fontsize=13)

    ax2.set_title(r"$Q(s)=\dfrac{H(s)}{1+K\,H(s)}=\dfrac{b}{\,s-a+K\,b\,}$")
    ax2.set_facecolor(_bg_for_stability(p_Q))
    _draw_axes(ax2)
    ax2.plot([p_Q],[0.0], marker="x", ms=18, mew=2.6, color="crimson")
    ax2.text(p_Q+1.2, 1.2, r"$\times\ \text{pole}$", color="crimson", fontsize=13)

    plt.tight_layout(pad=1.0, w_pad=3.4)
    img = _fig_to_svg_data_url(fig)
    plt.close(fig)
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
    a = float(data.get("a", DEFAULTS["a"]))
    b = float(data.get("b", DEFAULTS["b"]))
    K = float(data.get("K", DEFAULTS["K"]))
    seq = int(data.get("seq", 0))  # monotonically increasing from client

    with META_LOCK:
        if seq > _LATEST_SEQ:
            _LATEST_SEQ = seq
        is_latest = (seq == _LATEST_SEQ)

    if not is_latest:
        return jsonify(dict(image=_LAST_IMG, seq=seq, note="superseded")), 200

    a = float(np.clip(a, *BOUNDS["a"]))
    b = float(np.clip(b, *BOUNDS["b"]))
    K = float(np.clip(K, *BOUNDS["K"]))
    a_q = int(round(a * 2.0)); b_q = int(round(b * 2.0)); K_q = int(round(K * 2.0))

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
