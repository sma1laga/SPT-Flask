# pages/demos/sampling.py
from __future__ import annotations
import io, base64, threading
from functools import lru_cache
import numpy as np
from flask import Blueprint, render_template, request, jsonify

import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt
from utils.img import fig_to_base64


sampling_bp = Blueprint(
    "sampling", __name__, template_folder="../../templates"
)

DEFAULTS = dict(
    w_g_over_pi=2.0, 
    w_a_over_pi=3.0,  
    show_xt=True,
    show_xa=True,
    show_yt=True,
    show_partials=False,
    show_X=True,
    show_Xa=True,
    show_Y=False,
)
STEP = 0.5
BOUNDS = dict(
    w_g_over_pi=(0.5, 5.0),
    w_a_over_pi=(1.0, 10.0),
)

# Time
T_MIN, T_MAX, T_N = -3.0, 3.0, 1601
t_grid = np.linspace(T_MIN, T_MAX, T_N)

# Frequency
PI = np.pi
W_MIN, W_MAX, W_N = -10*PI, 10*PI, 4001
w_grid = np.linspace(W_MIN, W_MAX, W_N)

# Concurrency 
RENDER_LOCK = threading.Lock()
META_LOCK = threading.Lock()
_LAST_IMG = None
_LATEST_SEQ = 0


def si(x: np.ndarray) -> np.ndarray:
    """si(x) = sin(x)/x with si(0)=1 (np.sinc is sin(pi x)/(pi x))."""
    return np.sinc(x / np.pi)

def x_time(t, w_g):
    return (w_g/(2.0*np.pi)) * si( (w_g*t)/2.0 )**2

def tri(u):
    """Λ(u) = max(1-|u|,0)."""
    return np.maximum(1.0 - np.abs(u), 0.0)

def X_freq(w, w_g):
    return tri(w / w_g)

def Xa_freq(w, w_g, w_a):
    kmin = int(np.floor((W_MIN - w_g)/w_a)) - 1
    kmax = int(np.ceil((W_MAX + w_g)/w_a)) + 1
    acc = np.zeros_like(w, dtype=float)
    for k in range(kmin, kmax+1):
        acc += X_freq(w - k*w_a, w_g)
    return acc


def ideal_lp_mask(w, w_g):
    return (np.abs(w) <= w_g).astype(float)

def Y_freq(w, w_g, w_a):
    lpf = (np.abs(w) <= (w_a/2.0)).astype(float)
    return Xa_freq(w, w_g, w_a) * lpf


def sampled_times(T):
    n_min = int(np.floor((T_MIN - 1e-12) / T))
    n_max = int(np.ceil((T_MAX + 1e-12) / T))
    ns = np.arange(n_min, n_max+1)
    ts = ns * T
    mask = (ts >= T_MIN-1e-12) & (ts <= T_MAX+1e-12)
    return ts[mask], ns[mask]

def reconstruct_y(t, T, w_g, x_samples, t_samples, partials=False):
    """
    y(t) = Σ x[n] * sinc((t - nT)/T), exact for ω_g < ω_a/2.
    Also returns a few partial components if requested.
    """
    Nt = t.size
    Ns = t_samples.size
    if Ns == 0:
        return np.zeros_like(t), []

    tau = t.reshape(Nt, 1) - t_samples.reshape(1, Ns)
    kernels = np.sinc(tau / T)       

    y = kernels @ x_samples

    parts = None
    if partials:
        idx_center = np.argmin(np.abs(t_samples))
        lo = max(0, idx_center-3); hi = min(Ns, idx_center+4)
        parts = [kernels[:, i] * x_samples[i] for i in range(lo, hi)]
    return y, parts


def _fig_to_svg_data_url(fig) -> str:
    try:
        with plt.rc_context({"svg.fonttype": "none"}):
            buf = io.BytesIO()
            fig.savefig(buf, format="svg")
            plt.close(fig)
            return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return fig_to_base64(fig)

# -Cached renderer 
@lru_cache(maxsize=4096)
def _render_cached(
    w_g_q: int, w_a_q: int,
    flags: int  
) -> str:
    """
    Render to SVG/PNG and return data URL.
    Quantization: 0.5 steps => multiply by 2 and round.
    flags bit order: 0 xt, 1 xa, 2 yt, 3 partials, 4 X, 5 Xa, 6 Y
    """
    
    w_g = (w_g_q / 2.0) * np.pi   
    w_a = (w_a_q / 2.0) * np.pi

    show_xt        = bool(flags & (1<<0))
    show_xa        = bool(flags & (1<<1))
    show_yt        = bool(flags & (1<<2))
    show_partials  = bool(flags & (1<<3))
    show_X         = bool(flags & (1<<4))
    show_Xa        = bool(flags & (1<<5))
    show_Y         = bool(flags & (1<<6))

    # ---- Time-domain signals ----
    xt = x_time(t_grid, w_g) if show_xt or show_xa or show_yt else None

    T = 2.0*np.pi / w_a  
    ts, ns = sampled_times(T)  
    xa_vals = x_time(ts, w_g) if show_xa or show_yt or show_partials else None

    yt = None; parts = None
    if show_yt or show_partials:
        yt, parts = reconstruct_y(t_grid, T, w_g, xa_vals, ts, partials=show_partials)

    # ---- Frequency-domain curves ----
    Xw  = X_freq(w_grid, w_g)        if show_X  else None
    Xaw = Xa_freq(w_grid, w_g, w_a)  if show_Xa else None
    Yw  = Y_freq(w_grid, w_g, w_a)   if show_Y  else None

    # ---- Plotting ----
    fig, (axT, axF) = plt.subplots(1, 2, figsize=(8.2, 3.4), layout="constrained")

    # Time domain
    axT.set_title("Time Domain")
    axT.set_xlabel(r"$t$")
    axT.set_ylabel(r"$x(t),\,x_a(t),\,y(t)$")
    axT.set_xlim(T_MIN, T_MAX)
    axT.set_ylim(-0.6, 2.8)
    axT.grid(True)

    if show_xt:
        axT.plot(t_grid, xt, color="black", lw=1.3, label=r"$x(t)$")
    if show_xa:
        if ts.size:
            axT.vlines(ts, 0, xa_vals, colors="C0", linewidth=0.8, alpha=0.7)
            axT.plot(ts, xa_vals, "^", color="C0", ms=6, label=r"$x_a(t)$")
    if show_yt:
        axT.plot(t_grid, yt, color="C3", lw=1.6, ls="--", label=r"$y(t)$")
    if show_partials and parts:
        for p in parts:
            axT.plot(t_grid, p, color="C2", lw=1.0, ls="--", alpha=0.6)
    if any([show_xt, show_xa, show_yt]):
        axT.legend(loc="upper right")

    # Frequency domain
    axF.set_title("Frequency Domain")
    axF.set_xlabel(r"$\omega$")
    axF.set_ylabel(r"$X(\mathrm{j}\omega),\,X_a(\mathrm{j}\omega),\,Y(\mathrm{j}\omega)$")
    axF.set_xlim(W_MIN, W_MAX)
    ymax = 1.0
    for arr in (Xw, Xaw, Yw):
        if arr is not None:
            ymax = max(ymax, float(np.max(arr)))
    axF.set_ylim(-0.05, ymax + 0.2)
    # pi ticks
    k0 = int(np.ceil(W_MIN/np.pi)); k1 = int(np.floor(W_MAX/np.pi))
    ticks = [k*np.pi for k in range(k0, k1+1, 2)]
    ticklbl = [ (r"${0}\pi$".format(k) if k!=0 else r"$0$") for k in range(k0, k1+1, 2) ]
    axF.set_xticks(ticks, ticklbl)
    axF.grid(True)

    if show_X:
        axF.plot(w_grid, Xw, color="black", lw=1.5, label=r"$X(\mathrm{j}\omega)$")
    if show_Xa:
        axF.plot(w_grid, Xaw, color="C0", lw=1.2, label=r"$X_a(\mathrm{j}\omega)$")
    if show_Y:
        axF.plot(w_grid, Yw, color="C3", lw=1.6, ls="--", label=r"$Y(\mathrm{j}\omega)$")
    if any([show_X, show_Xa, show_Y]):
        axF.legend(loc="upper right")

    img = _fig_to_svg_data_url(fig)
    return img

def _flags_from_defaults() -> int:
    f = 0
    for i, key in enumerate(["show_xt","show_xa","show_yt","show_partials","show_X","show_Xa","show_Y"]):
        if DEFAULTS[key]: f |= (1<<i)
    return f

try:
    _ = _render_cached(int(round(DEFAULTS["w_g_over_pi"]*2)),
                       int(round(DEFAULTS["w_a_over_pi"]*2)),
                       _flags_from_defaults())
except Exception:
    pass

# Routes 
@sampling_bp.route("/", methods=["GET"])
def page():
    return render_template("demos/sampling.html",
                           defaults=dict(**DEFAULTS, step=STEP,
                                         min_wg=BOUNDS["w_g_over_pi"][0], max_wg=BOUNDS["w_g_over_pi"][1],
                                         min_wa=BOUNDS["w_a_over_pi"][0], max_wa=BOUNDS["w_a_over_pi"][1]))

@sampling_bp.route("/compute", methods=["POST"])
def compute():
    global _LATEST_SEQ, _LAST_IMG
    data = request.get_json(force=True) or {}
    seq = int(data.get("seq", 0))

    with META_LOCK:
        if seq > _LATEST_SEQ:
            _LATEST_SEQ = seq
        is_latest = (seq == _LATEST_SEQ)

    if not is_latest:
        return jsonify(dict(image=_LAST_IMG, seq=seq, note="superseded")), 200

    wg_over_pi = float(data.get("w_g_over_pi", DEFAULTS["w_g_over_pi"]))
    wa_over_pi = float(data.get("w_a_over_pi", DEFAULTS["w_a_over_pi"]))
    wg_over_pi = float(np.clip(wg_over_pi, *BOUNDS["w_g_over_pi"]))
    wa_over_pi = float(np.clip(wa_over_pi, *BOUNDS["w_a_over_pi"]))

    flags = 0
    keys = ["show_xt","show_xa","show_yt","show_partials","show_X","show_Xa","show_Y"]
    for i, k in enumerate(keys):
        if bool(data.get(k, False)): flags |= (1<<i)

    wg_q = int(round(wg_over_pi * 2.0))
    wa_q = int(round(wa_over_pi * 2.0))

    with META_LOCK:
        if seq != _LATEST_SEQ:
            return jsonify(dict(image=_LAST_IMG, seq=seq, note="preempted")), 200

    try:
        with RENDER_LOCK:
            img = _render_cached(wg_q, wa_q, flags)
            _LAST_IMG = img
        return jsonify(dict(image=img, seq=seq)), 200
    except Exception:
        if _LAST_IMG is not None:
            return jsonify(dict(image=_LAST_IMG, seq=seq, note="served_last_good")), 200
        img = _render_cached(int(round(DEFAULTS["w_g_over_pi"]*2)),
                             int(round(DEFAULTS["w_a_over_pi"]*2)),
                             _flags_from_defaults())
        _LAST_IMG = img
        return jsonify(dict(image=img, seq=seq, note="served_default")), 200
