# pages/demos/dtft_dft.py
from flask import Blueprint, render_template, request, make_response, url_for
import io, os, json
import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt

dtft_dft_bp = Blueprint("dtft_dft", __name__, template_folder="../../templates")

# --- Optional config (mirrors notebook) ---
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
CONFIG_PATH = os.path.join(SRC_DIR, "config.json")
_DEFAULT_CONFIG = {"fig_width": 8.0, "fig_height": 6.0}

def _load_config():
    cfg = dict(_DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg

# --- Core helpers (ported from notebook) ---
def update_x(k, norm_freq):
    return np.cos(np.pi * norm_freq * k)

def update_crop(m, norm_freq):
    k_crop = np.arange(m)
    return np.cos(np.pi * norm_freq * k_crop)

def update_DFT(crop):
    return np.abs(np.fft.fft(crop))

def get_DTFT_positions(norm_freq):
    # Impulspositionen (Ω/π): axis is -1..2, ticks: -π .. 2π
    h = 2 if norm_freq in (0.0, 1.0) else 1.0
    if norm_freq == 0.0:
        pos = [0.0, 2.0]
        val = [h, h]
    elif norm_freq == 1.0:
        pos = [-1.0, 1.0]
        val = [h, h]
    else:
        # main ±ν plus the periodic copy at 2-ν
        pos = [-norm_freq, norm_freq, 2.0 - norm_freq]
        val = [h, h, h]
    return np.array(pos, float), np.array(val, float)

def _init_axes_styles(ax):
    ax.margins(x=0)
    ax.grid()
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.tick_params(labelsize=10)

def _render_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def _make_figure(norm_freq, m, cfg):
    fig_w = float(cfg.get("fig_width", _DEFAULT_CONFIG["fig_width"]))
    fig_h = float(cfg.get("fig_height", _DEFAULT_CONFIG["fig_height"]))
    fig, axes = plt.subplots(2, 2, figsize=(1.2*fig_w, 1.0*fig_h), layout="constrained")
    (x_ax, dtft_ax), (crop_ax, dft_ax) = axes

    # --- Data ---
    k = np.arange(-20, 21)                  # time index
    x = update_x(k, norm_freq)              # full cosine
    crop = update_crop(m, norm_freq)        # center block (length M, starting in middle)
    dft = update_DFT(crop)                  # |FFT|
    # indices for red (crop window) vs blue stems in x-plot
    start_red = len(x)//2
    idx_red = np.arange(start_red, min(len(x), start_red + m))
    idx_blue = np.setdiff1d(np.arange(len(k)), idx_red, assume_unique=True)

    # --- Top-left: x[k] with highlighted crop ---
    _init_axes_styles(x_ax)
    x_ax.set_title("Time Domain Signal")
    x_ax.set_xlabel("Index $k$")
    x_ax.set_ylabel("$x[k]$")
    x_ax.set_xlim(-20.5, 20.5)
    x_ax.set_ylim(-1.1, 1.1)
    x_ax.set_yticks(np.arange(-1.0, 1.1, 0.5))
    x_ax.hlines(0, -20.5, 20.5, color="black", linewidth=1)
    # blue stems/markers
    x_ax.vlines(k[idx_blue], 0.0, x[idx_blue], linewidth=1.6)
    x_ax.plot(k[idx_blue], x[idx_blue], "o", markersize=5, markerfacecolor="none")
    # red stems/markers for crop
    x_ax.vlines(k[idx_red], 0.0, x[idx_red], color="red", linewidth=1.6)
    x_ax.plot(k[idx_red], x[idx_red], "o", markersize=5, markerfacecolor="none", color="red")

    # --- Top-right: DTFT impulses on Ω/π axis from -1..2 ---
    _init_axes_styles(dtft_ax)
    dtft_ax.set_title("DTFT")
    dtft_ax.set_xlabel(r"Frequency $\Omega$")
    dtft_ax.set_ylabel(r"$X(\mathrm{e}^{\mathrm{j}\Omega})$")
    dtft_ax.set_xlim(-1.0, 2.0)
    dtft_ax.set_ylim(0.0, 2.2)
    dtft_ax.set_xticks(np.arange(-1.0, 2.1, 0.5))
    dtft_ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    dtft_ax.set_yticks([0, 1, 2])
    dtft_ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    pos, val = get_DTFT_positions(norm_freq)
    dtft_ax.vlines(pos, 0.0, val, linewidth=1.8)
    dtft_ax.plot(pos, val, "^", markersize=6)

    # --- Bottom-left: crop ---
    _init_axes_styles(crop_ax)
    crop_ax.set_title("Window")
    crop_ax.set_xlabel("Index $k$")
    crop_ax.set_ylabel(r"$\tilde{x}[k]$")
    crop_ax.set_xlim(-0.5, len(crop))
    crop_ax.set_ylim(-1.1, 1.1)
    crop_ax.set_yticks(np.arange(-1.0, 1.1, 0.5))
    crop_ax.hlines(0, -0.5, len(crop), color="black", linewidth=1)
    crop_ax.vlines(np.arange(len(crop)), 0.0, crop, color="red", linewidth=1.6)
    crop_ax.plot(np.arange(len(crop)), crop, "o", markersize=5, markerfacecolor="none", color="red")

    # --- Bottom-right: DFT magnitude ---
    _init_axes_styles(dft_ax)
    dft_ax.set_title("DFT")
    dft_ax.set_xlabel(r"Frequency $\mu$")
    dft_ax.set_ylabel(r"$X[\mu]$")
    xlim = (-len(dft)//2, len(dft))
    dft_ax.set_xlim(*xlim)
    dft_ax.set_xticks([0, len(dft)//2])
    dft_ax.set_ylim(-len(dft)*0.05, len(dft)*1.05)
    yticks = [0, len(dft)//2, len(dft)]
    dft_ax.set_yticks(yticks)
    dft_ax.hlines(0, xlim[0], xlim[1], color="black", linewidth=1)
    dft_ax.vlines(np.arange(len(dft)), 0.0, dft, color="red", linewidth=1.6)
    dft_ax.plot(np.arange(len(dft)), dft, "o", markersize=5, markerfacecolor="none", color="red")

    return fig

# -------- Routes --------
@dtft_dft_bp.route("/", methods=["GET"])
def page():
    cfg = _load_config()
    # defaults exactly like notebook
    try:
        norm_freq = float(request.args.get("freq", "0"))
    except Exception:
        norm_freq = 0.0
    norm_freq = max(0.0, min(1.0, norm_freq))

    try:
        m = int(request.args.get("m", "16"))
    except Exception:
        m = 16
    m = max(2, min(1024, m))  # limit M to [2, 1024]

    ui = {
        "min_freq": 0.0,
        "max_freq": 1.0,
        "step": 0.05,
        "value": norm_freq,
        "m": m,
        "m_options": [2**i for i in range(1, 11)],  # 2...1024
        "img_url": url_for("dtft_dft.image", freq=norm_freq, m=m, _=0),
    }
    return render_template("demos/dtft_dft.html", ui=ui)

@dtft_dft_bp.route("/img", methods=["GET"])
def image():
    cfg = _load_config()
    try:
        norm_freq = float(request.args.get("freq", "0"))
    except Exception:
        norm_freq = 0.0
    norm_freq = max(0.0, min(1.0, norm_freq))

    try:
        m = int(request.args.get("m", "16"))
    except Exception:
        m = 16
    m = max(2, min(1024, m))  # limit M to [2, 1024]

    fig = _make_figure(norm_freq, m, cfg)
    png = _render_png(fig)
    plt.close(fig)

    resp = make_response(png)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "no-store"
    return resp
