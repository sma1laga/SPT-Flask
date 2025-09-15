# pages/demos/dtft_impulses.py
from flask import Blueprint, render_template, request, make_response, url_for
import io, json, os
import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt

dtft_impulses_bp = Blueprint(
    "dtft_impulses", __name__, template_folder="../../templates"
)

# --- Config (optional) ---
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
CONFIG_PATH = os.path.join(SRC_DIR, "config.json")
_DEFAULT_CONFIG = {"fig_width": 8.0, "fig_height": 6.0, "use_slider": True, "slider_multiplier": 1.0}

def _load_config():
    cfg = dict(_DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg

# --- Notebook-accurate helpers (same as before) ---
def update_x(k, norm_freq):
    # x[k] = cos(pi * norm_freq * k)
    return np.cos(np.pi * norm_freq * k)

def get_DTFT(norm_freq):
    # Impulspositionen und -höhen wie im Jupyter-Notebook
    dtft_height = 2 if norm_freq in [0, 1] else 1
    if norm_freq == 0:
        dtft_pos, dtft_val = [0.0], [dtft_height]
    else:
        dtft_pos = [-norm_freq, norm_freq]
        dtft_val = [dtft_height, dtft_height]
    return np.array(dtft_pos, float), np.array(dtft_val, float)

def _draw_pi_axis_labels(ax):
    ax.margins(x=0)
    ax.grid()
    ax.set_xlabel(r"Frequency $\Omega$")
    ax.set_ylabel(r"$X(\mathrm{e}^{\mathrm{j}\Omega})$")
    ax.set_title("DTFT")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 2.2)

    ax.set_xticks(np.arange(-1.0, 1.1, 0.5))
    ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])


def _make_figure(norm_freq, cfg):
    fig_w = float(cfg.get("fig_width", _DEFAULT_CONFIG["fig_width"]))
    fig_h = float(cfg.get("fig_height", _DEFAULT_CONFIG["fig_height"]))
    fig, (x_ax, dtft_ax) = plt.subplots(2, 1, figsize=(fig_w, fig_h), layout="constrained")

    # Zeitsignal
    k = np.arange(-20, 21)
    x = update_x(k, norm_freq)
    x_ax.grid()
    x_ax.set_xlabel("Index $k$")
    x_ax.set_ylabel("$x[k]$")
    x_ax.set_title("Time Domain")
    xlim = 20.5
    x_ax.set_xlim(-xlim, xlim)
    x_ax.set_ylim(-1.1, 1.1)
    x_ax.hlines(0, -xlim, xlim, color='black', linewidth=1)
    x_ax.stem(k, x, markerfmt="o", basefmt=" ")

    # Impulsspektrum
    _draw_pi_axis_labels(dtft_ax)
    pos, val = get_DTFT(norm_freq)
    dtft_ax.vlines(pos, 0.0, val, linewidth=2)
    dtft_ax.plot(pos, val, "^", markersize=7)

    return fig

def _fig_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.getvalue()

# ---------- Routes ----------
@dtft_impulses_bp.route("/", methods=["GET"])
def page():
    cfg = _load_config()
    # initial value (no full render here; the <img> will call /img)
    try:
        norm_freq = float(request.args.get("freq", "0"))
    except Exception:
        norm_freq = 0.0
    norm_freq = max(0.0, min(1.0, norm_freq))
    # adapt most important things...
    ui = {
        "use_slider": bool(cfg.get("use_slider", True)),
        "min_freq": 0.0,
        "max_freq": 1.0,
        "step": 0.05,
        "value": norm_freq,
        "desc": r"\omega / \pi",
        # initial image URL (cache-busted)
        "img_url": url_for("dtft_impulses.image", freq=norm_freq, _=0),
    }
    return render_template("demos/dtft_impulses.html", ui=ui)

@dtft_impulses_bp.route("/img", methods=["GET"])
def image():
    try:
        cfg = _load_config()
        try:
            norm_freq = float(request.args.get("freq", "0"))
        except Exception:
            norm_freq = 0.0
        norm_freq = max(0.0, min(1.0, norm_freq))

        fig = _make_figure(norm_freq, cfg)
        png = _fig_png_bytes(fig)

        resp = make_response(png)
        # Prevent stale images during rapid scrubbing; browser can still cache per URL
        resp.headers["Content-Type"] = "image/png"
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except Exception as e:
        print(e)
