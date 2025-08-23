
# pages/demos/exponential.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
import math

import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
from matplotlib import rcParams
rcParams["text.usetex"] = False
rcParams["text.parse_math"] = False
rcParams["mathtext.fontset"] = "cm"

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D  

from utils.img import fig_to_base64

# Blueprint in demos namespace
demos_exponential_bp = Blueprint(
    "demos_exponential", __name__, template_folder="../../templates"
)

# Fixed time axis 
T_MAX = 2.5 * np.pi
N_SAMPLES = 1000
TIME = np.linspace(0.0, T_MAX, N_SAMPLES)

# Defaults matching notebook
DEFAULTS = dict(
    magnitude=1.0,
    phase=0.0,            # rad
    omega=3.0,            # rad/s (normalized)
    sigma=0.2,            # real part
    time=6.0,             # Until t cursor (s)
    mode="Until t",       # Entirely or Until t
)

@demos_exponential_bp.route("/", methods=["GET"])
def page():
    return render_template(
        "demos/exponential.html",
        defaults=DEFAULTS,
        t_max=T_MAX,
        pi=math.pi,
    )

def _compute_curve(magnitude, phase, omega, sigma, t_cursor, mode):
    if mode == "Entirely":
        t = TIME
    else:
        idx = int(t_cursor * N_SAMPLES / T_MAX)
        idx = max(1, min(idx, N_SAMPLES))
        t = TIME[:idx]
    x = magnitude * np.exp(t * (1j*omega + sigma) + 1j*phase)
    return t, x

@demos_exponential_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        magnitude = float(data.get("magnitude", DEFAULTS["magnitude"]))
        phase     = float(data.get("phase", DEFAULTS["phase"]))
        omega     = float(data.get("omega", DEFAULTS["omega"]))
        sigma     = float(data.get("sigma", DEFAULTS["sigma"]))
        t_cursor  = float(data.get("time", DEFAULTS["time"]))
        mode      = (data.get("mode") or DEFAULTS["mode"]).strip()

        t, x = _compute_curve(magnitude, phase, omega, sigma, t_cursor, mode)

        # Figure layout: 3D spiral + Re/Im + abs(x| + phase
        fig = plt.figure(figsize=(11, 6))
        gs_main = GridSpec(1, 12, figure=fig)
        gs_3d   = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[0:5])
        gs_reim = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[6:9])
        gs_magph= GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[9:12])

        ax3d = fig.add_subplot(gs_3d[0], projection="3d")
        ax_re = fig.add_subplot(gs_reim[0])
        ax_im = fig.add_subplot(gs_reim[1])
        ax_ma = fig.add_subplot(gs_magph[0])
        ax_ph = fig.add_subplot(gs_magph[1])

        # 3D spiral (blue)
        ax3d.set_xlabel(r"$t$")
        ax3d.set_xlim(0, T_MAX)
        ax3d.set_ylabel(r"$\mathrm{Re}\{x(t)\}$")
        ax3d.set_ylim(-10, 10)
        ax3d.set_zlabel(r"$\mathrm{Im}\{x(t)\}$")
        ax3d.set_zlim(-10, 10)
        ax3d.plot(t, x.real, x.imag, color="blue")
        if mode == "Until t":
            ax3d.plot([t[-1]], [x[-1].real], [x[-1].imag], "o", color="black")
            ax3d.text(t[-1]+0.3, x[-1].real, x[-1].imag, r"$t$", fontsize=12)

        # Real (red)
        ax_re.plot(t, x.real, color="red")
        ax_re.set_xlabel(r"$t$", labelpad=-2)
        ax_re.set_xlim(0, T_MAX)
        ax_re.set_ylabel(r"$\mathrm{Re}\{x(t)\}$", labelpad=-4)
        ax_re.set_ylim(-10, 10)
        ax_re.grid(True)
        if mode == "Until t":
            ax_re.plot([t[-1]], [x[-1].real], "o", color="black")
            ax_re.text(t[-1]+0.3, x[-1].real, r"$t$", fontsize=12)

        # Imag (green)
        ax_im.plot(t, x.imag, color="green")
        ax_im.set_xlabel(r"$t$", labelpad=-2)
        ax_im.set_xlim(0, T_MAX)
        ax_im.set_ylabel(r"$\mathrm{Im}\{x(t)\}$", labelpad=-4)
        ax_im.set_ylim(-10, 10)
        ax_im.grid(True)
        if mode == "Until t":
            ax_im.plot([t[-1]], [x[-1].imag], "o", color="black")
            ax_im.text(t[-1]+0.3, x[-1].imag, r"$t$", fontsize=12)

        # Magnitude (purple)
        ax_ma.plot(t, np.abs(x), color="purple")
        ax_ma.set_xlabel(r"$t$", labelpad=-2)
        ax_ma.set_xlim(0, T_MAX)
        ax_ma.set_ylabel(r"$|x(t)|$", labelpad=-4)
        ax_ma.set_ylim(0, 10)
        ax_ma.grid(True)
        if mode == "Until t":
            ax_ma.plot([t[-1]], [abs(x[-1])], "o", color="black")
            ax_ma.text(t[-1]+0.3, abs(x[-1]), r"$t$", fontsize=12)

        # Phase (orange)
        ph = np.zeros_like(t) if magnitude == 0.0 else np.angle(x)
        ax_ph.plot(t, ph, color="orange")
        ax_ph.set_xlabel(r"$t$", labelpad=-2)
        ax_ph.set_xlim(0, T_MAX)
        ax_ph.set_ylabel(r"$\varphi\{x(t)\}$", labelpad=-4)
        yticks = np.arange(-np.pi, np.pi + 0.1, np.pi/2)
        ax_ph.set_yticks(yticks)
        ax_ph.set_yticklabels([
            r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"
        ])
        ax_ph.grid(True)
        if mode == "Until t":
            ax_ph.plot([t[-1]], [ph[-1]], "o", color="black")
            ax_ph.text(t[-1]+0.3, ph[-1], r"$t$", fontsize=12)

        fig.tight_layout()
        png = fig_to_base64(fig)
        plt.close(fig)

        return jsonify({"image": png})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
