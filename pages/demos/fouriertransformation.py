# pages/demos/fouriertransformation.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
from matplotlib import rcParams
rcParams["text.usetex"] = False
rcParams["text.parse_math"] = True
import matplotlib.pyplot as plt
from utils.img import fig_to_base64
from functools import lru_cache


demos_fouriertransformation_bp = Blueprint(
    "demos_fouriertransformation", __name__, template_folder="../../templates"
)

NUM_TIMEPOINTS = 2048
NUM_OMEGA_POINTS = 8096
OMEGA_MULTIPLES_OF_PI = 10
OMEGA_START = -OMEGA_MULTIPLES_OF_PI * np.pi / 2.0
OMEGA_END   =  OMEGA_MULTIPLES_OF_PI * np.pi / 2.0

MIN_WIDTH,  MAX_WIDTH,  STEP_WIDTH  = 0.5, 2.0, 0.25
MIN_HEIGHT, MAX_HEIGHT, STEP_HEIGHT = 0.5, 2.0, 0.25
STEP_DISPLACEMENT = 0.5  

DEFAULTS = dict(
    x_type="rect",
    x_displacement=0.0,
    x_width=1.0,
    x_height=1.0,
)

def rect(t, t_g=0.5, t_0=0.0):
    return np.where(np.abs(t - t_0) <= t_g, 1.0, 0.0)

def triangle(t, t_g=1.0, t_0=0.0):
    return np.maximum(1.0 - np.abs((t - t_0) / t_g), 0.0)

def epsilon(t, t_0=0.0):
    return np.where(t - t_0 >= 0, 1.0, 0.0)

def si(t):
    return np.sinc(t / np.pi)

def si_squared(t):
    return si(t) ** 2

# ===== Grids =====
def _timepoints_for(x_type):
    if x_type in ("si", "si squared"):
        t_start, t_end = -15.0, 15.0
    else:
        t_start, t_end = -5.0, 5.0
    t = np.linspace(t_start, t_end, NUM_TIMEPOINTS)
    return t, t_start, t_end

OMEGA_POINTS = np.linspace(OMEGA_START, OMEGA_END, NUM_OMEGA_POINTS)
TIME_GRIDS = {xt: _timepoints_for(xt) for xt in ("rect", "triangle", "si", "si squared")}

# ===== Notebook equivalent mappings =====
def generate_x(t, x_type, x_displacement, x_width, x_height):
    if x_type == "rect":
        return rect(t, 0.5 * x_width, x_displacement) * x_height
    elif x_type == "si":
        return si((t - x_displacement) / x_width) * x_height
    elif x_type == "triangle":
        return triangle(t, x_width, x_displacement) * x_height
    elif x_type == "si squared":
        return si_squared((t - x_displacement) / x_width) * x_height
    raise ValueError("Unknown x_type")

def generate_abs(w, x_type, x_displacement, x_width, x_height):
    if x_type == "rect":
        return np.abs(si(w / (2.0 / x_width))) * (x_height * x_width)
    elif x_type == "si":
        return np.abs(rect(w, 1.0 / x_width) * np.pi * x_width) * x_height
    elif x_type == "triangle":
        a = 1.0 / x_width
        return (1.0 / a) * si_squared(w / (2.0 * a)) * x_height
    elif x_type == "si squared":
        a = 1.0 / x_width
        return (np.pi * x_width) * triangle(w, 2.0 * a) * x_height
    raise ValueError("Unknown x_type")

def generate_phi(w, x_type, x_displacement, x_width, x_height):
    # phase of e^
    pure_phase = -x_displacement * w + np.pi

    if x_type == "rect":
        func = si(w / (2.0 / x_width)) * (x_height * x_width)
        ret = pure_phase.copy()
        ret[func < 0] += np.pi - 1e-5
        return np.remainder(ret, 2.0 * np.pi) - np.pi

    if x_type == "si":
        ret = np.remainder(pure_phase, 2.0 * np.pi) - np.pi
        ret[np.abs(w) > 1.0 / x_width] = 0.0
        return ret

    if x_type == "triangle":
        return np.remainder(pure_phase, 2.0 * np.pi) - np.pi

    if x_type == "si squared":
        ret = np.remainder(pure_phase, 2.0 * np.pi) - np.pi
        ret[np.abs(w) > 2.0 / x_width] = 0.0
        return ret

    raise ValueError("Unknown x_type")

# ===== Plot formatting =====
def _set_axis_formatting(fig, x_ax, abs_ax, phi_ax, x_type, t_start, t_end):
    # Time
    x_ax.set_xlim(t_start, t_end)
    x_ax.set_ylim(-2.1, 2.1)
    if x_type in ("si", "si squared"):
        k0 = int(np.floor(t_start / np.pi)) + 1
        k1 = int(np.floor(t_end / np.pi))
        ticks = [k * np.pi for k in range(k0, k1 + 1)]
        x_ax.set_xticks(ticks, [fr"${k}\pi$" for k in range(k0, k1 + 1)])
    else:
        x_ax.set_xticks(np.arange(np.ceil(t_start), np.floor(t_end) + 0.001, 0.5))
    x_ax.set_yticks([-2, -1, 0, 1, 2])
    x_ax.grid()
    x_ax.set_xlabel(r"$t$")
    x_ax.set_ylabel(r"$x(t)$")
    x_ax.set_title("Time signal")

    # Magnitude
    abs_ax.set_xlim(OMEGA_START, OMEGA_END)
    if x_type in ("si", "si squared"):
        abs_ax.set_ylim(-1.1 * np.pi, 4.1 * np.pi)
        k0 = int(np.ceil(OMEGA_START / np.pi))
        k1 = int(np.floor(OMEGA_END / np.pi))
        ticks = [k * np.pi for k in range(k0, k1 + 1)]
        abs_ax.set_xticks(ticks, [fr"${k}\pi$" for k in range(k0, k1 + 1)])
        abs_ax.set_yticks([k * np.pi for k in range(-1, 5)],
                          [fr"${k}\pi$" for k in range(-1, 5)])
    else:
        abs_ax.set_ylim(-1.1, 4.1)
        k0 = -OMEGA_MULTIPLES_OF_PI // 2
        k1 =  OMEGA_MULTIPLES_OF_PI // 2
        ticks = [k * np.pi for k in range(k0, k1 + 1)]
        abs_ax.set_xticks(ticks, [fr"${k}\pi$" for k in range(k0, k1 + 1)])
    abs_ax.grid()
    abs_ax.set_xlabel(r"$\omega$")
    abs_ax.set_ylabel(r"$|X(j\omega)|$")
    abs_ax.set_title("Magnitude")

    # Phase
    phi_ax.set_xlim(OMEGA_START, OMEGA_END)
    phi_ax.set_ylim(-np.pi - 1.0, np.pi + 1.0)
    phi_ax.set_yticks(
        [-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
    )
    k0 = -OMEGA_MULTIPLES_OF_PI // 2
    k1 =  OMEGA_MULTIPLES_OF_PI // 2
    ticks = [k * np.pi for k in range(k0, k1 + 1)]
    phi_ax.set_xticks(ticks, [fr"${k}\pi$" for k in range(k0, k1 + 1)])
    phi_ax.grid()
    phi_ax.set_xlabel(r"$\omega$")
    phi_ax.set_ylabel(r"$\phi(j\omega)$")
    phi_ax.set_title("Phase")

def _render_plot(x_type, x_displacement, x_width, x_height):
    t, t_start, t_end = TIME_GRIDS[x_type]
    w = OMEGA_POINTS

    x = generate_x(t, x_type, x_displacement, x_width, x_height)
    Xabs = generate_abs(w, x_type, x_displacement, x_width, x_height)
    Xphi = generate_phi(w, x_type, x_displacement, x_width, x_height)

    fig, (x_ax, abs_ax, phi_ax) = plt.subplots(3, 1, figsize=(9, 6))
    plt.tight_layout(h_pad=2.2, pad=1.6)

    _set_axis_formatting(fig, x_ax, abs_ax, phi_ax, x_type, t_start, t_end)

    x_ax.plot(t, x, linewidth=1.2, color="C3")
    abs_ax.plot(w, Xabs, linewidth=1.2, color="C0")
    phi_ax.plot(w, Xphi, linewidth=1.2, color="C2")

    return fig

@lru_cache(maxsize=128)
def _cached_image(x_type, x_displacement, x_width, x_height):
    """Return rendered plot as base64 image using an LRU cache."""
    fig = _render_plot(x_type, x_displacement, x_width, x_height)
    return fig_to_base64(fig)

# ===== Routes =====
@demos_fouriertransformation_bp.route("/", methods=["GET"])
def page():
    t_start, t_end = (-5.0, 5.0)  # for default "rect"
    min_disp = t_start + MAX_WIDTH
    max_disp = t_end - MAX_WIDTH
    return render_template(
        "demos/fouriertransformation.html",
        defaults=dict(
            **DEFAULTS,
            min_displacement=min_disp,
            max_displacement=max_disp,
            min_width=MIN_WIDTH, max_width=MAX_WIDTH, step_width=STEP_WIDTH,
            min_height=MIN_HEIGHT, max_height=MAX_HEIGHT, step_height=STEP_HEIGHT,
            step_displacement=STEP_DISPLACEMENT,
        ),
    )

@demos_fouriertransformation_bp.route("/compute", methods=["POST"])
def compute():
    data = request.get_json(force=True) or {}
    x_type = data.get("x_type", DEFAULTS["x_type"])
    x_displacement = float(data.get("x_displacement", DEFAULTS["x_displacement"]))
    x_width = float(data.get("x_width", DEFAULTS["x_width"]))
    x_height = float(data.get("x_height", DEFAULTS["x_height"]))

    x_width = round(float(np.clip(x_width, MIN_WIDTH, MAX_WIDTH)), 5)
    x_height = round(float(np.clip(x_height, MIN_HEIGHT, MAX_HEIGHT)), 5)

    t_start, t_end = TIME_GRIDS[x_type][1], TIME_GRIDS[x_type][2]
    min_disp = t_start + MAX_WIDTH
    max_disp = t_end - MAX_WIDTH
    x_displacement = round(float(np.clip(x_displacement, min_disp, max_disp)), 5)

    img = _cached_image(x_type, x_displacement, x_width, x_height)


    return jsonify(dict(
        image=img,
        x_type=x_type,
        x_displacement=x_displacement,
        x_width=x_width,
        x_height=x_height,
        min_displacement=min_disp,
        max_displacement=max_disp,
    ))
