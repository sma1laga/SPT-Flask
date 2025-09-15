# pages/demos/z_trafo.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.img import fig_to_base64

RC_PARAMS = {
    "agg.path.chunksize": 20000,
    "path.simplify": True,
    "path.simplify_threshold": 0.5,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "font.size": 13,
    "axes.xmargin": 0.05,
    "axes.ymargin": 0.05,
}

demos_z_trafo_bp = Blueprint(
    "demos_z_trafo", __name__, template_folder="././templates"
)

#           cached grids 
_K = np.arange(0, 41, dtype=np.int32)
_W = np.linspace(-np.pi, np.pi, 1024, dtype=np.float32)
_W_NORM = _W / np.pi
_THETA = np.linspace(0.0, 2.0 * np.pi, 512, dtype=np.float32)
_UNIT_CIRCLE = np.stack([np.cos(_THETA), np.sin(_THETA)], axis=0)

_last = {"a": None, "norm_freq": None, "image": None}


# ----------- math -----------
def _signal(k: np.ndarray, a: float, omega0: float) -> np.ndarray:
    # x[k] = a^k * cos(omega0 k) * u[k]
    return (np.power(a, k, dtype=np.float32) * np.cos(omega0 * k, dtype=np.float32)).astype(np.float32)

def _dtft(a: float, omega0: float, w: np.ndarray) -> np.ndarray:
    # X(e^{jω}) = 1/2 * [ 1/(1 - a e^{j(ω0-ω)}) + 1/(1 - a e^{-j(ω0+ω)}) 
    jw1 = 1j * (omega0 - w)
    jw2 = -1j * (omega0 + w)
    z1 = (1.0 - a * np.exp(jw1, dtype=np.complex64)).astype(np.complex64)
    z2 = (1.0 - a * np.exp(jw2, dtype=np.complex64)).astype(np.complex64)
    return 0.5 * (1.0 / z1 + 1.0 / z2)

def _pn_poles_zeros(a: float, omega0: float):
    # poles at a e^{±jω0}; one finite zero at z = a cos(ω_0) (RE Achse)
    poles = a * np.exp(1j * np.array([omega0, -omega0], dtype=np.float32)).astype(np.complex64)
    zeros = np.array([a * np.cos(omega0, dtype=np.float32)], dtype=np.float32)
    return poles, zeros

def _group_markers(vals: np.ndarray, tol: float = 1e-3, complex_vals: bool = True):
    """
    Group nearly-identical positions to show multiplicities (×m).
    Returns list of (representative_value, multiplicity).
    """
    vals = np.asarray(vals)
    used = np.zeros(len(vals), dtype=bool)
    groups = []
    for i, vi in enumerate(vals):
        if used[i]:
            continue
        idx = [i]
        for j in range(i + 1, len(vals)):
            if used[j]:
                continue
            vj = vals[j]
            d = np.abs(vi - vj) if complex_vals else np.abs(float(vi) - float(vj))
            if d <= tol:
                idx.append(j)
        used[idx] = True
        groups.append((vi, len(idx)))
    return groups


@demos_z_trafo_bp.route("/", methods=["GET"])
def page():
    defaults = {"norm_freq": 0.25, "a": 0.9}
    return render_template("demos/z_trafo.html", defaults=defaults)


@demos_z_trafo_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        norm_freq = float(np.clip(round(float(data.get("norm_freq", 0.25)), 3), 0.0, 1.0))
        a = float(np.clip(round(float(data.get("a", 0.9)), 3), 0.0, 1.0))
        omega0 = norm_freq * np.pi

        if _last["a"] == a and _last["norm_freq"] == norm_freq and _last["image"] is not None:
            return jsonify({"image": _last["image"]})

        # ----- time sequence -----
        x = _signal(_K, a, omega0)

        # ----- DTFT magnitude -----
        Xw = _dtft(a, omega0, _W)
        mag = np.abs(Xw, dtype=np.float32)
        if a == 1.0:
            mag = np.minimum(mag, 20.0, dtype=np.float32)
        else:
            mag = np.minimum(mag, np.float32(min(1.1 / (1.0 - a), 20.0)))
        peak_h = (20.0 if a == 1.0 else min(1.1 / max(1e-8, 1.0 - a), 20.0))

        # ----- pole/zero , roc -----
        poles, zeros = _pn_poles_zeros(a, omega0)
        pole_groups = _group_markers(poles, tol=1e-4, complex_vals=True)
        zero_groups = _group_markers(zeros, tol=1e-4, complex_vals=False)

        # Make the PN diagram the visual focus: larger figure + height ratios
        with plt.rc_context(RC_PARAMS):
            fig = plt.figure(figsize=(10.5, 8.2), layout="constrained")
            gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.0, 2.2])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[2, 0])

            # 1 - time domain (stem-like)
            ax1.vlines(_K, 0.0, x, linewidth=1)
            ax1.plot(_K, x, "o", markersize=3)
            ax1.set_title(r"Time sequence  $x[k]=a^k \cos(\omega_0 k)\,\varepsilon[k]$")
            ax1.set_xlabel("$k$")
            ax1.set_ylabel("$x[k]$")
            ax1.grid(True, alpha=0.25)

            # 2 - DTFT magnitude
            ax2.plot(_W_NORM, mag, linewidth=1)
            if norm_freq > 0.0:
                ax2.vlines([-norm_freq, norm_freq], 0.0, [peak_h, peak_h], linewidth=1)
                ax2.plot([-norm_freq, norm_freq], [peak_h, peak_h], "^", markersize=6)
            else:
                ax2.vlines([0.0], 0.0, [peak_h], linewidth=1)
                ax2.plot([0.0], [peak_h], "^", markersize=6)
            ax2.set_xlim(-1.0, 1.0)
            ax2.set_ylim(0.0, 20.0 if a == 1.0 else min(1.1 / max(1e-8, 1.0 - a), 20.0))
            ax2.set_xlabel(r"$\omega/\pi$")
            ax2.set_ylabel(r"$|X(\mathrm{e}^{\mathrm{j}\omega})|$")
            ax2.set_title(r"DTFT  $|X(\mathrm{e}^{\mathrm{j}\omega})|$")
            ax2.grid(True, alpha=0.25)

            # 3 - z: unit circle, z=a dashed, ROC shading, multiplicities
            ax3.axhline(0, color="tab:gray", linewidth=0.8)
            ax3.axvline(0, color="tab:gray", linewidth=0.8)

            # unit circle thicker for clarity
            uc = _UNIT_CIRCLE
            ax3.plot(uc[0], uc[1], linewidth=1.8, label="$|z| = 1$")

            # dashed poleradius circle z=a
            if a > 0.0:
                ax3.add_patch(
                    patches.Circle((0, 0), radius=a, fill=False, ls="--", lw=1.2, alpha=0.9, color="tab:gray", label="$|z| = a$")
                )

            # ROC shading: z > a
            R = 2.6
            if a >= 0.0:
                outer = patches.Circle((0, 0), radius=R, facecolor="tab:blue", alpha=0.06, edgecolor=None, label="ROC ($|z|>a$)")
                ax3.add_patch(outer)
                inner = patches.Circle((0, 0), radius=a, facecolor="white", edgecolor=None)
                ax3.add_patch(inner)

            # poles dashed rays
            for p, m in pole_groups:
                ax3.plot(p.real, p.imag, "x", markersize=10, mew=2, label=None)
                # pointer/ray
                ax3.plot([0.0, p.real], [0.0, p.imag], ls="--", lw=0.9, alpha=0.6, color="tab:gray")
                if m > 1:
                    ax3.text(p.real, p.imag, f"({m})", fontsize=10, ha="left", va="bottom", color="tab:red")

            # zeros 
            for z0, m in zero_groups:
                ax3.plot(z0, 0.0, "o", markersize=7, mfc="none", mec="tab:orange", mew=1.5, label=None)
                if m > 1:
                    ax3.text(z0, 0.0, f"({m})", fontsize=10, ha="left", va="bottom", color="tab:red")

            ax3.set_aspect("equal", adjustable="box")
            ax3.set_xlim(-R, R)
            ax3.set_ylim(-max(1.2, R if a > 0 else 1.2), R)  # keep origin centered
            ax3.set_xlabel(r"$\mathrm{Re}\{z\}$")
            ax3.set_ylabel(r"$\mathrm{Im}\{z\}$")
            ax3.set_title("Pole-Zero Plot & ROC ($z$-Plane)")
            # small legend
            ax3.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.8)
            ax3.grid(True, alpha=0.25)

            png = fig_to_base64(fig)

        _last.update({"a": a, "norm_freq": norm_freq, "image": png})
        return jsonify({"image": png})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
