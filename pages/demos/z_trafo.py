# pages/demos/z_trafo.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from utils.img import fig_to_base64

RC_PARAMS = {
    "agg.path.chunksize": 20000,
    "path.simplify": True,
    "path.simplify_threshold": 0.5,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "font.size": 13,
}

demos_z_trafo_bp = Blueprint(
    "demos_z_trafo", __name__, template_folder="././templates"
)

#           cached grids 
_K = np.arange(-2, 41, dtype=np.int32)
_OMEGA = np.linspace(-np.pi, np.pi, 1024, dtype=np.float32)
_OMEGA_NORM = _OMEGA / np.pi
_THETA = np.linspace(0.0, 2.0 * np.pi, 512, dtype=np.float32)
_UNIT_CIRCLE = np.stack([np.cos(_THETA), np.sin(_THETA)], axis=0)

_last = {"a": None, "norm_freq": None, "image": None}


# ----------- math -----------
def _signal(k: np.ndarray, a: float, w: float) -> np.ndarray:
    # x[k] = a^k * cos(w k) * u[k]
    x = np.zeros(len(k), dtype=np.float32)
    x[k >= 0] = np.power(a, k[k >= 0], dtype=np.float32) * np.cos(w * k[k >= 0], dtype=np.float32)
    return x

def _dtft(a: float, w: float, Omega: np.ndarray) -> np.ndarray:
    if a > 1: return None
    # X(e^{jω}) = 1/2 * [ 1/(1 - a e^{-j(Ω-w)}) + 1/(1 - a e^{-j(Ω+w)}) ]
    jw1 = -1j * (Omega - w)
    jw2 = -1j * (Omega + w)
    z1 = (1.0 - a * np.exp(jw1, dtype=np.complex64)).astype(np.complex64)
    z2 = (1.0 - a * np.exp(jw2, dtype=np.complex64)).astype(np.complex64)
    z1[np.isclose(z1, 0.0)] = 1e8  # avoid div by zero
    z2[np.isclose(z2, 0.0)] = 1e8
    return np.abs(0.5 * (1 / z1 + 1 / z2))

def _dtft_peaks(norm_freq, a):
    if a!=1: # no singularities or DTFT not defined
        dtft_pos, dtft_val = [], []
    else:
        peak_height = np.pi if norm_freq in [0, 1] else np.pi/2 # NOTE: corrected values (1/2 missing in notebooks)
        if norm_freq==0:
            dtft_pos, dtft_val = [0], [peak_height]
        else:
            dtft_pos = [-norm_freq, norm_freq]
            dtft_val = 2*[peak_height]
    return np.array(dtft_pos), np.array(dtft_val)

def _pn_poles_zeros(a: float, w: float):
    # poles at a e^{±j Omega}; two zeros at z = {0, a cos(w)}
    poles, zeros = np.array([], dtype=np.complex64), np.array([], dtype=np.float32)
    if np.abs(a) > 0:
        poles = a * np.exp(1j * np.array([w, -w], dtype=np.float32)).astype(np.complex64)
        zeros = np.array([0, a * np.cos(w, dtype=np.float32)], dtype=np.float32)
    return poles, zeros

def _group_markers(vals: np.ndarray, tol: float = 1e-3, complex_vals: bool = True):
    """
    Group nearly-identical positions to show multiplicities (m).
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
        a = float(np.clip(round(float(data.get("a", 0.9)), 3), 0.0, 2.0))
        w = norm_freq * np.pi

        if _last["a"] == a and _last["norm_freq"] == norm_freq and _last["image"] is not None:
            return jsonify({"image": _last["image"]})

        # ----- time sequence -----
        x = _signal(_K, a, w)

        # ----- DTFT magnitude -----
        mag = _dtft(a, w, _OMEGA)

        # ----- pole/zero , roc -----
        poles, zeros = _pn_poles_zeros(a, w)
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
            ax1.axhline(0, color="tab:gray", linewidth=0.8)
            ax1.axvline(0, color="tab:gray", linewidth=0.8)
            ax1.vlines(_K, 0.0, x, linewidth=1)
            ax1.plot(_K, x, "o", markersize=3)
            ax1.set_title(r"Time sequence  $x[k]=a^k \cos(\omega k)\varepsilon[k]$")
            ax1.set_xlabel("$k$")
            ax1.set_ylabel("$x[k]$")
            ax1.grid(True, alpha=0.25)
            xlim = (_K[0]-0.5, _K[-1]+0.5)
            ylim_abs = max(1.1, np.max(np.abs(x))*1.1)
            ax1.set_xlim(*xlim)
            ax1.set_ylim(-ylim_abs, ylim_abs)

            # 2 - DTFT magnitude
            xlim = 1
            if a > 1: # DTFT not defined
                ylim = [0, 1]
                ax2.set_ylim(*ylim)
                ax2.plot([-xlim,xlim], ylim, [-xlim,xlim], ylim[::-1], color='tab:blue')
            else:
                ax2.plot(_OMEGA_NORM, mag, linewidth=1)
                if a == 1:
                    X_peaks_pos, X_peaks_val = _dtft_peaks(norm_freq, a)
                    ax2.vlines(X_peaks_pos, 0.0, X_peaks_val, linewidth=1, color="tab:orange")
                    ax2.plot(X_peaks_pos, X_peaks_val, "^", markersize=7, color="tab:orange")
                ax2.set_ylim((0, 11) if a==1 else (0, min(1/(1-a) * 1.1 , 11)))
            ax2.set_xlim(-xlim, xlim)
            ax2.set_xticks(np.arange(-xlim, xlim+0.1, 0.5))
            ax2.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
            ax2.set_xlabel(r"$\Omega$")
            ax2.set_ylabel(r"$|X(\mathrm{e}^{\mathrm{j}\Omega})|$")
            ax2.set_title(r"DTFT  $|X(\mathrm{e}^{\mathrm{j}\Omega})|$")
            ax2.grid(True, alpha=0.25)

            # 3 - z: unit circle, z=a dashed, ROC shading, multiplicities
            ax3.axhline(0, color="tab:gray", linewidth=0.8)
            ax3.axvline(0, color="tab:gray", linewidth=0.8)

            # unit circle thicker for clarity
            uc = _UNIT_CIRCLE
            ax3.plot(uc[0], uc[1], color="tab:gray", linewidth=0.8, label="$|z| = 1$")

            # dashed poleradius circle z=a
            if a > 0.0:
                ax3.add_patch(
                    patches.Circle((0, 0), radius=a, fill=False, ls="--", lw=1.2, alpha=0.9, color="tab:gray", label="$|z| = a$")
                )

            # ROC shading: z > a
            ylim_abs = max(1.1, np.abs(a)*1.1)
            xlim_abs = ylim_abs * 2.5
            if a >= 0.0:
                outer = patches.Rectangle((-xlim_abs, -ylim_abs), 2*xlim_abs, 2*ylim_abs, facecolor="tab:blue", alpha=0.06, edgecolor=None, label="ROC ($|z|>a$)")
                ax3.add_patch(outer)
                inner = patches.Circle((0, 0), radius=a, facecolor="white", edgecolor=None)
                ax3.add_patch(inner)

            # poles dashed rays
            for p, m in pole_groups:
                ax3.plot(p.real, p.imag, "x", markersize=10, color="tab:orange", mew=2, label=None)
                if m > 1:
                    ax3.text(p.real+.1, p.imag+.1, f"({m})", ha="left", va="bottom", color="tab:orange")

            # zeros
            for z0, m in zero_groups:
                ax3.plot(z0, 0.0, "o", markersize=10, mfc="none", mec="tab:blue", mew=1.5, label=None)
                if m > 1:
                    ax3.text(z0-.1, .1, f"({m})", ha="right", va="bottom", color="tab:blue")

            ax3.set_aspect("equal", adjustable="box")
            ax3.set_xlim(-xlim_abs, xlim_abs)
            ax3.set_ylim(-ylim_abs, ylim_abs)  # keep origin centered
            ax3.xaxis.set_major_locator(MultipleLocator(1))
            ax3.yaxis.set_major_locator(MultipleLocator(1))
            ax3.set_xlabel(r"$\mathrm{Re}\{z\}$")
            ax3.set_ylabel(r"$\mathrm{Im}\{z\}$")
            ax3.set_title("Pole-Zero Plot ($z$-Plane)")
            # small legend
            ax3.legend(loc="upper right", frameon=True, framealpha=0.8)
            ax3.grid(True, alpha=0.25)

            png = fig_to_base64(fig)

        _last.update({"a": a, "norm_freq": norm_freq, "image": png})
        return jsonify({"image": png})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
