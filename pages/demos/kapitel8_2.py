from flask import Blueprint, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve

from utils.img import fig_to_base64

demos_kapitel8_2_bp = Blueprint(
    "demos_kapitel8_2", __name__, template_folder="../../templates"
)

# ---------- helpers ----------

def make_grating(H=240, W=320, f_min=0.03*np.pi, f_max=0.9*np.pi):
    """Synthetic grating with increasing column frequency (like the nb)."""
    cols = np.arange(W, dtype=np.float32)[None, :]
    omega = np.linspace(f_min, f_max, W, dtype=np.float32)[None, :]
    img = np.sin(omega * cols)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img.astype(np.float32)

def hann(M):
    n = np.arange(M, dtype=np.float32)
    return 0.5 * (1 - np.cos(2*np.pi*n/(M-1))) if M > 1 else np.ones(1, dtype=np.float32)

def bandpass_impulse(M, Omegag, Omega0, use_hann=True):
    """
    Windowed-sinc bandpass:
      h[n] = (2*Omegag/π) * sinc((Omegag/π)*(n-n0)) * cos(Omega0*(n-n0)) * w[n]
      with n0 = ceil(M/2). Omegag is HALF-bandwidth, Omega0 is center.
    """
    n = np.arange(M, dtype=np.float32)
    n0 = int(np.ceil(M/2))
    x = (n - n0).astype(np.float32)
    h = (2*Omegag/np.pi) * np.sinc((Omegag/np.pi) * x) * np.cos(Omega0 * x)
    if use_hann:
        h *= hann(M)
    # normalize peak like nb visuals
    if np.max(np.abs(h)) > 0:
        h = h / np.max(np.abs(h))
    return h.astype(np.float32)

def filter_columns(img, h):
    """Convolve each row with h (filter along columns)."""
    H, W = img.shape
    y = np.stack([convolve(img[r], h, mode="same") for r in range(H)], axis=0)
    mn, mx = float(y.min()), float(y.max())
    if mx - mn > 1e-12:
        y = (y - mn) / (mx - mn)
    else:
        y = np.zeros_like(y)
    return y.astype(np.float32)

def mag_response(h, nfft=4096):
    H = np.abs(np.fft.fft(h, n=8192))
    H /= (H.max() + 1e-12)

    return H

# ---------- routes ----------

@demos_kapitel8_2_bp.route("/", methods=["GET"])
def page():
    defaults = {
        "M_L": 128,                # 16, 32, 64, 128
        "Omega0_over_pi": 0.50,    # center (π/2)
        "Omega_g_over_pi": 0.07,   # half-width
        "use_hann": True,
    }
    return render_template("demos/kapitel8_2.html", defaults=defaults)

@demos_kapitel8_2_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        M = int(data.get("M_L", 128))
        o0 = float(data.get("Omega0_over_pi", 0.50))   # center / π
        og = float(data.get("Omega_g_over_pi", 0.07))  # half-width / π
        use_hann = bool(data.get("use_hann", True))

        if M not in (16, 32, 64, 128):
            return jsonify(error="M_L must be one of 16, 32, 64, 128"), 400
        if not (0.0 <= o0 <= 1.0):
            return jsonify(error="Omega0/pi must be in [0,1]"), 400
        if not (0.01 <= og <= 0.15):
            return jsonify(error="Omega_g/pi must be in [0.01, 0.15]"), 400

        # build test image
        x_img = make_grating()

        # design bandpass
        Omega0 = o0 * np.pi      # center
        Omegag = og * np.pi      # half-width
        h = bandpass_impulse(M, Omegag, Omega0, use_hann=use_hann)

        # filter image along columns
        y_img = filter_columns(x_img, h)

        # frequency response |H(e^{jΩ})| over 0..2pi
        H = mag_response(h, nfft=8192)
        N = len(H)
        w = np.linspace(0, 2*np.pi, N, endpoint=False)

        # ---------- plotting ----------
        fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.4))
        ax_x, ax_h, ax_y, ax_H = axs.flatten()

        # Originalbild
        ax_x.imshow(x_img, cmap="gray", origin="upper", aspect="auto")
        ax_x.set_title("Originalbild")
        ax_x.axis("off")

        # h[n] (stem)
        ax_h.grid(True, alpha=0.25)
        ax_h.set_title("Bandpass (Ortsbereich)")
        ax_h.set_xlabel("n")
        ax_h.set_ylabel("h[n]")
        ax_h.hlines(0, 0, M+1, color="black", linewidth=1)
        markerline, stemlines, baseline = ax_h.stem(np.arange(1, M+1), h, basefmt='none')
        markerline.set_markerfacecolor('none')
        ax_h.set_xlim(0, M+1)
        ax_h.set_ylim(min(-0.1, 1.1*h.min()), max(0.1, 1.1*h.max()))

        # Gefiltertes Bild
        ax_y.imshow(y_img, cmap="gray", origin="upper", aspect="auto")
        ax_y.set_title("Gefiltertes Bild")
        ax_y.axis("off")

        ax_H.grid(True, alpha=0.25)
        ax_H.set_title("Bandpass (Frequenzbereich)")
        ax_H.set_xlabel("Ω")
        ax_H.set_ylabel("|H(e^{jΩ})|")
        ax_H.plot(w, H)
        ax_H.set_xlim(0, 2*np.pi)
        ax_H.set_ylim(0, 1.2*np.max(H) + 1e-12)
        ax_H.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax_H.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])

        fig.tight_layout()
        png = fig_to_base64(fig)
        return jsonify({"image": png})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=str(e)), 500
