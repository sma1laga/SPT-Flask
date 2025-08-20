from flask import Blueprint, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
from matplotlib import rcParams
rcParams["text.usetex"] = False
rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
from scipy.signal import convolve

from utils.img import fig_to_base64

demos_kapitel8_2_bp = Blueprint(
    "demos_kapitel8_2", __name__, template_folder="../../templates"
)

# ---------- helpers ----------
def bandpass_impulse(L, Omegag, Omega0, use_hann=True):
    """
    Windowed-sinc bandpass:
      h[k] = (2*Omegag/π) * sinc((Omegag/π)*(k-k0)) * cos(Omega0*(k-k0)) * w[k]
      with k0 = floor(L/2). Omegag is HALF-bandwidth, Omega0 is center.
    """
    k = np.arange(L, dtype=np.float32) - np.floor(L/2)
    h = (2*Omegag/np.pi) * np.sinc((Omegag/np.pi) * k) * np.cos(Omega0 * k)
    if use_hann:
        h *= np.hanning(L)
    return h.astype(np.float32)

def mag_response(h):
    h_dft = np.abs(np.fft.fft(h))
    # append first value due to periodicity
    h_dft = np.concatenate((h_dft, [h_dft[0]]))
    return h_dft

# ---------- routes ----------

@demos_kapitel8_2_bp.route("/", methods=["GET"])
def page():
    defaults = {
        "L": 128,                # 16, 32, 64, 128
        "Omega0_over_pi":0.3,    # center (π/2)
        "Omega_g_over_pi": 0.1,   # half-width
        "use_hann": False,
    }
    return render_template("demos/kapitel8_2.html", defaults=defaults)

@demos_kapitel8_2_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        L = int(data.get("L", 128))
        o0 = float(data.get("Omega0_over_pi", 0.3))   # center / π
        og = float(data.get("Omega_g_over_pi", 0.1))  # half-width / π
        use_hann = bool(data.get("use_hann", False))

        if L not in (16, 32, 64, 128):
            return jsonify(error="L must be one of 16, 32, 64, 128"), 400
        if not (0 <= o0 <= 1):
            return jsonify(error="Omega0/pi must be in [0,1]"), 400
        if not (0 <= og <= 0.5):
            return jsonify(error="Omega_g/pi must be in [0,0.5]"), 400

        # build test image
        M, N = 240, 416
        x = np.cos(np.arange(N, dtype=np.float32)**2 / 400)

        # design bandpass
        Omega0 = o0 * np.pi      # center
        Omegag = og * np.pi      # half-width
        h = bandpass_impulse(L, Omegag, Omega0, use_hann=use_hann)

        # filter image row-wise
        xf = convolve(x, h, 'same')
        xf = np.clip(xf, -1, 1) / 2 + 0.5

        # frequency response |H(e^{jΩ})| over 0..2pi
        H = mag_response(h)

        # ---------- plotting ----------
        fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.4))
        ax_x, ax_h, ax_xf, ax_h_dft = axs.flatten()

        # Originalbild
        ax_x.imshow(np.repeat(x[None], M, axis=0), cmap="gray", interpolation="none", aspect='auto')
        ax_x.set_title("Original Image")
        ax_x.set_ylabel("m")
        ax_x.set_xlabel("n")
        ax_x.tick_params(axis='both', which='both', left=False, labelleft=False, bottom=False, labelbottom=False)

        # h[n] (stem)
        ax_h.grid(True)
        ax_h.set_title("Band-Pass (Spatial Domain)")
        ax_h.set_xlabel("n")
        ax_h.set_ylabel("h[n]")
        ax_h.hlines(0, -1, L, color="black")
        markerline, _, _ = ax_h.stem(np.arange(L), h, basefmt='none')
        markerline.set_markerfacecolor('none')
        ax_h.set_xlim(-1, L)
        ax_h.set_ylim(-0.65, 1)

        # Filtered Image
        ax_xf.imshow(np.repeat(xf[None], M, axis=0), cmap="gray", interpolation="none", aspect='auto', vmin=0, vmax=1)
        ax_xf.set_title("Filtered Image")
        ax_xf.set_ylabel("m")
        ax_xf.set_xlabel("n")
        ax_xf.tick_params(axis='both', which='both', left=False, labelleft=False, bottom=False, labelbottom=False)

        ax_h_dft.grid(True)
        ax_h_dft.set_title("Band-Pass (Frequency Domain)")
        ax_h_dft.set_xlabel("Ω")
        ax_h_dft.set_ylabel("|H(jΩ)|")
        w = np.linspace(0, 2*np.pi, len(H), endpoint=True)
        ax_h_dft.plot(w, H)
        ax_h_dft.set_xlim(0, 2*np.pi)
        ax_h_dft.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax_h_dft.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax_h_dft.set_ylim(0, 2.2)

        fig.tight_layout()
        png = fig_to_base64(fig)
        return jsonify({"image": png})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=str(e)), 500
