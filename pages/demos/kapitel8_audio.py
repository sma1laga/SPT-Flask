# pages/demos/kapitel8_audio.py (faster,)
from flask import Blueprint, render_template, request, jsonify, current_app
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.signal import convolve, resample

from utils.img import fig_to_base64
from utils.audio import wav_data_url, read_mono_audio

demos_kapitel8_audio_bp = Blueprint(
    "demos_kapitel8_audio", __name__, template_folder="../../templates"
)

AUDIO_MAP = {
    "Fanfare": "fanfare.wav",
    "Piano":     "piano_mono32.wav",
}

L_OPTIONS = [16, 32, 64, 128]  # length L

def _audio_path(filename: str) -> str:
    return os.path.join(current_app.static_folder, "demos", "audio", filename)

@lru_cache(maxsize=8)
def _load_mono_float_cached(path: str):
    """Load+normalize audio once; cache by path."""
    fn = os.path.basename(path)
    fs, x = read_mono_audio(path)
    if "fanfare" in fn.lower():
        # take next power of 2 for fft performance
        n_2 = 2**int(np.ceil(np.log2(len(x))))
        x = np.pad(x, (0, n_2-len(x)))
    elif "piano" in fn.lower():
        # discard initial silence
        start = np.argmax(np.abs(x) > 0)
        x = x[start:len(x)//2]
        # resample to 16 kHz
        x = resample(x, len(x)//4)
        fs //= 4
        # assure power of 2 for fft performance
        n_2 = 2**int(np.log2(len(x)))
        x = x[:n_2]
        # fade in and out
        n_in, n_out = 100, n_2//8
        x[:n_in] *= np.hanning(2*n_in)[:n_in]
        x[-n_out:] *= np.hanning(2*n_out)[n_out:]
    return fs, x

def _lowpass_h(L: int, fg_norm: float, use_hann: bool):
    """
    Diskreter Tiefpass:
        h[k] = fg_norm * sinc(fg_norm * (k - k0)), k0 = ceil(L/2)
    fg_norm = Ωg/π in [0, 1]
    """
    k = np.arange(-np.ceil(L/2), L-np.ceil(L/2), dtype=np.float32)
    assert len(k) == L, f"Expected L={L}, got {len(k)}"
    h = fg_norm * np.sinc(fg_norm * k).astype(np.float32)
    if use_hann:
        h *= np.hanning(L).astype(np.float32)
    return h

def _abs_rfft(x):
    """Magnitude of real-valued FFT."""
    X = np.abs(np.fft.rfft(x))
    return X

@demos_kapitel8_audio_bp.route("/", methods=["GET"])
def page():
    defaults = {"x_type": "Fanfare", "L": 128, "fg": 0.2, "win": False}
    return render_template(
        "demos/kapitel8_audio.html",
        audio_options=list(AUDIO_MAP.keys()),
        defaults=defaults,
        L_options=L_OPTIONS
    )

@demos_kapitel8_audio_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        x_type = (data.get("x_type", "Fanfare")).strip()
        L      = int(data.get("L", 128))
        fg     = float(data.get("fg", 0.2))  # Ωg/π
        win    = bool(data.get("win"))

        if x_type not in AUDIO_MAP:
            return jsonify(error=f"Unknown x_type: {x_type}"), 400
        if L not in L_OPTIONS:
            return jsonify(error=f"Unsupported L: {L}"), 400
        if not (0.0 <= fg <= 1.0):
            return jsonify(error=f"fg (Ωg/π) must be in [0, 1], got {fg}"), 400

        path = _audio_path(AUDIO_MAP[x_type])
        if not os.path.exists(path):
            return jsonify(error=f"Audio file not found: {AUDIO_MAP[x_type]}", path=path), 500

        fs, x = _load_mono_float_cached(path)  # cached
        h     = _lowpass_h(L, fg, win)
        xf     = convolve(x, h)

        # ---------- plotting lightweight ----------
        fig, axs = plt.subplots(2, 2, figsize=(9.0, 5.0))
        ax_x, ax_h, ax_xf, ax_h_DFT = axs.flatten()

        # DFT(x)
        x_dft = _abs_rfft(x)
        ax_x.grid(True)
        ax_x.set_title("DFT of original Input")
        ax_x.set_xlabel("Ω")
        ax_x.set_ylabel("Magnitude")
        ax_x.plot(np.arange(len(x_dft)), x_dft, linewidth=0.5)
        ax_x.set_xlim(0, len(x_dft)-1)
        ax_x.set_xticks([0, len(x_dft)//2, len(x_dft)-1])
        ax_x.set_xticklabels(["0", "π/2", "π"])
        ax_x.set_ylim(0, 1000)

        ax_h.grid(True)
        ax_h.set_title("Low-Pass (Time Domain)")
        ax_h.set_xlabel("k")
        ax_h.set_ylabel("h[k]")
        ax_h.hlines(0, -1, L, color='black')
        ml2, _, _ = ax_h.stem(np.arange(L), h, basefmt='none')
        ml2.set_markerfacecolor('none')
        ax_h.set_xlim(-1, L)
        ax_h.set_ylim(-0.2, 1.1)

        xf_DFT = _abs_rfft(xf)
        ax_xf.grid(True)
        ax_xf.set_title("DFT of filtered Signal")
        ax_xf.set_xlabel("Ω")
        ax_xf.set_ylabel("Magnitude")
        ax_xf.plot(np.arange(len(xf_DFT)), xf_DFT, linewidth=0.5)
        ax_xf.set_xlim(0, len(xf_DFT)-1)
        ax_xf.set_xticks([0, len(xf_DFT)//2, len(xf_DFT)-1])
        ax_xf.set_xticklabels(["0", "π/2", "π"])
        ax_xf.set_ylim(0, 1000)

        # |H(jΩ)| (length L -> tiny; plot full rFFT)
        h_DFT = _abs_rfft(h)
        ax_h_DFT.grid(True)
        ax_h_DFT.set_title("Low-Pass (Frequency Domain)")
        ax_h_DFT.set_xlabel("Ω")
        ax_h_DFT.set_ylabel("|H(jΩ)|")
        ax_h_DFT.plot(np.arange(len(h_DFT)), h_DFT, linewidth=1.0)
        ax_h_DFT.set_xlim(0, len(h_DFT)-1)
        ax_h_DFT.set_xticks([0, len(h_DFT)//2, len(h_DFT)-1])
        ax_h_DFT.set_xticklabels(["0", "π/2", "π"])
        ax_h_DFT.set_ylim(0, 1.2)

        fig.tight_layout(h_pad=2.5, pad=2.5)
        png = fig_to_base64(fig)

        return jsonify({
            "image": png,
            "x_audio": wav_data_url(x, fs),
            "y_audio": wav_data_url(xf, fs)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
