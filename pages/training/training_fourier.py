from __future__ import annotations
"""Interactive Fourier training blueprint."""

from dataclasses import dataclass
from typing import Callable, Dict, Any, List
from flask import Blueprint, render_template, request, jsonify
import io
import base64
import random
import traceback
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt



@dataclass
class Signal:
    """Simple description of a time/frequency transform pair."""

    name: str
    time_fn: Callable[[np.ndarray], np.ndarray]
    freq_fn: Callable[[np.ndarray], np.ndarray]
    latex_time: str
    latex_freq: str


def _delta(arr: np.ndarray, pos: float, amp: complex = 1.0) -> np.ndarray:
    """Return an array with a stem-like Dirac delta at ``pos``."""
    d = np.zeros_like(arr, dtype=complex)
    idx = int(np.argmin(np.abs(arr - pos)))
    d[idx] = amp
    return d


def make_signal_pool(w0: float) -> Dict[str, Signal]:
    """Construct a dictionary of available base signals."""

    def rect(t: np.ndarray) -> np.ndarray:
        return np.where(np.abs(t) < 0.5, 1.0, 0.0)

    def tri(t: np.ndarray) -> np.ndarray:
        return np.maximum(1 - np.abs(t), 0.0)

    def sinc(t: np.ndarray) -> np.ndarray:
        return np.sinc(t)

    def sinc2(t: np.ndarray) -> np.ndarray:
        return np.sinc(t) ** 2

    def inv_t(t: np.ndarray) -> np.ndarray:
        return np.where(t != 0, 1.0 / t, 0.0)

    def sign(t: np.ndarray) -> np.ndarray:
        return np.sign(t)

    def cexp(t: np.ndarray) -> np.ndarray:
        return np.exp(1j * w0 * t)

    def cos_fn(t: np.ndarray) -> np.ndarray:
        return np.cos(w0 * t)

    def sin_fn(t: np.ndarray) -> np.ndarray:
        return np.sin(w0 * t)

    def t_sinc(t: np.ndarray) -> np.ndarray:
        return t * np.sinc(t)

    def F_rect(w: np.ndarray) -> np.ndarray:
        x = w / 2
        return np.where(w == 0, 1.0, np.sin(x) / x)

    def F_tri(w: np.ndarray) -> np.ndarray:
        x = w / 2
        return (np.sin(x) / x) ** 2

    def F_sinc(w: np.ndarray) -> np.ndarray:
        return np.where(np.abs(w) <= np.pi, 1.0, 0.0)

    def F_sinc2(w: np.ndarray) -> np.ndarray:
        return np.where(np.abs(w) <= 2 * np.pi, np.pi * (1 - np.abs(w) / (2 * np.pi)), 0.0)

    def F_inv_t(w: np.ndarray) -> np.ndarray:
        return -1j * np.pi * np.sign(w)

    def F_sign(w: np.ndarray) -> np.ndarray:
        eps = 1e-12
        return -2j / (w + eps)

    def F_cexp(w: np.ndarray) -> np.ndarray:
        return 2 * np.pi * _delta(w, w0)

    def F_cos(w: np.ndarray) -> np.ndarray:
        return np.pi * (_delta(w, w0) + _delta(w, -w0))

    def F_sin(w: np.ndarray) -> np.ndarray:
        return -1j * np.pi * (_delta(w, w0) - _delta(w, -w0))

    def F_t_sinc(w: np.ndarray) -> np.ndarray:
        return 1j * (_delta(w, np.pi) - _delta(w, -np.pi))

    return {
        "rect": Signal("rect", rect, F_rect, r"\mathrm{rect}(t)", r"2\,\mathrm{sinc}(\omega/2)"),
        "tri": Signal("tri", tri, F_tri, r"\mathrm{tri}(t)", r"(\mathrm{sinc}(\omega/2))^2"),
        "sinc": Signal("sinc", sinc, F_sinc, r"\mathrm{sinc}(t)", r"\mathbf{1}_{|\omega|\le \pi}"),
        "sinc2": Signal("sinc^2", sinc2, F_sinc2, r"\mathrm{sinc}^2(t)", r"\triangle(\omega/4\pi)"),
        "inv_t": Signal("1/t", inv_t, F_inv_t, r"1/t", r"-j\pi\,\mathrm{sgn}(\omega)"),
        "sign": Signal("sign", sign, F_sign, r"\mathrm{sgn}(t)", r"-\frac{2j}{\omega}"),
        "exp": Signal("exp", cexp, F_cexp, fr"e^{{j{w0}t}}", fr"2\pi\delta(\omega-{w0})"),
        "cos": Signal("cos", cos_fn, F_cos, fr"\cos({w0}t)", fr"\pi\,[\delta(\omega-{w0})+\delta(\omega+{w0})]"),
        "sin": Signal("sin", sin_fn, F_sin, fr"\sin({w0}t)", fr"-j\pi[\delta(\omega-{w0})-\delta(\omega+{w0})]"),
        "t_sinc": Signal("t*sinc", t_sinc, F_t_sinc, r"t\,\mathrm{sinc}(t)", r"j[\delta(\omega+\pi)-\delta(\omega-\pi)]"),
    }


def is_transform_pair(sig: Callable[[np.ndarray], np.ndarray],
                       X: Callable[[np.ndarray], np.ndarray]) -> bool:
    """Check numerically if two callables form a transform pair."""
    t = np.linspace(-10, 10, 1024)
    dt = t[1] - t[0]
    omega = np.fft.fftfreq(t.size, d=dt) * 2 * np.pi
    x = sig(t)
    X_num = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) * dt
    X_true = X(np.fft.fftshift(omega))
    return np.linalg.norm(X_num - X_true) < 1e-3

training_fourier_bp = Blueprint("training_fourier", __name__)

@training_fourier_bp.route("/")
def training_fourier() -> str:
    """Render the training page."""
    return render_template("training_fourier.html")

@training_fourier_bp.route("/generate", methods=["POST"])
def generate_problem() -> Any:
    """Return a newly generated quiz problem."""
    data = request.get_json(force=True)
    difficulty = data.get("difficulty", "EASY").upper()
    direction = data.get("direction", "TIME_TO_FREQ").upper()
    res = create_fourier_problem(difficulty, direction)
    status = 400 if "error" in res else 200
    return jsonify(res), status

@training_fourier_bp.route("/check_answer", methods=["POST"])
def check_answer() -> Dict[str, str]:
    """Simple correctness check."""
    data = request.get_json(force=True)
    feedback = (
        "Correct!" if data.get("selectedIndex") == data.get("correctIndex") else "Incorrect. Try again!"
    )
    return jsonify({"feedback": feedback})


def _parameter_ranges(difficulty: str) -> Dict[str, Any]:
    """Return parameter ranges for a difficulty level."""
    if difficulty == "EASY":
        return {
            "pool": ["rect", "tri", "sinc"],
            "shift": [-1, 0, 1],
            "scale": [1.0],
            "width": [1.0],
        }
    if difficulty == "MEDIUM":
        return {
            "pool": ["rect", "tri", "sinc", "cos", "sin", "exp"],
            "shift": list(range(-3, 4)),
            "scale": [0.5, 1.0, 2.0],
            "width": [0.5, 1.0, 2.0],
        }
    return {
        "pool": ["rect", "tri", "sinc", "cos", "sin", "exp", "sign", "inv_t", "t_sinc"],
        "shift": None,  # use uniform
        "scale": None,
        "width": None,
    }

def _sample_param(choices: List[float] | None, low: float, high: float) -> float:
    return random.uniform(low, high) if choices is None else float(random.choice(choices))


def _generate_distractors(
    direction: str,
    true_obj: np.ndarray,
    base_signal: Signal,
    t: np.ndarray,
    omega: np.ndarray,
    scale: float,
    width: float,
    shift: float,
) -> List[np.ndarray]:
    """Create three plausible distractors."""
    distractors: List[np.ndarray] = []

    if direction == "TIME_TO_FREQ":
        def no_width():
            return scale * base_signal.freq_fn(omega) * np.exp(-1j * omega * shift)

        def wrong_shift():
            return scale * width * base_signal.freq_fn(omega * width) * np.exp(1j * omega * shift)

        def random_amp():
            return random.uniform(0.5, 1.5) * true_obj

        def swap_mag_phase():
            mag = np.abs(true_obj)
            ang = np.angle(true_obj)
            return np.abs(ang) * np.exp(1j * mag)

        candidates = [no_width(), wrong_shift(), random_amp(), swap_mag_phase(), 1j * true_obj]
    else:
        def no_width_t():
            return scale * base_signal.time_fn(t - shift)

        def wrong_shift_t():
            return scale * base_signal.time_fn((t + shift) / width)

        def random_amp_t():
            return random.uniform(0.5, 1.5) * true_obj

        def swap_mag_phase_t():
            mag = np.abs(true_obj)
            ang = np.angle(true_obj)
            return np.abs(ang) * np.exp(1j * mag)

        candidates = [no_width_t(), wrong_shift_t(), random_amp_t(), swap_mag_phase_t(), -true_obj]

    random.shuffle(candidates)
    distractors = candidates[:3]
    return distractors


def create_fourier_problem(difficulty: str, direction: str) -> Dict[str, Any]:
    """Generate quiz problem data for the chosen direction and difficulty."""
    ranges = _parameter_ranges(difficulty)
    w0 = random.uniform(1.0, 3.0)
    pool = make_signal_pool(w0)

    name = random.choice(ranges["pool"])
    sig = pool[name]

    shift = _sample_param(ranges["shift"], -3, 3)
    scale = _sample_param(ranges["scale"], 0.3, 3)
    width = _sample_param(ranges["width"], 0.3, 3)

    t = np.linspace(-10, 10, 512)
    omega = np.linspace(-10, 10, 512)

    x = scale * sig.time_fn((t - shift) / width)
    X = scale * width * sig.freq_fn(omega * width) * np.exp(-1j * omega * shift)

    latex_time = fr"{scale:.2f}\,{sig.latex_time}\bigl((t-{shift:.2f})/{width:.2f}\bigr)"
    latex_freq = fr"{scale * width:.2f}\,{sig.latex_freq}\bigl(\omega\,{width:.2f}\bigr)e^{{-j\omega{shift:.2f}}}"
    props = []
    if abs(shift) > 1e-9:
        props.append("Time-shift ⇒ $e^{-j\\omega t_0}$ factor.")
    if abs(width - 1) > 1e-9:
        props.append("Scaling ⇒ width factor in amplitude and argument.")
    if abs(scale - 1) > 1e-9:
        props.append("Amplitude scales both domains.")
    property_msg = " ".join(props) or "Basic transform pair."

    if direction == "TIME_TO_FREQ":
        correct_obj = X
        distractors = _generate_distractors(direction, X, sig, t, omega, scale, width, shift)
    else:
        correct_obj = x
        distractors = _generate_distractors(direction, x, sig, t, omega, scale, width, shift)

    options = [correct_obj] + distractors
    indices = list(range(4))
    random.shuffle(indices)
    shuffled = [options[i] for i in indices]
    correct_idx = indices.index(0)

    try:
        fig = plt.figure(figsize=(10, 8))
        if direction == "TIME_TO_FREQ":
            gs = fig.add_gridspec(nrows=3, ncols=2)
            ax0 = fig.add_subplot(gs[0, :])
            ax0.plot(t, x.real)
            ax0.set_title("Signal")
            ax0.grid(True)
            ax0.set_xlim(t[0], t[-1])

            for i, opt in enumerate(shuffled):
                ax = fig.add_subplot(gs[1 + i // 2, i % 2])
                if np.count_nonzero(opt) <= 5:
                    ax.stem(omega, np.abs(opt), basefmt=" ", use_line_collection=True)
                    ax.stem(omega, np.angle(opt), basefmt=" ", linefmt="r--", markerfmt="ro", use_line_collection=True)
                else:
                    ax.plot(omega, np.abs(opt), label="|X|")
                    ax.plot(omega, np.angle(opt), "--", label="∠X")
                ax.set_xlim(omega[0], omega[-1])
                ax.set_title(f"Option {i + 1}")
                ax.grid(True)
        else:
            gs = fig.add_gridspec(nrows=3, ncols=2)
            ax0 = fig.add_subplot(gs[0, :])
            if np.count_nonzero(X) <= 5:
                ax0.stem(omega, np.abs(X), basefmt=" ", use_line_collection=True)
                ax0.stem(omega, np.angle(X), basefmt=" ", linefmt="r--", markerfmt="ro", use_line_collection=True)
            else:
                ax0.plot(omega, np.abs(X), label="|X|")
                ax0.plot(omega, np.angle(X), "--", label="∠X")
            ax0.set_title("Spectrum")
            ax0.grid(True)
            ax0.set_xlim(omega[0], omega[-1])

            for i, opt in enumerate(shuffled):
                ax = fig.add_subplot(gs[1 + i // 2, i % 2])
                ax.plot(t, opt.real)
                ax.set_xlim(t[0], t[-1])
                ax.set_title(f"Option {i + 1}")
                ax.grid(True)

        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
    except Exception:
        return {"error": traceback.format_exc()}

    return {
        "plot_data": plot_data,
        "correctIndex": correct_idx,
        "latex_time": latex_time,
        "latex_freq": latex_freq,
        "property_msg": property_msg,
    }
