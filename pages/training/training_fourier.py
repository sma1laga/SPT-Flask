
from __future__ import annotations
"""
Fourier-Training
- Exam-like layout: 3 columns (|Y|, phase, y(t)) and 4 rows of answer optionss.s
- The given object (time signal or spectrum) is drawn in **black** in the top row.
- Each option row shows the student answer in **green** (magnitude+phase or time).
- Clickable areas now come from the server (hit_boxes) so the HTML overlay is robust WORK
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple
from flask import Blueprint, render_template, request, jsonify
import io, base64, random, traceback, inspect, math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- utilities ----------

def _sinc_rad(x: np.ndarray) -> np.ndarray:
    """np.sinc normalized so that sinc_rad(0)=1, argument in radians."""
    return np.sinc(x / math.pi)

_STEM_HAS_LINE_COLLECTION = "use_line_collection" in inspect.signature(plt.stem).parameters
def _stem(ax: matplotlib.axes.Axes, x, y, **kwargs):
    if _STEM_HAS_LINE_COLLECTION:
        kwargs.setdefault("use_line_collection", True)
    return ax.stem(x, y, **kwargs)

def _delta(arr: np.ndarray, pos: float, amp: complex = 1.0) -> np.ndarray:
    d = np.zeros_like(arr, dtype=complex)
    idx = int(np.argmin(np.abs(arr - pos)))
    d[idx] = amp
    return d

@dataclass
class Signal:
    """Simple time/frequency transform pair x(t) <-> X(w)."""
    name: str
    time_fn: Callable[[np.ndarray], np.ndarray]
    freq_fn: Callable[[np.ndarray], np.ndarray]
    latex_time: str
    latex_freq: str

def make_signal_pool(w0: float) -> Dict[str, Signal]:
    """Pool of archetypes used in the exams."""
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

    # Frequency responses
    def F_rect(w: np.ndarray) -> np.ndarray:
        return _sinc_rad(w / 2)

    def F_tri(w: np.ndarray) -> np.ndarray:
        return _sinc_rad(w / 2) ** 2

    def F_sinc(w: np.ndarray) -> np.ndarray:
        return np.where(np.abs(w) <= np.pi, 1.0, 0.0)

    def F_sinc2(w: np.ndarray) -> np.ndarray:
        return np.where(np.abs(w) <= 2 * np.pi, 1 - np.abs(w) / (2 * np.pi), 0.0)

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
        "rect":  Signal("rect",  rect,  F_rect,  r"\mathrm{rect}(t)",         r"\mathrm{sinc}(\omega/2\pi)"),
        "tri":   Signal("tri",   tri,   F_tri,   r"\mathrm{tri}(t)",          r"\mathrm{sinc}^2(\omega/2\pi)"),
        "sinc":  Signal("sinc",  sinc,  F_sinc,  r"\mathrm{sinc}(t)",         r"\mathbf{1}_{|\omega|\le \pi}"),
        "sinc2": Signal("sinc^2",sinc2, F_sinc2, r"\mathrm{sinc}^2(t)",       r"\max\{1-|{\omega}|/2\pi,0\}"),
        "inv_t": Signal("1/t",   inv_t, F_inv_t, r"1/t",                      r"-j\pi\,\mathrm{sgn}(\omega)"),
        "sign":  Signal("sign",  sign,  F_sign,  r"\mathrm{sgn}(t)",          r"-\frac{2j}{\omega}"),
        "exp":   Signal("exp",   cexp,  F_cexp,  fr"e^{{j{w0}t}}",            fr"2\pi\delta(\omega-{w0})"),
        "cos":   Signal("cos",   cos_fn,F_cos,   fr"\cos({w0}t)",             fr"\pi[\delta(\omega\!-\!{w0})+\delta(\omega\!+\!{w0})]"),
        "sin":   Signal("sin",   sin_fn,F_sin,   fr"\sin({w0}t)",             fr"-j\pi[\delta(\omega\!-\!{w0})-\delta(\omega\!+\!{w0})]"),
        "t_sinc":Signal("t*sinc",t_sinc,F_t_sinc,r"t\,\mathrm{sinc}(t)",      r"j[\delta(\omega+\pi)-\delta(\omega-\pi)]"),
    }

# ---------- difficulty parameterization ----------
# Adaptable @Paul

def _parameter_ranges(difficulty: str) -> Dict[str, Any]:
    if difficulty == "EASY":
        return {"pool": ["rect", "tri", "sinc", "cos"], "shift": [-1, 0, 1], "scale": [1.0], "width": [1.0]}
    if difficulty == "MEDIUM":
        return {"pool": ["rect", "tri", "sinc", "sinc2", "cos", "sin"], "shift": list(range(-3, 4)),
                "scale": [0.5, 1.0, 2.0], "width": [0.5, 1.0, 2.0]}
    return {"pool": ["rect","tri","sinc","sinc2","cos","sin","sign","inv_t","t_sinc"],
            "shift": None, "scale": None, "width": None}

def _sample_param(choices: List[float] | None, low: float, high: float) -> float:
    return random.uniform(low, high) if choices is None else float(random.choice(choices))

# ---------- plotting helpers ----------

_GREEN = "#2e8b57"

def _format_time_axis(ax: matplotlib.axes.Axes, t: np.ndarray):
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.35)
    ax.set_title(r"$y(t)$", fontsize=11)
    ax.set_xlabel(r"$t \rightarrow$", fontsize=10)

def _format_mag_axis(ax: matplotlib.axes.Axes, omega: np.ndarray):
    ax.set_xlim(omega[0], omega[-1])
    ax.grid(True, alpha=0.35)
    ax.set_title(r"$|Y(j\omega)|$", fontsize=11)
    # show ticks as w/pi
    xt = np.array([-2*math.pi, -math.pi, 0, math.pi, 2*math.pi])
    ax.set_xticks(xt)
    ax.set_xticklabels([r"$-2$", r"$-1$", r"$0$", r"$1$", r"$2$"])
    ax.set_xlabel(r"$\omega/\pi \rightarrow$", fontsize=10)

def _format_phase_axis(ax: matplotlib.axes.Axes, omega: np.ndarray):
    ax.set_xlim(omega[0], omega[-1])
    ax.grid(True, alpha=0.35)
    ax.set_title(r"$\varphi(j\omega)$", fontsize=11)
    ax.set_ylim(-math.pi*1.1, math.pi*1.1)
    # reference dashed lines at +-pi and 0
    ax.axhline(math.pi, color="k", ls="--", lw=0.7, alpha=0.4)
    ax.axhline(0, color="k", ls="--", lw=0.7, alpha=0.35)
    ax.axhline(-math.pi, color="k", ls="--", lw=0.7, alpha=0.4)
    xt = np.array([-2*math.pi, -math.pi, 0, math.pi, 2*math.pi])
    ax.set_xticks(xt)
    ax.set_xticklabels([r"$-2$", r"$-1$", r"$0$", r"$1$", r"$2$"])
    ax.set_xlabel(r"$\omega/\pi \rightarrow$", fontsize=10)

def _plot_spec(ax_mag, ax_ph, omega, X, color="k", heavy=False):
    if np.count_nonzero(X) <= 10:  # deltas → stem
        lc = _stem(ax_mag, omega, np.abs(X), basefmt=" ")
        try:
            lc.markerline.set_color(color)
            for ln in (lc.stemlines if hasattr(lc, "stemlines") else [lc]): ln.set_color(color)
        except Exception:
            pass
        lc2 = _stem(ax_ph, omega, np.angle(X), basefmt=" ")
        try:
            lc2.markerline.set_color(color)
            for ln in (lc2.stemlines if hasattr(lc2, "stemlines") else [lc2]): ln.set_color(color)
        except Exception:
            pass
    else:
        lw = 2.5 if heavy else 1.8
        ax_mag.plot(omega, np.abs(X), color=color, lw=lw)
        ang = np.unwrap(np.angle(X))
        # force to [-pi,pi] where apropriate for exam look
        ang = np.mod(ang + math.pi, 2*math.pi) - math.pi
        ax_ph.plot(omega, ang, color=color, lw=lw)

def _rect_union(a: Tuple[float,float,float,float],
                b: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    """Return union of two rects (x0,y0,w,h) in figure normalized cooords."""
    x0 = min(a[0], b[0]); y0 = min(a[1], b[1])
    x1 = max(a[0]+a[2], b[0]+b[2]); y1 = max(a[1]+a[3], b[1]+b[3])
    return (x0, y0, x1-x0, y1-y0)

# ---------- Blueprint ----------

training_fourier_bp = Blueprint("training_fourier", __name__)

@training_fourier_bp.route("/")
def training_fourier() -> str:
    return render_template("training_fourier.html")

@training_fourier_bp.route("/generate", methods=["POST"])
def generate_problem() -> Any:
    data = request.get_json(force=True)
    difficulty = data.get("difficulty", "EASY").upper()
    direction  = data.get("direction", "TIME_TO_FREQ").upper()
    res = create_fourier_problem(difficulty, direction)
    status = 400 if "error" in res else 200
    return jsonify(res), status

@training_fourier_bp.route("/check_answer", methods=["POST"])
def check_answer() -> Dict[str, str]:
    data = request.get_json(force=True)
    feedback = "Correct!" if data.get("selectedIndex") == data.get("correctIndex") else "Incorrect. Try again!"
    return jsonify({"feedback": feedback})

# ---------- core generation ----------


def _generate_distractors(direction: str, true_obj: np.ndarray, base_signal: Signal,
                          t: np.ndarray, omega: np.ndarray,
                          scale: float, width: float, shift: float) -> List[np.ndarray]:
    """
    Create 3 strong, exam-style distractors:
    - integer time-shift errors (+-1, +-2)
    - phase offsets (+-pi), wrong sign, or missed linear phase
    - wrong width / amplitude scaling
    - for impulses (cos/sin): wrong line positions or wrong imaginary sign
    """
    cand: List[np.ndarray] = []

    def uniq_push(arr):
        # Keep only distractors that differ visibly (L2) from those already in cand and from the truth
        def far(a,b):
            da = np.linalg.norm((a - b).reshape(-1)) / (np.linalg.norm(b.reshape(-1)) + 1e-9)
            return da > 0.12  # like 12% relative difference..
        if all(far(arr, c) for c in ([true_obj] + cand)):
            cand.append(arr)

    if direction == "TIME_TO_FREQ":
        Xtrue = scale*width*base_signal.freq_fn(omega*width) * np.exp(-1j * omega * shift)

        # 1) Miss linear phase (forget shift)
        uniq_push(scale*width*base_signal.freq_fn(omega*width))

        # 2) Wrong shift by +-1 or +-2
        k = random.choice([1, 2]) * random.choice([-1, 1])
        uniq_push(scale*width*base_signal.freq_fn(omega*width) * np.exp(-1j * omega * (shift + k)))

        # 3) Phase offset by pi (flip sign)
        uniq_push(-Xtrue)

        # 4) Wrong width (+-1 step if possible)
        width_alt = width
        for delta in [1.0, -1.0, 2.0]:
            if width + delta in [0.5, 1.0, 2.0, 3.0]:
                width_alt = width + delta
                break
        uniq_push(scale*width_alt*base_signal.freq_fn(omega*width_alt) * np.exp(-1j*omega*shift))

        # 5) Wrong amplitude (forget "·width")
        uniq_push(scale*base_signal.freq_fn(omega*width) * np.exp(-1j*omega*shift))

        # 6) Special cases: impulses and odd signals
        name = base_signal.name
        if name in ("cos", "sin"):
            # move impulses to wrong freq: +-(w0 +- pi)
            # estimate w0 from max of |X|
            idx = np.argmax(np.abs(true_obj))
            w0_est = abs(omega[idx])
            wrong_w0 = w0_est + random.choice([-math.pi, math.pi])
            # construct delta-like lines with small Gaussian blobs so difference shows in raster
            amp = 1.0
            if name == "cos":
                # cos has real equal spikes
                Xw = np.exp(-0.5*((omega-wrong_w0)/0.05)**2) + np.exp(-0.5*((omega+wrong_w0)/0.05)**2)
                uniq_push(np.pi*Xw)
            else:
                # sin has imaginary opposite spikes -> flip sign to mimic wrong j sign
                Xw = 1j*(np.exp(-0.5*((omega-wrong_w0)/0.05)**2) - np.exp(-0.5*((omega+wrong_w0)/0.05)**2))
                uniq_push(-Xw)

    else:
        # FREQ_TO_TIME: make timedomain distractors
        xtrue = scale * base_signal.time_fn((t - shift) / width)
        # 1) Wrong integer shift
        k = random.choice([1, 2]) * random.choice([-1, 1])
        uniq_push(scale * base_signal.time_fn((t - (shift + k)) / width))
        # 2) Wrong width
        width_alt = width
        for delta in [1.0, -1.0, 2.0]:
            if width + delta in [0.5, 1.0, 2.0, 3.0]:
                width_alt = width + delta
                break
        uniq_push(scale * base_signal.time_fn((t - shift) / width_alt))
        # 3) Wrong amplitude
        uniq_push(2.0 * xtrue if abs(scale) <= 1.5 else 0.5 * xtrue)
        # 4) Negated
        uniq_push(-xtrue)
        # 5) Missed shift (t0 = 0)
        uniq_push(scale * base_signal.time_fn(t / width))

    random.shuffle(cand)
    return cand[:3]

def create_fourier_problem(difficulty: str, direction: str) -> Dict[str, Any]:
    try:
        ranges = _parameter_ranges(difficulty)
        w0 = random.choice([0.5*math.pi, 1.0*math.pi, 1.5*math.pi, 2.0*math.pi])
        pool = make_signal_pool(w0)

        name = random.choice(ranges["pool"])
        sig = pool[name]

        shift = int(round(_sample_param(ranges["shift"], -3, 3)))
        scale = _sample_param(ranges["scale"], 0.4, 2.5)
        width = _sample_param(ranges["width"], 0.6, 2.2)

        # axes domains: time roughly [-4,4], frequency roughly [-2pi,2pi]
        t = np.linspace(-4, 4, 800)
        omega = np.linspace(-2*math.pi, 2*math.pi, 900)

        x = scale * sig.time_fn((t - shift) / width)
        X = scale * width * sig.freq_fn(omega * width) * np.exp(-1j * omega * shift)

        latex_time = fr"{scale:.2f}\,{sig.latex_time}\Big((t-{shift:.2f})/{width:.2f}\Big)"
        latex_freq = fr"{scale*width:.2f}\,{sig.latex_freq}\big(\omega\,{width:.2f}\big)e^{{-j\omega {shift:.2f}}}"

        # Build options (correct + 3 distractors)
        true_spec = X
        true_time = x
        if direction == "TIME_TO_FREQ":
            correct_obj = true_spec
            distractors = _generate_distractors(direction, true_spec, sig, t, omega, scale, width, shift)
            options = [correct_obj] + distractors
        else:
            correct_obj = true_time
            distractors = _generate_distractors(direction, true_time, sig, t, omega, scale, width, shift)
            options = [correct_obj] + distractors

        indices = list(range(4))
        random.shuffle(indices)
        shuffled = [options[i] for i in indices]
        correct_idx = indices.index(0)

        # ------- Figure creation -------
        fig = plt.figure(figsize=(10.2, 9.2), layout="constrained")
        # grid: rows = 1(header given) + 4(option rows), cols = 3 (|Y|, phase, y)
        gs = fig.add_gridspec(nrows=5, ncols=3, height_ratios=[1.1, 1, 1, 1, 1])

        # row 0: GIVEN (black)
        ax_mag_g = fig.add_subplot(gs[0, 0])
        ax_ph_g  = fig.add_subplot(gs[0, 1])
        ax_t_g   = fig.add_subplot(gs[0, 2])

        if direction == "TIME_TO_FREQ":
            # show given y(t) only
            ax_mag_g.axis("off"); ax_ph_g.axis("off")
            ax_t_g.plot(t, true_time.real, color="k", lw=2.2)
            _format_time_axis(ax_t_g, t)
            ax_t_g.set_title(r"given $y(t)$", fontsize=11)
        else:
            _plot_spec(ax_mag_g, ax_ph_g, omega, true_spec, color="k", heavy=True)
            _format_mag_axis(ax_mag_g, omega)
            _format_phase_axis(ax_ph_g, omega)
            ax_t_g.axis("off")
            ax_mag_g.set_title(r"given $|Y(j\omega)|$", fontsize=11)
            ax_ph_g.set_title(r"given $\varphi(j\omega)$", fontsize=11)

        # rows 1..4: answer options in GREEN
        hit_boxes: List[Tuple[float,float,float,float]] = []
        for i in range(4):
            ax_mag = fig.add_subplot(gs[1 + i, 0])
            ax_ph  = fig.add_subplot(gs[1 + i, 1])
            ax_t   = fig.add_subplot(gs[1 + i, 2])

            if direction == "TIME_TO_FREQ":
                # draw the option spectrum in green, and faint gray reference of the given y(t) at right
                _plot_spec(ax_mag, ax_ph, omega, shuffled[i], color=_GREEN, heavy=True)
                _format_mag_axis(ax_mag, omega)
                _format_phase_axis(ax_ph, omega)

                ax_t.plot(t, true_time.real, color="0.5", lw=1.2, ls="--", alpha=0.8)
                _format_time_axis(ax_t, t)
                ax_t.set_title(fr"$\mathcal{{O}}_{i+1}$", fontsize=11)

                # hit box: union of mag+phase axes
                r1 = ax_mag.get_position().bounds
                r2 = ax_ph.get_position().bounds
                hit_boxes.append(_rect_union(r1, r2))

            else:  # FREQ_TO_TIME
                # faint gray reference of given spectrum on left
                _plot_spec(ax_mag, ax_ph, omega, true_spec, color="0.6", heavy=False)
                _format_mag_axis(ax_mag, omega)
                _format_phase_axis(ax_ph, omega)

                ax_t.plot(t, shuffled[i].real, color=_GREEN, lw=2.3)
                _format_time_axis(ax_t, t)
                ax_t.set_title(fr"$\mathcal{{O}}_{i+1}$", fontsize=11)

                # hit box: just the time axis area (rightmost col)
                hit_boxes.append(ax_t.get_position().bounds)

        # encode
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=140)
        plot_data = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)

        # property message (short)
        props = []
        if abs(shift) > 1e-9: props.append("Time shift ⇒ $e^{-j\\omega t_0}$ in $Y(j\\omega)$.")
        if abs(width - 1) > 1e-9: props.append("Time scaling ⇒ amplitude·width and argument scaling.")
        if name in ("cos", "sin"): props.append("Trigonometric ⇒ impulses at +-w₀.")
        property_msg = " ".join(props) or "Basic transform pair."

        return {
            "plot_data": plot_data,
            "correctIndex": correct_idx,
            "latex_time": latex_time,
            "latex_freq": latex_freq,
            "property_msg": property_msg,
            "hit_boxes": hit_boxes,  # figure-normalized rectangles for clickable overlay
        }
    except Exception:
        return {"error": traceback.format_exc()}
