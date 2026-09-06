"""Fourier training: match a signal to its spectrum, or the other way round.

Conventions follow the rest of the toolkit: ``si(t) = sin(t)/t`` as in
``utils/math_utils.py``, and the transform is

    X(jw) = integral x(t) e^{-jwt} dt

Line spectra are part of the exam material, so a waveform carries a continuous
part *and* a list of weighted Dirac impulses. Both the algebra and the plotting
treat the two together, which is what keeps cos/sin/e^{jw0t} drawable.
"""

from __future__ import annotations

import base64
import io
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Blueprint, jsonify, render_template, request

# ---------------------------------------------------------------- constants --

T_LIMIT = 6.0
W_LIMIT = 2.0 * math.pi
N_SAMPLES = 901           # odd, so w = 0 and t = 0 are sampled
GIVEN_COLOUR = "#111111"
ANSWER_COLOUR = "#2e8b57"
IMAG_COLOUR = "#7f8c8d"
PHASE_FLOOR = 0.02        # fraction of the peak below which the phase is noise

Complex = complex


def _si(x: np.ndarray) -> np.ndarray:
    """sin(x)/x with si(0) = 1 -- the toolkit's si, not the normalised sinc."""
    return np.sinc(np.asarray(x, dtype=float) / np.pi)


# ----------------------------------------------------------------- waveform --

@dataclass(frozen=True)
class Waveform:
    """A signal as a sampled continuous part plus weighted Dirac impulses."""

    values: np.ndarray
    impulses: Tuple[Tuple[float, Complex], ...] = ()

    @property
    def has_impulses(self) -> bool:
        return bool(self.impulses)

    def peak(self) -> float:
        """Largest magnitude that will be drawn, impulses included."""
        continuous = float(np.max(np.abs(self.values))) if self.values.size else 0.0
        impulse = max((abs(w) for _, w in self.impulses), default=0.0)
        return max(continuous, impulse)

    def scaled(self, factor: Complex) -> "Waveform":
        return Waveform(self.values * factor,
                        tuple((p, w * factor) for p, w in self.impulses))

    def conjugated(self) -> "Waveform":
        return Waveform(np.conj(self.values),
                        tuple((p, np.conj(w)) for p, w in self.impulses))

    def rasterised(self, grid: np.ndarray) -> np.ndarray:
        """Impulses folded onto the grid, so two waveforms can be compared."""
        out = self.values.copy()
        for position, weight in self.impulses:
            idx = int(np.argmin(np.abs(grid - position)))
            out[idx] += weight
        return out


# ------------------------------------------------------------ transform pool --

@dataclass(frozen=True)
class Pair:
    """One transform pair x(t) <-> X(jw), in its unshifted, unscaled form."""

    key: str
    latex_time: str
    latex_freq: str
    time: Callable[[np.ndarray], Waveform]
    freq: Callable[[np.ndarray], Waveform]
    real_even: bool = False   # phase is 0 or pi, so the magnitude decides
    real_time: bool = True    # the time signal has no imaginary part


def _continuous(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], Waveform]:
    return lambda grid: Waveform(np.asarray(fn(grid), dtype=complex))


def _impulses(*items: Tuple[float, Complex]) -> Callable[[np.ndarray], Waveform]:
    return lambda grid: Waveform(np.zeros_like(grid, dtype=complex), tuple(items))


def build_pairs(w0: float) -> Dict[str, Pair]:
    """The transform pairs the training draws from, for one choice of w0."""
    pi = math.pi

    def rect(t):
        return np.where(np.abs(t) < 0.5, 1.0, 0.0)

    def tri(t):
        return np.maximum(1.0 - np.abs(t), 0.0)

    def causal_exp(t):
        # clipped so that e^{-t} does not overflow for the suppressed half
        return np.where(t >= 0.0, np.exp(-np.clip(t, 0.0, None)), 0.0)

    def odd_rect(t):
        return np.where((t > 0) & (t < 1), 1.0, 0.0) - np.where((t > -1) & (t < 0), 1.0, 0.0)

    w0_tex = _pi_multiple_tex(w0)

    pairs = [
        Pair("rect", r"\mathrm{rect}(t)", r"\mathrm{si}\!\left(\tfrac{\omega}{2}\right)",
             _continuous(rect), _continuous(lambda w: _si(w / 2)), real_even=True),
        Pair("tri", r"\mathrm{tri}(t)", r"\mathrm{si}^2\!\left(\tfrac{\omega}{2}\right)",
             _continuous(tri), _continuous(lambda w: _si(w / 2) ** 2), real_even=True),
        Pair("si", r"\mathrm{si}(\pi t)", r"\mathrm{rect}\!\left(\tfrac{\omega}{2\pi}\right)",
             _continuous(lambda t: _si(pi * t)),
             _continuous(lambda w: np.where(np.abs(w) < pi, 1.0, 0.0)), real_even=True),
        Pair("si2", r"\mathrm{si}^2(\pi t)", r"\mathrm{tri}\!\left(\tfrac{\omega}{2\pi}\right)",
             _continuous(lambda t: _si(pi * t) ** 2),
             _continuous(lambda w: np.maximum(1.0 - np.abs(w) / (2 * pi), 0.0)), real_even=True),
        Pair("exp_abs", r"e^{-|t|}", r"\frac{2}{1+\omega^{2}}",
             _continuous(lambda t: np.exp(-np.abs(t))),
             _continuous(lambda w: 2.0 / (1.0 + w ** 2)), real_even=True),
        Pair("delta", r"\delta(t)", r"1",
             lambda grid: Waveform(np.zeros_like(grid, dtype=complex), ((0.0, 1.0),)),
             _continuous(lambda w: np.ones_like(w)), real_even=True),
        Pair("exp_causal", r"e^{-t}\,\varepsilon(t)", r"\frac{1}{1+\mathrm{j}\omega}",
             _continuous(causal_exp),
             _continuous(lambda w: 1.0 / (1.0 + 1j * w))),
        Pair("odd_rect", r"\mathrm{rect}(t-\tfrac{1}{2})-\mathrm{rect}(t+\tfrac{1}{2})",
             r"-\mathrm{j}\,\omega\,\mathrm{si}^2\!\left(\tfrac{\omega}{2}\right)",
             _continuous(odd_rect),
             _continuous(lambda w: -1j * w * _si(w / 2) ** 2)),
        Pair("cos", fr"\cos({w0_tex}t)",
             fr"\pi\left[\delta(\omega-{w0_tex})+\delta(\omega+{w0_tex})\right]",
             _continuous(lambda t: np.cos(w0 * t)),
             _impulses((w0, pi), (-w0, pi))),
        Pair("sin", fr"\sin({w0_tex}t)",
             fr"-\mathrm{{j}}\pi\left[\delta(\omega-{w0_tex})-\delta(\omega+{w0_tex})\right]",
             _continuous(lambda t: np.sin(w0 * t)),
             _impulses((w0, -1j * pi), (-w0, 1j * pi))),
        Pair("cexp", fr"e^{{\mathrm{{j}}{w0_tex}t}}", fr"2\pi\,\delta(\omega-{w0_tex})",
             _continuous(lambda t: np.exp(1j * w0 * t)),
             _impulses((w0, 2 * pi)), real_time=False),
        Pair("rect_cos", fr"\mathrm{{rect}}(t)\cos({w0_tex}t)",
             fr"\tfrac{{1}}{{2}}\left[\mathrm{{si}}\!\left(\tfrac{{\omega-{w0_tex}}}{{2}}\right)"
             fr"+\mathrm{{si}}\!\left(\tfrac{{\omega+{w0_tex}}}{{2}}\right)\right]",
             _continuous(lambda t: rect(t) * np.cos(w0 * t)),
             _continuous(lambda w: 0.5 * (_si((w - w0) / 2) + _si((w + w0) / 2))),
             real_even=True),
    ]
    return {p.key: p for p in pairs}


def _pi_multiple_tex(value: float) -> str:
    """Render a multiple of pi as TeX; the old module printed raw floats."""
    ratio = value / math.pi
    for numerator, denominator, tex in [(1, 2, r"\tfrac{\pi}{2}"), (1, 1, r"\pi"),
                                        (3, 2, r"\tfrac{3\pi}{2}"), (2, 1, r"2\pi"),
                                        (3, 1, r"3\pi")]:
        if abs(ratio - numerator / denominator) < 1e-9:
            return tex
    return f"{value:.2f}"


# ------------------------------------------------------- applying properties --

def apply_properties(pair: Pair, t: np.ndarray, w: np.ndarray,
                     *, scale: float, shift: float, width: float,
                     width_factor: bool = True) -> Tuple[Waveform, Waveform]:
    """Build y(t) = scale * x((t - shift)/width) and its spectrum.

    Y(jw) = scale * width * X(width * w) * e^{-j w shift}. ``width_factor`` exists
    so a distractor can drop the amplitude factor that time scaling introduces.
    """
    base_time = pair.time((t - shift) / width)
    time_wave = Waveform(
        base_time.values * scale,
        tuple((shift + position * width, weight * scale * width)
              for position, weight in base_time.impulses),
    )

    amplitude = scale * width if width_factor else scale
    base_freq = pair.freq(w * width)
    freq_wave = Waveform(
        base_freq.values * amplitude * np.exp(-1j * w * shift),
        tuple((position / width,
               weight * scale * np.exp(-1j * (position / width) * shift))
              for position, weight in pair.freq(w).impulses),
    )
    return time_wave, freq_wave


# ------------------------------------------------------------------ drawing --

def _pi_ticks(ax: matplotlib.axes.Axes) -> None:
    ticks = np.array([-2, -1, 0, 1, 2]) * math.pi
    ax.set_xticks(ticks)
    ax.set_xticklabels([r"$-2$", r"$-1$", r"$0$", r"$1$", r"$2$"])
    ax.set_xlim(-W_LIMIT, W_LIMIT)
    ax.set_xlabel(r"$\omega/\pi\ \rightarrow$", fontsize=9)


def _impulse_label(weight: Complex) -> str:
    """Impulse weights are almost always multiples of pi -- say so."""
    magnitude = abs(weight)
    ratio = magnitude / math.pi
    if abs(ratio - round(ratio)) < 1e-6 and round(ratio) >= 1:
        factor = int(round(ratio))
        return r"$\pi$" if factor == 1 else fr"${factor}\pi$"
    if abs(ratio - 0.5) < 1e-6:
        return r"$\pi/2$"
    if abs(magnitude - round(magnitude)) < 1e-6:
        return f"${int(round(magnitude))}$"
    return f"${magnitude:.2f}$"


def _draw_impulse(ax: matplotlib.axes.Axes, position: float, height: float,
                  colour: str, label: str | None) -> None:
    ax.annotate("", xy=(position, height), xytext=(position, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0,
                                shrinkA=0, shrinkB=0, mutation_scale=13))
    if label:
        ax.annotate(label, xy=(position, height), xytext=(3, 1),
                    textcoords="offset points", fontsize=8, color=colour)


def draw_spectrum(ax_mag: matplotlib.axes.Axes, ax_phase: matplotlib.axes.Axes,
                  w: np.ndarray, wave: Waveform, *, colour: str,
                  mag_limit: float, annotate: bool = True) -> None:
    magnitude = np.abs(wave.values)
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak > 1e-9:
        ax_mag.plot(w, magnitude, color=colour, lw=1.9)
        # Where the magnitude vanishes the phase is numerical noise, so leave a
        # gap instead of a line the student would try to read.
        phase = np.angle(wave.values)
        phase[magnitude < PHASE_FLOOR * peak] = np.nan
        ax_phase.plot(w, phase, color=colour, lw=1.9)

    for position, weight in wave.impulses:
        _draw_impulse(ax_mag, position, abs(weight), colour,
                      _impulse_label(weight) if annotate else None)
        ax_phase.plot([position], [np.angle(weight)], marker="o", ms=5, color=colour)

    ax_mag.set_ylim(0.0, mag_limit)
    ax_mag.grid(True, alpha=0.3)
    _pi_ticks(ax_mag)

    ax_phase.set_ylim(-1.25 * math.pi, 1.25 * math.pi)
    ax_phase.set_yticks([-math.pi, 0, math.pi])
    ax_phase.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax_phase.axhline(0.0, color="0.6", lw=0.7)
    ax_phase.grid(True, alpha=0.3)
    _pi_ticks(ax_phase)


def time_window(t: np.ndarray, waves: Sequence[Waveform]) -> Tuple[float, float]:
    """Smallest symmetric window that still holds every waveform.

    A fixed +-6 s axis turns a rect into a sliver in the middle of an empty plot,
    which is most of what made the old figures hard to read.
    """
    reach = 1.5
    for wave in waves:
        magnitude = np.abs(wave.values)
        peak = float(magnitude.max()) if magnitude.size else 0.0
        if peak > 1e-9:
            inside = np.nonzero(magnitude > 0.02 * peak)[0]
            if inside.size:
                reach = max(reach, abs(float(t[inside[0]])), abs(float(t[inside[-1]])))
        for position, _ in wave.impulses:
            reach = max(reach, abs(position))
    reach = float(min(T_LIMIT, math.ceil(reach + 0.8)))
    return -reach, reach


def draw_time(ax: matplotlib.axes.Axes, t: np.ndarray, wave: Waveform, *,
              colour: str, limit: float, window: Tuple[float, float]) -> None:
    # Decided per waveform: a complex distractor next to real ones has to show
    # its imaginary part, whatever the true answer looks like.
    complex_signal = bool(np.any(np.abs(wave.values.imag) > 1e-9))
    ax.plot(t, wave.values.real, color=colour, lw=1.9,
            label=r"$\mathrm{Re}$" if complex_signal else None)
    if complex_signal:
        ax.plot(t, wave.values.imag, color=IMAG_COLOUR, lw=1.4, ls=":",
                label=r"$\mathrm{Im}$")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85, ncol=2)

    for position, weight in wave.impulses:
        _draw_impulse(ax, position, weight.real, colour, _impulse_label(weight))

    low, high = window
    step = 1.0 if high <= 4.0 else 2.0
    ax.set_xlim(low, high)
    ax.set_ylim(-limit, limit)
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.set_xticks(np.arange(low, high + step / 2, step))
    ax.set_xticks(np.arange(low, high + step / 4, step / 2), minor=True)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.18, ls="--")
    ax.set_xlabel(r"$t\ \rightarrow$", fontsize=9)


# ---------------------------------------------------------------- difficulty --

@dataclass(frozen=True)
class Level:
    pool: Tuple[str, ...]
    shifts: Tuple[float, ...]
    scales: Tuple[float, ...]
    widths: Tuple[float, ...]


_REAL_EVEN = ("rect", "tri", "si", "si2", "exp_abs", "delta")
_WITH_PHASE = _REAL_EVEN + ("exp_causal", "cos", "sin")

LEVELS: Dict[str, Level] = {
    # Easy: no shift, so the phase is flat and the magnitude alone decides.
    "EASY": Level(_REAL_EVEN, (0.0,), (1.0, 2.0), (1.0, 2.0)),
    # Medium: shifting adds linear phase, scaling couples width and amplitude.
    "MEDIUM": Level(_WITH_PHASE, (-2.0, -1.0, 1.0, 2.0), (0.5, 1.0, 2.0), (1.0, 2.0)),
    # Hard: odd and complex signals, modulation, and subtler distractors.
    "HARD": Level(_WITH_PHASE + ("odd_rect", "cexp", "rect_cos"),
                  (-2.0, -1.0, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)),
}

W0_CHOICES = (math.pi / 2, math.pi, 1.5 * math.pi)


def _pick_other(options: Sequence[Any], current: Any) -> Any:
    alternatives = [o for o in options if o != current]
    return random.choice(alternatives) if alternatives else current


# --------------------------------------------------------------- distractors --

def _reversed_time(wave: Waveform) -> Waveform:
    return Waveform(wave.values[::-1].copy(),
                    tuple((-p, w) for p, w in wave.impulses))


def _relative_distance(candidate: Waveform, truth: Waveform, grid: np.ndarray) -> float:
    a = candidate.rasterised(grid)
    b = truth.rasterised(grid)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def _build_candidates(pair: Pair, level: Level, pairs: Dict[str, Pair],
                      t: np.ndarray, w: np.ndarray,
                      *, scale: float, shift: float, width: float,
                      direction: str) -> List[Tuple[Waveform, Waveform]]:
    """Every plausible mistake, as a full (time, spectrum) pair.

    Errors that change the magnitude come first so a problem never degenerates
    into "count the phase wraps", which is what the previous version produced
    whenever all four options shared one magnitude.
    """
    by_magnitude: List[Tuple[Waveform, Waveform]] = []
    by_phase: List[Tuple[Waveform, Waveform]] = []

    def variant(**overrides):
        params = dict(scale=scale, shift=shift, width=width)
        params.update(overrides)
        return apply_properties(pair, t, w, **params)

    # -- errors that change the magnitude ----------------------------------
    by_magnitude.append(variant(width=_pick_other(level.widths, width)))
    by_magnitude.append(variant(scale=_pick_other(level.scales, scale)))
    other_key = _pick_other(level.pool, pair.key)
    if other_key != pair.key:
        by_magnitude.append(apply_properties(pairs[other_key], t, w,
                                             scale=scale, shift=shift, width=width))
    if abs(width - 1.0) > 1e-9:
        # forgetting that time scaling also scales the amplitude
        by_magnitude.append(variant(width_factor=False))

    # -- errors that only move the phase -----------------------------------
    true_time, true_freq = apply_properties(pair, t, w, scale=scale, shift=shift, width=width)
    if abs(shift) > 1e-9:
        by_phase.append(variant(shift=0.0))
        by_phase.append(variant(shift=-shift))
        by_phase.append(variant(shift=shift + random.choice([-1.0, 1.0])))
    by_phase.append((true_time.scaled(-1.0), true_freq.scaled(-1.0)))
    if not pair.real_even:
        by_phase.append((true_time.conjugated(), true_freq.conjugated()))
        if direction == "FREQ_TO_TIME":
            by_phase.append((_reversed_time(true_time), true_freq.conjugated()))

    random.shuffle(by_magnitude)
    random.shuffle(by_phase)

    # Interleaved starting with the magnitude group, so a problem never collapses
    # into four identical magnitudes that differ only in phase slope.
    made: List[Tuple[Waveform, Waveform]] = []
    for first, second in zip(by_magnitude, by_phase):
        made.extend((first, second))
    made.extend(by_magnitude[len(by_phase):])
    made.extend(by_phase[len(by_magnitude):])

    # Guaranteed-distinct fallbacks, so three options always exist. The old
    # module filtered candidates without a floor and then indexed four of them.
    for factor in (2.0, 0.5, 3.0, -2.0):
        made.append((true_time.scaled(factor), true_freq.scaled(factor)))
    return made


def _choose_distractors(candidates: Sequence[Tuple[Waveform, Waveform]],
                        truth: Tuple[Waveform, Waveform], grid: np.ndarray,
                        answer_index: int, count: int = 3) -> List[Tuple[Waveform, Waveform]]:
    chosen: List[Tuple[Waveform, Waveform]] = []
    for candidate in candidates:
        if len(chosen) == count:
            break
        shown = candidate[answer_index]
        if _relative_distance(shown, truth[answer_index], grid) < 0.08:
            continue
        if any(_relative_distance(shown, c[answer_index], grid) < 0.08 for c in chosen):
            continue
        chosen.append(candidate)
    return chosen


# ------------------------------------------------------------------- wording --

def _num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def _argument_tex(shift: float, width: float) -> str:
    inner = "t"
    if abs(shift) > 1e-9:
        inner = fr"t{'-' if shift > 0 else '+'}{_num(abs(shift))}"
    if abs(width - 1.0) < 1e-9:
        return inner
    # x(t/0.5) reads badly; write the equivalent x(2t) whenever 1/width is whole.
    inverse = 1.0 / width
    if abs(inverse - round(inverse)) < 1e-9:
        factor = int(round(inverse))
        return fr"{factor}\,t" if inner == "t" else fr"{factor}\,({inner})"
    return fr"\tfrac{{{inner}}}{{{_num(width)}}}"


def _describe(pair: Pair, scale: float, shift: float, width: float) -> Dict[str, str]:
    base = (fr"x(t)={pair.latex_time}\quad\circ\!\!-\!\!\bullet\quad "
            fr"X(\mathrm{{j}}\omega)={pair.latex_freq}")

    factor = "" if abs(scale - 1.0) < 1e-9 else fr"{_num(scale)}\,"
    applied = fr"y(t)={factor}x\!\left({_argument_tex(shift, width)}\right)"

    amplitude = scale * width
    spectrum = fr"Y(\mathrm{{j}}\omega)="
    spectrum += "" if abs(amplitude - 1.0) < 1e-9 else fr"{_num(amplitude)}\,"
    spectrum += fr"X({'' if abs(width - 1.0) < 1e-9 else _num(width)}\omega)"
    if abs(shift) > 1e-9:
        sign = "-" if shift > 0 else "+"
        spectrum += fr"\,e^{{{sign}\mathrm{{j}}\omega\cdot{_num(abs(shift))}}}"

    notes: List[str] = []
    if abs(shift) > 1e-9:
        notes.append(f"Time shift by t0 = {_num(shift)} multiplies the spectrum by "
                     f"e^(-jw*{_num(shift)}): the magnitude is unchanged, the phase gains a "
                     f"slope of {_num(-shift)}.")
    if abs(width - 1.0) > 1e-9:
        notes.append(f"Time scaling by {_num(width)} compresses the spectrum by the same "
                     f"factor and multiplies its amplitude by {_num(width)}.")
    if pair.key in ("cos", "sin", "cexp"):
        notes.append("A harmonic signal has a line spectrum: Dirac impulses, not a curve.")
    if pair.key == "rect_cos":
        notes.append("Multiplying by cos(w0*t) shifts the spectrum to +-w0 (modulation).")
    if pair.key == "odd_rect":
        notes.append("The signal is real and odd, so its spectrum is purely imaginary.")
    if not notes:
        notes.append("A basic transform pair, used without any further property.")

    return {"latex_time": base, "latex_freq": f"{applied}\\qquad {spectrum}",
            "property_msg": " ".join(notes)}


# -------------------------------------------------------------------- figure --

def _render(direction: str, t: np.ndarray, w: np.ndarray,
            given: Tuple[Waveform, Waveform],
            options: Sequence[Tuple[Waveform, Waveform]],
            ) -> Tuple[str, List[Tuple[float, float, float, float]]]:
    answer_index = 1 if direction == "TIME_TO_FREQ" else 0

    # One set of limits for every option, so the axes cannot give the answer away.
    shown = [opt[answer_index] for opt in options]
    if direction == "TIME_TO_FREQ":
        mag_limit = 1.15 * max([wave.peak() for wave in shown] + [1e-3])
        time_limit = 1.2 * max(given[0].peak(), 1e-3)
    else:
        mag_limit = 1.15 * max(given[1].peak(), 1e-3)
        time_limit = 1.2 * max([wave.peak() for wave in shown] + [1e-3])

    # A constrained layout plus the canvas.draw() needed to measure it rendered
    # the whole figure twice and cost ~600 ms per problem; a fixed gridspec gives
    # the same positions from get_position() without drawing at all.
    fig = plt.figure(figsize=(10.0, 9.6))
    gs = fig.add_gridspec(nrows=5, ncols=2, height_ratios=[1.25, 1, 1, 1, 1],
                          hspace=0.62, wspace=0.20,
                          left=0.075, right=0.985, top=0.945, bottom=0.05)
    hit_axes: List[List[matplotlib.axes.Axes]] = []

    if direction == "TIME_TO_FREQ":
        ax_given = fig.add_subplot(gs[0, :])
        draw_time(ax_given, t, given[0], colour=GIVEN_COLOUR, limit=time_limit,
                  window=time_window(t, [given[0]]))
        ax_given.set_title(r"given signal $y(t)$ — which spectrum belongs to it?",
                           fontsize=12, fontweight="bold")
        for row, option in enumerate(options):
            ax_mag = fig.add_subplot(gs[1 + row, 0])
            ax_phase = fig.add_subplot(gs[1 + row, 1])
            draw_spectrum(ax_mag, ax_phase, w, option[1], colour=ANSWER_COLOUR,
                          mag_limit=mag_limit)
            ax_mag.set_title(fr"$\mathcal{{O}}_{row + 1}$:  $|Y(\mathrm{{j}}\omega)|$", fontsize=10)
            ax_phase.set_title(r"$\varphi(\omega)$", fontsize=10)
            hit_axes.append([ax_mag, ax_phase])
    else:
        ax_mag_g = fig.add_subplot(gs[0, 0])
        ax_phase_g = fig.add_subplot(gs[0, 1])
        draw_spectrum(ax_mag_g, ax_phase_g, w, given[1], colour=GIVEN_COLOUR,
                      mag_limit=mag_limit)
        ax_mag_g.set_title(r"given $|Y(\mathrm{j}\omega)|$", fontsize=12, fontweight="bold")
        ax_phase_g.set_title(r"given $\varphi(\omega)$ — which signal belongs to it?",
                             fontsize=12, fontweight="bold")
        # One window for all four, otherwise the axis alone would give away which
        # option is shifted.
        window = time_window(t, [opt[0] for opt in options])
        for row, option in enumerate(options):
            ax_time = fig.add_subplot(gs[1 + row, :])
            draw_time(ax_time, t, option[0], colour=ANSWER_COLOUR, limit=time_limit,
                      window=window)
            ax_time.set_title(fr"$\mathcal{{O}}_{row + 1}$:  $y(t)$", fontsize=10)
            hit_axes.append([ax_time])

    hit_boxes = []
    for axes in hit_axes:
        boxes = [ax.get_position().bounds for ax in axes]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        pad_y = 0.012
        hit_boxes.append((x0, max(0.0, y0 - pad_y), x1 - x0,
                          min(1.0, y1 - y0 + 2 * pad_y)))

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode(), hit_boxes


# ---------------------------------------------------------------- generation --

def _impulses_visible(wave: Waveform, limit: float) -> bool:
    return all(abs(position) <= 0.92 * limit for position, _ in wave.impulses)


def create_fourier_problem(difficulty: str, direction: str) -> Dict[str, Any]:
    level = LEVELS.get(difficulty.upper(), LEVELS["EASY"])
    direction = direction.upper()
    if direction not in ("TIME_TO_FREQ", "FREQ_TO_TIME"):
        direction = "TIME_TO_FREQ"

    t = np.linspace(-T_LIMIT, T_LIMIT, N_SAMPLES)
    w = np.linspace(-W_LIMIT, W_LIMIT, N_SAMPLES)
    answer_index = 1 if direction == "TIME_TO_FREQ" else 0

    for _ in range(40):
        w0 = random.choice(W0_CHOICES)
        pairs = build_pairs(w0)
        pair = pairs[random.choice(level.pool)]
        scale = random.choice(level.scales)
        shift = random.choice(level.shifts)
        width = random.choice(level.widths)

        truth = apply_properties(pair, t, w, scale=scale, shift=shift, width=width)
        # Impulses pushed outside the axis by scaling would be invisible, and an
        # answer nobody can see is not an answer.
        if not (_impulses_visible(truth[1], W_LIMIT) and _impulses_visible(truth[0], T_LIMIT)):
            continue

        candidates = _build_candidates(pair, level, pairs, t, w,
                                       scale=scale, shift=shift, width=width,
                                       direction=direction)
        candidates = [c for c in candidates
                      if _impulses_visible(c[1], W_LIMIT) and _impulses_visible(c[0], T_LIMIT)]
        distractors = _choose_distractors(candidates, truth, w if answer_index else t,
                                          answer_index)
        if len(distractors) < 3:
            continue

        options = [truth] + distractors
        order = list(range(4))
        random.shuffle(order)
        shuffled = [options[i] for i in order]

        plot_data, hit_boxes = _render(direction, t, w, truth, shuffled)
        payload = {"plot_data": plot_data,
                   "correctIndex": order.index(0),
                   "hit_boxes": hit_boxes}
        payload.update(_describe(pair, scale, shift, width))
        return payload

    return {"error": "Could not build a problem with four distinct options."}


# ---------------------------------------------------------------- blueprint --

training_fourier_bp = Blueprint("training_fourier", __name__)


@training_fourier_bp.route("/")
def training_fourier() -> str:
    return render_template("training_fourier.html")


@training_fourier_bp.route("/generate", methods=["POST"])
def generate_problem() -> Any:
    data = request.get_json(force=True)
    difficulty = str(data.get("difficulty", "EASY")).upper()
    direction = str(data.get("direction", "TIME_TO_FREQ")).upper()
    result = create_fourier_problem(difficulty, direction)
    return jsonify(result), (400 if "error" in result else 200)


@training_fourier_bp.route("/check_answer", methods=["POST"])
def check_answer() -> Any:
    data = request.get_json(force=True)
    correct = data.get("selectedIndex") == data.get("correctIndex")
    return jsonify({"feedback": "Correct!" if correct else "Incorrect. Try again!"})
