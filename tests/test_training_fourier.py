"""Tests for the Fourier training generator.

Two things have to hold. The transform pairs must be right -- a training module
that teaches the wrong correspondence is worse than none -- so every continuous
pair is checked against a numerical Fourier integral. And every generated problem
must be answerable: four options, exactly one correct, all visibly different. The
previous version raised IndexError on roughly one problem in twenty-five because
its distractor filter could return fewer than three candidates.
"""

import math

import numpy as np
import pytest

from pages.training import training_fourier as tf

W0 = math.pi
PAIRS = tf.build_pairs(W0)

# Pairs with a closed-form continuous spectrum, checkable against an FFT.
CONTINUOUS = ["rect", "tri", "si", "si2", "exp_abs", "exp_causal", "odd_rect", "rect_cos"]
# Pairs whose spectrum is a line spectrum instead.
IMPULSIVE = ["delta", "cos", "sin", "cexp"]


def numeric_spectrum(pair: tf.Pair, omega: np.ndarray) -> np.ndarray:
    """X(jw) from the definition, on a window wide enough for slow decay."""
    span, count = 400.0, 2 ** 20
    t = np.linspace(-span, span, count, endpoint=False)
    dt = t[1] - t[0]
    x = pair.time(t).values
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) * dt
    grid = np.fft.fftshift(np.fft.fftfreq(count, dt)) * 2 * math.pi
    return (np.interp(omega, grid, spectrum.real)
            + 1j * np.interp(omega, grid, spectrum.imag))


# --- are the transform pairs correct? ---------------------------------------

@pytest.mark.parametrize("key", CONTINUOUS)
def test_spectrum_matches_the_fourier_integral(key):
    pair = PAIRS[key]
    omega = np.linspace(-2 * math.pi, 2 * math.pi, 121)
    analytic = pair.freq(omega).values
    numeric = numeric_spectrum(pair, omega)

    # si and si2 transform to a rect and a triangle; the FFT of a truncated
    # signal rings around those corners, so the edges are left out.
    if key in ("si", "si2"):
        edge = math.pi if key == "si" else 2 * math.pi
        keep = np.abs(np.abs(omega) - edge) > 0.25
        analytic, numeric = analytic[keep], numeric[keep]

    np.testing.assert_allclose(analytic, numeric, atol=0.03,
                               err_msg=f"{key}: analytic spectrum and Fourier integral disagree")


def test_dc_value_is_the_area_under_the_signal():
    """X(0) = integral x(t) dt, the cheapest sanity check there is."""
    for key, area in [("rect", 1.0), ("tri", 1.0), ("si", 1.0), ("si2", 1.0),
                      ("exp_abs", 2.0), ("exp_causal", 1.0), ("odd_rect", 0.0)]:
        value = PAIRS[key].freq(np.array([0.0])).values[0]
        assert value == pytest.approx(area, abs=1e-9), key


def test_si_uses_the_toolkit_convention():
    """si(t) = sin(t)/t, so si(pi*t) is an ideal lowpass cutting off at |w| = pi.

    The module used to plot np.sinc under the name sinc, which put the band edge
    at pi**2 instead.
    """
    band = PAIRS["si"].freq(np.array([0.0, 2.0, 3.0, 3.3, 5.0])).values.real
    assert band[0] == pytest.approx(1.0)
    assert band[1] == pytest.approx(1.0)      # inside |w| < pi
    assert band[2] == pytest.approx(1.0)      # still inside
    assert band[3] == pytest.approx(0.0)      # just outside
    assert band[4] == pytest.approx(0.0)


@pytest.mark.parametrize("key,positions,weights", [
    ("cos", (math.pi, -math.pi), (math.pi, math.pi)),
    ("sin", (math.pi, -math.pi), (-1j * math.pi, 1j * math.pi)),
    ("cexp", (math.pi,), (2 * math.pi,)),
])
def test_line_spectra(key, positions, weights):
    impulses = PAIRS[key].freq(np.linspace(-8, 8, 101)).impulses
    assert tuple(p for p, _ in impulses) == pytest.approx(positions)
    assert tuple(w for _, w in impulses) == pytest.approx(weights)


def test_real_even_signals_have_a_real_spectrum():
    omega = np.linspace(-2 * math.pi, 2 * math.pi, 201)
    for key in ("rect", "tri", "si", "si2", "exp_abs"):
        assert np.max(np.abs(PAIRS[key].freq(omega).values.imag)) < 1e-9, key


def test_real_odd_signal_has_an_imaginary_spectrum():
    omega = np.linspace(-2 * math.pi, 2 * math.pi, 201)
    values = PAIRS["odd_rect"].freq(omega).values
    assert np.max(np.abs(values.real)) < 1e-9
    assert np.max(np.abs(values.imag)) > 0.5


# --- do the properties behave like the theorems? ----------------------------

T_GRID = np.linspace(-tf.T_LIMIT, tf.T_LIMIT, tf.N_SAMPLES)
W_GRID = np.linspace(-tf.W_LIMIT, tf.W_LIMIT, tf.N_SAMPLES)


def test_time_shift_leaves_the_magnitude_and_adds_linear_phase():
    pair = PAIRS["rect"]
    _, plain = tf.apply_properties(pair, T_GRID, W_GRID, scale=1.0, shift=0.0, width=1.0)
    _, shifted = tf.apply_properties(pair, T_GRID, W_GRID, scale=1.0, shift=2.0, width=1.0)

    np.testing.assert_allclose(np.abs(shifted.values), np.abs(plain.values), atol=1e-12)
    expected = plain.values * np.exp(-1j * W_GRID * 2.0)
    np.testing.assert_allclose(shifted.values, expected, atol=1e-12)


def test_time_scaling_stretches_the_spectrum_and_the_amplitude():
    pair = PAIRS["tri"]
    _, plain = tf.apply_properties(pair, T_GRID, W_GRID, scale=1.0, shift=0.0, width=1.0)
    _, wide = tf.apply_properties(pair, T_GRID, W_GRID, scale=1.0, shift=0.0, width=2.0)

    # x(t/2) <-> 2 X(2w)
    expected = 2.0 * np.interp(2.0 * W_GRID, W_GRID, plain.values.real)
    inside = np.abs(2.0 * W_GRID) <= tf.W_LIMIT
    np.testing.assert_allclose(wide.values.real[inside], expected[inside], atol=1e-3)


def test_impulse_positions_follow_the_scaling():
    """cos(w0 t/width) has its lines at w0/width, with the weight unchanged."""
    _, spectrum = tf.apply_properties(PAIRS["cos"], T_GRID, W_GRID,
                                      scale=1.0, shift=0.0, width=2.0)
    positions = sorted(p for p, _ in spectrum.impulses)
    assert positions == pytest.approx([-W0 / 2, W0 / 2])
    assert all(abs(w) == pytest.approx(math.pi) for _, w in spectrum.impulses)


def test_shifting_a_delta_moves_it_in_time():
    time, _ = tf.apply_properties(PAIRS["delta"], T_GRID, W_GRID,
                                  scale=2.0, shift=1.0, width=1.0)
    assert [p for p, _ in time.impulses] == pytest.approx([1.0])
    assert [abs(w) for _, w in time.impulses] == pytest.approx([2.0])


# --- is every generated problem answerable? ---------------------------------

DIFFICULTIES = ["EASY", "MEDIUM", "HARD"]
DIRECTIONS = ["TIME_TO_FREQ", "FREQ_TO_TIME"]


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_problem_payload(difficulty, direction):
    problem = tf.create_fourier_problem(difficulty, direction)
    assert "error" not in problem, problem.get("error")
    assert 0 <= problem["correctIndex"] <= 3
    assert len(problem["hit_boxes"]) == 4
    assert problem["plot_data"]
    for key in ("latex_time", "latex_freq", "property_msg"):
        assert problem[key].strip()
    for x0, y0, width, height in problem["hit_boxes"]:
        assert 0.0 <= x0 < 1.0 and 0.0 <= y0 < 1.0
        assert 0.0 < width <= 1.0 and 0.0 < height <= 1.0


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_options_are_always_four_and_distinguishable(difficulty, direction):
    """The generator used to hand out fewer than four options and then crash."""
    level = tf.LEVELS[difficulty]
    answer_index = 1 if direction == "TIME_TO_FREQ" else 0
    grid = W_GRID if answer_index else T_GRID

    for _ in range(120):
        pairs = tf.build_pairs(tf.random.choice(tf.W0_CHOICES))
        pair = pairs[tf.random.choice(level.pool)]
        scale = tf.random.choice(level.scales)
        shift = tf.random.choice(level.shifts)
        width = tf.random.choice(level.widths)

        truth = tf.apply_properties(pair, T_GRID, W_GRID,
                                    scale=scale, shift=shift, width=width)
        candidates = tf._build_candidates(pair, level, pairs, T_GRID, W_GRID,
                                          scale=scale, shift=shift, width=width,
                                          direction=direction)
        chosen = tf._choose_distractors(candidates, truth, grid, answer_index)
        assert len(chosen) == 3, f"{pair.key}: only {len(chosen)} distractors"

        shown = [truth[answer_index]] + [c[answer_index] for c in chosen]
        for i in range(len(shown)):
            for j in range(i + 1, len(shown)):
                distance = tf._relative_distance(shown[i], shown[j], grid)
                assert distance >= 0.08, f"{pair.key}: options {i} and {j} look alike"


def test_easy_has_no_time_shift():
    """Easy is meant to be decidable from the magnitude alone."""
    assert tf.LEVELS["EASY"].shifts == (0.0,)
    assert all(key in tf.LEVELS["EASY"].pool for key in ("rect", "tri", "si"))


def test_impulses_stay_inside_the_plotted_range():
    for difficulty in DIFFICULTIES:
        for _ in range(60):
            level = tf.LEVELS[difficulty]
            pairs = tf.build_pairs(tf.random.choice(tf.W0_CHOICES))
            pair = pairs[tf.random.choice(level.pool)]
            time, spectrum = tf.apply_properties(
                pair, T_GRID, W_GRID,
                scale=tf.random.choice(level.scales),
                shift=tf.random.choice(level.shifts),
                width=tf.random.choice(level.widths))
            if tf._impulses_visible(spectrum, tf.W_LIMIT):
                assert all(abs(p) <= tf.W_LIMIT for p, _ in spectrum.impulses)
            if tf._impulses_visible(time, tf.T_LIMIT):
                assert all(abs(p) <= tf.T_LIMIT for p, _ in time.impulses)
