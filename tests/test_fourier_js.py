"""Regression tests for the browser-side Fourier transform.

All of the module's maths lives in ``static/js/fourier_compute.js``; the Flask
route only renders the page. The values below are known correspondences, so a
change in windowing, thresholds or the exp_iwt handling shows up here instead of
in a lecture.

``tests/fourier_probe.js`` runs the module in a Node sandbox and reports the
numbers; the assertions live here.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).resolve().parent / "fourier_probe.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@pytest.fixture(scope="module")
def report():
    proc = subprocess.run(
        ["node", str(PROBE)], capture_output=True, text=True, timeout=120, cwd=REPO_ROOT
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# --- the transform ----------------------------------------------------------

def test_dc_value_is_the_area_under_the_signal(report):
    """X(0) = integral of x(t); normalising it away used to hide this."""
    assert report["rect"]["areaAtDc"] == pytest.approx(1.0, abs=0.01)
    assert report["tri"]["areaAtDc"] == pytest.approx(1.0, abs=0.01)
    assert report["scaledTri"]["areaAtDc"] == pytest.approx(4.0, abs=0.02)


def test_rect_transform_vanishes_at_first_zero(report):
    assert report["rect"]["atFirstZero"] == pytest.approx(0.0, abs=0.01)


def test_delta_has_a_flat_spectrum(report):
    delta = report["delta"]
    assert delta["atDc"] == pytest.approx(1.0, abs=0.01)
    assert delta["atW3"] == pytest.approx(1.0, abs=0.02)
    assert delta["atW10"] == pytest.approx(1.0, abs=0.05)


def test_si_is_sin_t_over_t(report):
    """si(pi*t) is an ideal lowpass cutting off at |w| = pi.

    A copy of si() that read sin(pi*t)/(pi*t) stretched the passband out to
    pi**2 without anything failing, which is what this pins down.
    """
    si = report["si"]
    assert si["inBand"] == pytest.approx(1.0, abs=0.1)
    assert si["atCutoff"] == pytest.approx(1.0, abs=0.1)
    assert si["aboveCutoff"] < 0.1
    assert si["farAbove"] < 0.05


def test_time_shift_gives_linear_phase(report):
    shift = report["shiftByExpression"]
    assert shift["phase"] == pytest.approx(-2 * shift["omega"], abs=0.01)


# --- Fourier properties -----------------------------------------------------

def test_properties_match_the_written_out_expression(report):
    """The sliders re-evaluate the signal, so they must be exact, not approximate."""
    props = report["properties"]
    assert props["shift"] == pytest.approx(0.0, abs=1e-12)
    assert props["scale"] == pytest.approx(0.0, abs=1e-12)
    assert props["modulate"] == pytest.approx(0.0, abs=1e-12)
    assert props["combined"] == pytest.approx(0.0, abs=1e-12)


def test_shift_property_reproduces_the_shift_theorem(report):
    props = report["properties"]
    expected = report["shiftByExpression"]["phase"]
    assert props["shiftPhase"] == pytest.approx(expected, abs=1e-9)


def test_scaling_property_halves_the_area(report):
    assert report["properties"]["scaleDcValue"] == pytest.approx(0.5, abs=0.02)


def test_broken_property_values_fall_back_to_neutral(report):
    neutral = report["rect"]["areaAtDc"]
    for value in report["propertyFallbacks"]:
        assert value == pytest.approx(neutral, abs=1e-12)


# --- exp_iwt ----------------------------------------------------------------

def test_real_summand_does_not_leak_into_the_imaginary_part(report):
    """rect(t)+exp_iwt(t) must give Im{x} = sin(t), not rect(t)+sin(t)."""
    exp = report["expIwt"]
    assert exp["im"] == pytest.approx(exp["expectedIm"], abs=0.01)


def test_unsupported_exp_iwt_forms_are_refused(report):
    guards = report["expIwtGuards"]
    accepted = ["exp_iwt(t)", "exp_iwt(t,2)", "rect(t)*exp_iwt(t)", "exp_iwt(t)+exp_iwt(t,-1)",
                "2*exp_iwt(t)", "exp_iwt(t)/2", "rect(t)+exp_iwt(t)"]
    rejected = ["exp_iwt(t)**2", "exp_iwt(t)*exp_iwt(t)", "1/exp_iwt(t)", "exp_iwt(2*(t+1))"]
    assert [guards[e] for e in accepted] == ["accepted"] * len(accepted)
    assert [guards[e] for e in rejected] == ["rejected"] * len(rejected)


# --- symmetry, windowing, errors -------------------------------------------

def test_symmetry_properties(report):
    """Real and even transforms to purely real, real and odd to purely imaginary."""
    sym = report["symmetry"]
    assert sym["evenMaxImag"] == pytest.approx(0.0, abs=1e-9)
    assert sym["oddMaxReal"] == pytest.approx(0.0, abs=1e-9)
    assert sym["oddMaxImag"] > 0.1


def test_unbounded_signals_keep_the_origin_in_view(report):
    """step(t) used to recentre on the scan centroid and hide its own edge."""
    assert report["windows"]["step"] == pytest.approx([-20.0, 20.0], abs=0.5)


def test_distant_signals_are_still_recentred(report):
    assert report["windows"]["shiftedFar"] == pytest.approx([30.0, 70.0], abs=0.5)


def test_truncation_is_reported(report):
    trunc = report["truncation"]
    assert trunc["sine"] is True
    assert trunc["step"] is True
    assert trunc["decaying"] is False
    assert trunc["rect"] is False


def test_bad_input_reports_an_error_instead_of_empty_plots(report):
    errors = report["errors"]
    assert errors["nonFinite"] and "not finite" in errors["nonFinite"]
    assert errors["syntax"]
    assert errors["unknownName"] and "not defined" in errors["unknownName"]


def test_phase_is_a_gap_below_the_noise_floor(report):
    """null draws a gap; 0 would read as 'the phase is zero here'."""
    gaps = report["phaseGaps"]
    assert gaps["isPlainArray"] is True
    assert gaps["nullCount"] > 0
    assert gaps["numericCount"] > 0


def test_every_quick_button_expression_is_finite(report):
    assert set(report["quickButtons"].values()) == {"finite"}


def test_window_is_the_expected_forty_seconds(report):
    rect = report["rect"]
    assert rect["windowEnd"] - rect["windowStart"] == pytest.approx(40.0, abs=1e-6)
    assert math.isclose(rect["windowStart"], -20.0, abs_tol=1e-9)
