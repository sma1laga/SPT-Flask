"""Bode plotter: response computation, asymptotic approximations, parsing and routes."""

import csv
import io
import json
import re

import numpy as np
import pytest

from main import create_app
from pages import bode_plot as bp

app = create_app()
app.config["TESTING"] = True


@pytest.fixture(scope="module")
def client():
    with app.test_client() as c:
        yield c


def _response(num, den, rng=None):
    return bp.analyse_transfer_function(num, den, rng).response


def _fine_reference_phase(num, den, w):
    """Independent continuous phase: unwrap on a very dense grid, then interpolate."""
    wf = np.logspace(np.log10(w[0]), np.log10(w[-1]), 200000)
    h = np.polyval(num, 1j * wf) / np.polyval(den, 1j * wf)
    ph = np.degrees(np.unwrap(np.angle(h)))
    return np.interp(np.log10(w), np.log10(wf), ph)


# --- straight-line phase: start value and direction ---------------------------------

@pytest.mark.parametrize("num, den, start, end", [
    ("(s+2)", "(s+10)(s+0.1)", 0.0, -90.0),       # minimum phase: textbook 0 -> -90
    ("(s-1)", "(s+1)", 180.0, 0.0),               # RHP zero: starts at 180, all-pass
    ("[-1]", "(s+1)", 180.0, 90.0),               # negative gain
    ("[-1]", "[1, 2, 1]", 180.0, 0.0),            # negative gain, two poles
    ("1", "(s-1)", -180.0, -90.0),                # RHP pole
    ("1", "s^2", -180.0, -180.0),                 # double integrator
    ("1", "s^2(s+1)", -180.0, -270.0),            # type-2 system
    ("-s", "s+1", -90.0, -180.0),                 # differentiator with negative gain
    ("(s^2-2s+5)", "(s+1)^2", 0.0, -360.0),       # RHP complex pair: 0 -> -180 from the zeros
    ("1", "(s^2+1)(s+100)", 0.0, -270.0),         # undamped pair plus lag
])
def test_straight_line_phase_starts_at_low_frequency_phase(num, den, start, end):
    r = _response(num, den)
    straight = r["phase_straight_deg"]
    assert straight[0] == pytest.approx(start, abs=1e-6)
    assert straight[-1] == pytest.approx(end, abs=1e-6)


def test_straight_line_phase_ramps_45_deg_per_decade():
    r = _response("1", "(s+1)")
    w = r["omega"]
    straight = np.asarray(r["phase_straight_deg"])
    for target, expected in ((0.1, 0.0), (1.0, -45.0), (10.0, -90.0)):
        assert np.interp(np.log10(target), np.log10(w), straight) == pytest.approx(expected, abs=0.5)


def test_single_complex_root_starts_at_its_own_dc_phase():
    r = _response("[1, 1+2j]", "[1, 3]")
    # zero at s = -1-2j: phase of (jw - z) at w -> 0 is atan2(2, 1) = 63.43 deg
    assert r["phase_straight_deg"][0] == pytest.approx(63.43, abs=0.01)
    assert r["phase_deg"][0] == pytest.approx(63.43, abs=0.5)


# --- exact phase: branch consistency and robustness ---------------------------------

@pytest.mark.parametrize("num, den", [
    ([1, 2], np.polymul([1, 10], [1, 0.1]).tolist()),
    ([1, -1], [1, 1]),
    ([-1], [1, 1]),
    ([-1, -1], [1]),
    ([1], [1, -1]),
    ([1], [1, 0, 0]),
    ([1], [1, 1, 0, 0]),
    ([-1, 0], [1, 1]),
    ([1], np.polymul([1, 0, 1], [1, 100]).tolist()),
    ([1], [1, 2e-4, 1]),
    ([1, 1 + 2j], [1, 3]),
    ([100], np.polymul([1, 0], np.polymul([1, 1], [1, 10])).tolist()),
    ([1], np.poly([-1, -1, -1, -1]).tolist()),
    ([1, -2, 5], [1, 2, 1]),
    ([1, 2, 1], [1, 10]),
])
def test_exact_phase_matches_dense_reference_and_straight_line_branch(num, den):
    w = bp._make_freq_vector(num, den)
    r = bp._frequency_response(num, den, w)
    exact = np.asarray(r["phase_deg"])
    straight = np.asarray(r["phase_straight_deg"])
    reference = _fine_reference_phase(num, den, w)
    diff = (exact - reference + 180.0) % 360.0 - 180.0
    # the dense reference is interpolated, which costs a little accuracy near sharp resonances
    assert np.max(np.abs(diff)) < 1e-3
    # both curves start on the same branch and end on the same branch
    assert abs(exact[0] - straight[0]) < 20.0
    assert abs(exact[-1] - straight[-1]) < 20.0


def test_exact_phase_of_lightly_damped_pair_does_not_flip_branch():
    r = _response("1", "[1, 2e-5, 1]")
    assert r["phase_deg"][-1] == pytest.approx(-180.0, abs=0.1)


def test_undamped_pair_with_lag_ends_at_minus_270():
    r = _response("1", "(s^2+1)(s+100)")
    assert r["phase_deg"][-1] == pytest.approx(-270.0, abs=1.0)


def test_low_frequency_phase_is_reported_on_principal_branch():
    assert _response("(s-1)", "(s+1)")["phase_dc_deg"] == 180.0
    assert _response("1", "s^2")["phase_dc_deg"] == -180.0
    assert _response("1", "s")["phase_dc_deg"] == -90.0


# --- straight-line magnitude --------------------------------------------------------

def test_straight_line_magnitude_is_independent_of_plotted_range():
    num, den = [1], np.polymul([1, 1], [1, 100]).tolist()
    r = _response("1", "(s+1)(s+100)", rng=("10", "10000"))
    w = r["omega"]
    straight = np.asarray(r["magnitude_straight_db"])
    textbook = -40.0 - 20.0 * np.log10(np.maximum(w, 1.0)) - 20.0 * np.log10(np.maximum(w, 100.0) / 100.0)
    assert np.max(np.abs(straight - textbook)) < 1e-9


def test_straight_line_magnitude_matches_textbook_for_default_example():
    r = _response("(s+2)", "(s+10)(s+0.1)")
    w = r["omega"]
    straight = np.asarray(r["magnitude_straight_db"])
    dc = 20 * np.log10(2.0)
    textbook = (dc + 20 * np.log10(np.maximum(w, 2.0) / 2.0)
                - 20 * np.log10(np.maximum(w, 0.1) / 0.1) - 20 * np.log10(np.maximum(w, 10.0) / 10.0))
    assert np.max(np.abs(straight - textbook)) < 1e-9


def test_integrator_magnitude_slope():
    r = _response("1", "s")
    w = r["omega"]
    straight = np.asarray(r["magnitude_straight_db"])
    assert np.allclose(straight, -20 * np.log10(w))


# --- margins, crossovers, bandwidth ---------------------------------------------------

def test_crossover_frequencies_have_the_right_meaning():
    a = bp.analyse_transfer_function("100", "s(s+1)(s+10)")
    p = a.bode_payload()
    w = np.asarray(p["omega"])
    phase = np.asarray(p["phase_deg"])
    mag = np.asarray(p["magnitude_db"])
    phase_at_pc = np.interp(np.log10(p["phase_crossover_freq"]), np.log10(w), phase)
    mag_at_gc = np.interp(np.log10(p["gain_crossover_freq"]), np.log10(w), mag)
    assert phase_at_pc == pytest.approx(-180.0, abs=0.5)
    assert mag_at_gc == pytest.approx(0.0, abs=0.1)
    assert p["gain_margin_db"] == pytest.approx(20 * np.log10(1.1), abs=1e-6)
    assert p["phase_crossover_level_deg"] == -180.0


def test_infinite_margins_are_flagged():
    p = bp.analyse_transfer_function("[1]", "[1, 1]").bode_payload()
    assert p["margins_available"] is True
    assert p["gain_margin_infinite"] is True and p["gain_margin_db"] is None
    assert p["phase_margin_infinite"] is True and p["phase_margin_deg"] is None


def test_bandwidth_uses_one_over_sqrt2_and_handles_negative_gain():
    assert bp.analyse_transfer_function("[1]", "[1, 1]").bandwidth["value"] == pytest.approx(1.0, abs=1e-6)
    assert bp.analyse_transfer_function("[-1]", "[1, 2, 1]").bandwidth["value"] == pytest.approx(0.6436, abs=1e-3)
    assert bp.analyse_transfer_function("[1]", "[1, 0]").bandwidth["status"] == "undefined"


def test_frequency_range_includes_far_away_crossover():
    r = _response("[1e9]", "(s+1)^3")
    # gain crossover at ~1000 rad/s must be inside the range with a decade of padding
    assert r["omega"][-1] >= 9.9e3


# --- parsing ---------------------------------------------------------------------------

def test_factorized_repeated_roots_are_exact():
    parsed = bp.parse_polynomial("(s+1)^3")
    assert np.allclose(parsed.roots, -1.0)
    assert bp._corner_frequencies(parsed.roots, []) == [1.0]
    assert parsed.latex == r"\left(s + 1\right)^{3}"


def test_coefficient_list_repeated_roots_are_exact():
    parsed = bp.parse_polynomial("[1, 3, 3, 1]")
    assert np.allclose(parsed.roots, -1.0)


@pytest.mark.parametrize("text, coeffs", [
    ("2s", [2.0, 0.0]),
    ("s^2+2s+1", [1.0, 2.0, 1.0]),
    ("1e5", [100000.0]),
    ("1e-3(s+1)", [1e-3, 1e-3]),
    ("(s+1)(s+2)s", [1.0, 3.0, 2.0, 0.0]),
    ("S+1", [1.0, 1.0]),
    ("[0, 1, 2]", [1.0, 2.0]),
    ("2j", [2j]),
])
def test_parser_accepts_common_notation(text, coeffs):
    assert bp.parse_polynomial(text).coeffs == coeffs


@pytest.mark.parametrize("text", [
    '__import__("os").getpid()',
    'len("ab")',
    's.real',
    'summation(s,(s,0,10))',
    '2^2^2^2',
    's^1001',
    '(s+1)^200',
    'exp(s)',
    '1/(s+1)',
    'x+1',
    '[0]',
    '[]',
    '0',
    '',
    '[1, 1e400]',
    'oo',
    'nan',
])
def test_parser_rejects_unsafe_or_invalid_input(text):
    with pytest.raises(ValueError):
        bp.parse_polynomial(text)


def test_format_polynomial_uses_latex_conventions():
    assert bp.format_polynomial([1] + [0] * 10) == "s^{10}"
    assert bp.format_polynomial([1, 2]) == "s + 2"
    assert bp.format_polynomial([1, -1]) == "s - 1"
    assert bp.format_polynomial([1, 1e-5]) == "s + 10^{-5}"
    assert bp.format_polynomial([1e-20, 1]) == "10^{-20}s + 1"


def test_format_helpers():
    assert bp._format_real_latex(9999.9) == "10^{4}"
    assert bp.format_complex(-1e-7 + 0j) == r"\(-10^{-7}\)"
    assert bp.format_complex(-1 + 2j) == r"\(-1 + 2\mathrm{j}\)"


# --- routes ---------------------------------------------------------------------------

def _bode_data(html):
    match = re.search(r"window\.bodeData = (.*?);\n", html)
    return json.loads(match.group(1))


def test_post_renders_and_embeds_consistent_payload(client):
    resp = client.post("/bode_plot/", data={"numerator": "(s-1)", "denominator": "(s+1)", "submit_action": "bode"})
    assert resp.status_code == 200
    data = _bode_data(resp.get_data(as_text=True))
    assert data["phase_straight_deg"][0] == 180.0
    assert data["phase_deg"][0] == pytest.approx(180.0, abs=2.0)
    assert "Non-minimum-phase" in resp.get_data(as_text=True)


@pytest.mark.parametrize("den", ["[0]", "[]", "0"])
def test_zero_denominator_gives_error_message_not_500(client, den):
    resp = client.post("/bode_plot/", data={"numerator": "[1]", "denominator": den, "submit_action": "bode"})
    assert resp.status_code == 200
    assert "Error parsing denominator" in resp.get_data(as_text=True)


def test_unsafe_expression_is_rejected_by_route(client):
    resp = client.post("/bode_plot/", data={"numerator": '__import__("os").getpid()', "denominator": "[1, 1]",
                                            "submit_action": "bode"})
    assert resp.status_code == 200
    assert "Error parsing numerator" in resp.get_data(as_text=True)


def test_biproper_system_has_no_improper_warning(client):
    resp = client.post("/bode_plot/", data={"numerator": "(s+1)", "denominator": "(s+2)", "submit_action": "bode"})
    assert "Non-proper" not in resp.get_data(as_text=True)


def test_repeated_imaginary_axis_pair_is_not_reported_as_unstable(client):
    resp = client.post("/bode_plot/", data={"numerator": "[1]", "denominator": "(s^2+1)^2", "submit_action": "bode"})
    html = resp.get_data(as_text=True)
    assert "open-loop unstable" not in html
    assert "imaginary axis" in html


def test_sample_on_imaginary_axis_pole_yields_null_not_nan(client):
    resp = client.post("/bode_plot/", data={"numerator": "[1]", "denominator": "[1, 0, 1]",
                                            "w_min": "1", "w_max": "100", "submit_action": "bode"})
    html = resp.get_data(as_text=True)
    data = _bode_data(html)
    assert data["magnitude_db"][0] is None
    assert all(v is not None for v in data["phase_deg"])
    assert "NaN" not in html.split("window.bodeData")[1].split(";\n")[0]


def test_csv_matches_page_and_contains_asymptotes(client):
    page = client.post("/bode_plot/", data={"numerator": "[1]", "denominator": "(s+1)^3", "submit_action": "bode"})
    data = _bode_data(page.get_data(as_text=True))
    resp = client.get("/bode_plot/download_csv?numerator=[1]&denominator=(s%2B1)^3")
    assert resp.status_code == 200
    rows = list(csv.reader(io.StringIO(resp.get_data(as_text=True))))
    assert rows[0] == ['Frequency (rad/s)', 'Magnitude (dB)', 'Phase (deg)',
                       'Magnitude asymptote (dB)', 'Phase asymptote (deg)']
    assert float(rows[-1][2]) == pytest.approx(data["phase_deg"][-1], abs=1e-3)
    assert float(rows[-1][2]) < -260


def test_download_error_is_plain_text_and_escaped(client):
    resp = client.get("/bode_plot/download_csv?numerator=%3Cimg%20src%3Dx%3E&denominator=[1,1]")
    assert resp.status_code == 400
    assert resp.mimetype == "text/plain"
    assert "<img" not in resp.get_data(as_text=True)


def test_download_png_works(client):
    resp = client.get("/bode_plot/download_png?numerator=(s%2B2)&denominator=(s%2B10)(s%2B0.1)")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    assert resp.data[:8] == b"\x89PNG\r\n\x1a\n"


# --- Nyquist ----------------------------------------------------------------------------

def _nyquist_items(num, den):
    a = bp.analyse_transfer_function(num, den)
    data, items = bp._make_nyquist_data(a.num, a.den, a.response["poles"], None, None,
                                        zeros=a.response["zeros"], w=a.response["omega"], margins=a.margins)
    return data, {item["title"]: item for item in items}


def test_nyquist_first_order_system_has_no_real_axis_crossing():
    data, items = _nyquist_items("[1]", "(s+1)")
    assert items["Real-axis intercepts"]["value"] == "0 crossing(s)"
    assert data["negative"]["frequencies"][0] == -data["positive"]["frequencies"][-1]
    assert 0.0 not in data["negative"]["frequencies"]


def test_nyquist_crossing_left_of_minus_one_is_explained_as_negative_gain_margin():
    data, items = _nyquist_items("[10]", "(s+1)^3")
    item = items["Real-axis intercepts"]
    assert item["value"] == "1 crossing(s)"
    assert "left of" in item["detail"]
    assert "additional phase margin" not in item["detail"]
    assert data["real_axis_crossings"][0]["real"] == pytest.approx(-1.25, abs=0.01)
    assert items["Unity-feedback verdict"]["value"] == "Predicted unstable"


def test_nyquist_encirclements_for_open_loop_unstable_plant():
    _, items = _nyquist_items("[2]", "(s-1)")
    verdict = items["Unity-feedback verdict"]
    assert verdict["value"] == "Predicted stable"
    assert "1 counter-clockwise encirclement" in verdict["detail"]
    assert "Z = P − N = 0" in verdict["detail"]
    assert "counter-clockwise" in items["Open-loop pole distribution"]["detail"]


def test_nyquist_real_locus_is_reported_as_such():
    _, items = _nyquist_items("[2]", "[1]")
    assert items["Real-axis intercepts"]["value"] == "on the real axis"


def test_nyquist_ill_posed_loop():
    _, items = _nyquist_items("[-1]", "[1]")
    assert items["Unity-feedback verdict"]["value"] == "Ill-posed loop"


def test_nyquist_route(client):
    resp = client.post("/bode_plot/", data={"numerator": "[10]", "denominator": "[1, 3, 2, 0]",
                                            "submit_action": "nyquist"})
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "window.nyquistData" in html
    assert "Nyquist Plot Analysis" in html


# --- findings from the review pass ---------------------------------------------------------

def test_repeated_axis_roots_from_coefficient_lists_are_snapped_to_the_axis():
    # (s^2+1)^2 (s+1)^5 as a degree-9 coefficient list goes through np.roots, which scatters
    # the double axis pair by ~3e-8 into both half-planes
    coeffs = np.polymul(np.polymul([1, 0, 2, 0, 1], np.poly([-1] * 5)), [1]).tolist()
    b = bp.analyse_transfer_function("1", "[" + ", ".join(str(c) for c in coeffs) + "]")
    a = bp.analyse_transfer_function("1", "(s^2+1)^2(s+1)^5")
    assert [bp._root_kind(p) for p in b.response["poles"]].count("axis") == 4
    assert not any("open-loop unstable" in w for w in b.warnings)
    assert a.response["phase_deg"][-1] == pytest.approx(b.response["phase_deg"][-1], abs=1.0)


def test_phase_branch_does_not_depend_on_plotted_window():
    default = bp.analyse_transfer_function("(s+1)", "(s-1)(s+10)^3").response
    below = bp.analyse_transfer_function("(s+1)", "(s-1)(s+10)^3", ("0.001", "0.01")).response
    above = bp.analyse_transfer_function("(s+1)", "(s-1)(s+10)^3", ("100", "1000")).response
    assert default["phase_dc_deg"] == below["phase_dc_deg"] == above["phase_dc_deg"] == -180.0
    at_100 = np.interp(2.0, np.log10(default["omega"]), default["phase_deg"])
    assert above["phase_deg"][0] == pytest.approx(at_100, abs=0.5)


def test_sample_on_axis_zero_keeps_factor_phase():
    r = bp.analyse_transfer_function("(s^2+100)", "(s+1)^2", ("0.1", "10")).response
    assert abs(r["phase_deg"][-1] - r["phase_straight_deg"][-1]) < 20.0


def test_bandwidth_finds_narrow_notch():
    a = bp.analyse_transfer_function("[1, 2e-6, 1]", "[1, 2e-3, 1]")
    assert a.bandwidth["status"] == "value"
    assert a.bandwidth["value"] == pytest.approx(0.999, abs=2e-3)


def test_gain_margin_is_zero_when_phase_passes_180_at_undamped_pole():
    m = bp.analyse_transfer_function("1", "(s^2+1)(s+1)").margins
    assert m["gain_margin_infinite"] is False
    assert m["gain_margin_zero"] is True
    assert m["phase_crossover_freq"] == pytest.approx(1.0)


def test_phase_crossover_at_dc_is_flagged():
    p = bp.analyse_transfer_function("1", "(s-1)").bode_payload()
    assert p["phase_crossover_at_dc"] is True
    assert p["phase_crossover_freq"] is None
    assert p["gain_margin_db"] == pytest.approx(0.0)


@pytest.mark.parametrize("text", ["(s+1)^+5000", "(s+1)^(+5000)", "(s+1)^(2*3000)", "(s^2+1)^1000",
                                  "(s+1)^99*(s+2)^99", "(s^5+s^4+s^3+s^2+s+1)^1000"])
def test_degree_bound_is_enforced_before_expansion(text):
    import time
    start = time.perf_counter()
    with pytest.raises(ValueError):
        bp.parse_polynomial(text)
    assert time.perf_counter() - start < 1.0


def test_root_overflow_gives_error_message_not_500(client):
    resp = client.post("/bode_plot/", data={"numerator": "[1]", "denominator": "[1e-320, 1]", "submit_action": "bode"})
    assert resp.status_code == 200
    assert "Error parsing denominator" in resp.get_data(as_text=True)
    resp = client.get("/bode_plot/download_csv?numerator=[1]&denominator=[1e-320,%201]")
    assert resp.status_code == 400


def test_format_helpers_handle_non_finite():
    assert bp._format_real_latex(float("inf")) == r"\infty"
    assert bp.format_complex(complex(float("-inf"), 0)) == r"\(\infty\)"


@pytest.mark.parametrize("num, den", [
    ("(s+1)", "(s^2+1)"),
    ("1", "(s^2+1)(s+100)"),
    ("1", "(s^2+1)(s+1)"),
    ("(s+2)", "(s^2+4)(s+1)"),
    ("1", "s(s^2+1)"),
])
def test_nyquist_axis_pole_jumps_are_not_crossings(num, den):
    data, _ = _nyquist_items(num, den)
    assert data["real_axis_crossings"] == []


def test_nyquist_crossing_is_refined_by_bisection():
    data, _ = _nyquist_items("[10]", "(s+1)^3")
    assert data["real_axis_crossings"][0]["real"] == pytest.approx(-1.25, abs=1e-6)
    assert data["real_axis_crossings"][0]["frequency"] == pytest.approx(np.sqrt(3), abs=1e-6)


@pytest.mark.parametrize("gain", ["8.0001", "8.001", "8"])
def test_nyquist_encirclement_text_never_contradicts_closed_loop_poles(gain):
    _, items = _nyquist_items(gain, "(s+1)^3")
    detail = items["Unity-feedback verdict"]["detail"]
    assert "Z = P − N = 0" not in detail


def test_nyquist_axis_poles_are_not_called_stable():
    for num, den in (("1", "s(s+1)"), ("1", "s^2"), ("1", "(s^2+1)")):
        _, items = _nyquist_items(num, den)
        item = items["Open-loop pole distribution"]
        assert item["level"] == "warning"
        assert "left half-plane" not in item["detail"]


def test_nyquist_negative_dc_gain_text_matches_margin_card():
    for num, den in (("-0.5", "(s+1)"), ("2", "(s-1)")):
        _, items = _nyquist_items(num, den)
        detail = items["Real-axis intercepts"]["detail"]
        assert "gain margin is infinite" not in detail
        assert "negative real axis" in detail


def test_nyquist_distance_card_has_no_nested_math_delimiters():
    _, items = _nyquist_items("(s+2)", "(s+10)(s+0.1)")
    detail = items["Distance to critical point"]["detail"]
    assert detail.count(r"\(") == detail.count(r"\)")
    assert r"= \(" not in detail
