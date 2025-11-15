from flask import Blueprint, render_template, request, Response
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from ast import literal_eval
import re
import sympy as sp
import csv
import io
import control
from itertools import zip_longest
from typing import Optional


def _evaluate_transfer(num, den, s_values):
    """Safely evaluate the transfer function ``num/den`` at the complex points ``s_values``."""
    s_array = np.atleast_1d(s_values).astype(complex)
    num_vals = np.polyval(num, s_array)
    den_vals = np.polyval(den, s_array)

    response = np.full_like(num_vals, np.nan, dtype=complex)
    valid = den_vals != 0
    response[valid] = num_vals[valid] / den_vals[valid]
    if np.isscalar(s_values):
        return response.item()
    return response

bode_plot_bp = Blueprint('bode_plot', __name__, template_folder='templates')

def _corner_frequencies(poles, zeros):
    """Return sorted unique positive magnitudes of finitepoles and zeros"""
    raw_values = []
    for root in np.concatenate([poles, zeros]):
        if np.isfinite(root):
            mag = abs(root)
            if mag > 0:
                raw_values.append(float(mag))

    if not raw_values:
        return []

    raw_values.sort()
    deduped = []
    for value in raw_values:
        if not deduped or not np.isclose(value, deduped[-1], rtol=1e-9, atol=1e-12):
            deduped.append(value)
    return deduped

def parse_poly_input(expr_str):
    """
    Parse the user input for a polynomial.
    
    The user can either input a list of coefficients using square brackets (e.g. [1, 1+10j, 1-10j])
    or a factorized expression using normal parentheses (e.g. (s+3+2*j)(s+3-2*j)).
    
    Returns a tuple: (coefficients, display_string)
      - If the user input is a coefficient list, display_string is None.
      - If the user input is factorized, display_string is the raw input (as entered) used for display.
    """
    expr_str = expr_str.strip()
    if expr_str.startswith('['):
        try:
            coeffs = literal_eval(expr_str)
            if not (isinstance(coeffs, list) and all(isinstance(x, (int, float, complex)) for x in coeffs)):
                raise ValueError
        except Exception:
            raise ValueError("Coefficients must be entered as a list of numbers, e.g. [1, 1+10j, 1-10j].")
        return coeffs, None
    else:
        # Factorized mode.
        # Insert multiplication operator between adjacent parentheses if missing.
        expr_fixed = re.sub(r'\)\s*\(', ')*(', expr_str)
        # Replace "j" with sympy's "I"
        expr_fixed = expr_fixed.replace("j", "I")
        s = sp.symbols('s', complex=True)
        try:
            expr_sympy = sp.sympify(expr_fixed, locals={'s': s})
        except Exception as e:
            raise ValueError("Could not parse the factorized expression: " + str(e))
        # Expand the expression to get a standard polynomial.
        expr_expanded = sp.expand(expr_sympy)
        poly = sp.Poly(expr_expanded, s)
        coeffs = [complex(coeff.evalf()) for coeff in poly.all_coeffs()]
        return coeffs, expr_str  # use the user's original factorized expression for display

def format_polynomial(coeffs, var="s"):
    """
    Format a list of coefficients into a human-friendly polynomial string.
    This function handles real and complex coefficients.
    """
    terms = []
    n = len(coeffs)
    for i, coeff in enumerate(coeffs):
        power = n - i - 1
        if abs(coeff) < 1e-12:
            continue
        if isinstance(coeff, complex):
            a = coeff.real
            b = coeff.imag
            coeff_str = f"({a:.2g}"
            if b >= 0:
                coeff_str += f"+{b:.2g}j)"
            else:
                coeff_str += f"{b:.2g}j)"
        else:
            if power > 0:
                if coeff == 1:
                    coeff_str = ""
                elif coeff == -1:
                    coeff_str = "-"
                else:
                    coeff_str = f"{coeff}"
            else:
                coeff_str = f"{coeff}"
        if power == 0:
            term = f"{coeff_str}"
        elif power == 1:
            term = f"{coeff_str}{var}"
        else:
            term = f"{coeff_str}{var}^{power}"
        terms.append(term)
    if not terms:
        return "0"
    poly_str = terms[0]
    for term in terms[1:]:
        if term[0] == "-":
            poly_str += term
        else:
            poly_str += "+" + term
    return poly_str

def _format_real_latex(value: float) -> str:
    """Return a LaTeX-friendly string for a real number."""
    if abs(value) < 1e-12:
        return "0"

    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    exponent = int(np.floor(np.log10(magnitude))) if magnitude != 0 else 0

    if exponent >= 3 or exponent <= -3:
        scaled = magnitude / (10 ** exponent)
        # Avoid printing 1.0 × 10^n when the mantissa is effectively one.
        if abs(scaled - 1) < 1e-9:
            mantissa_part = ""
        else:
            mantissa_part = f"{scaled:.3g}\\times"
        return f"{sign}{mantissa_part}10^{{{exponent}}}"

    return f"{sign}{magnitude:.6g}"


def format_complex(val: complex) -> str:
    """Format a complex number as an inline LaTeX string."""
    real_part = val.real
    imag_part = val.imag

    real_str = None
    if abs(real_part) >= 1e-6:
        real_str = _format_real_latex(real_part)

    imag_str = None
    if abs(imag_part) >= 1e-6:
        imag_abs = _format_real_latex(abs(imag_part))
        if imag_abs in {"0", "1"}:
            imag_abs = ""
        imag_unit = "\\mathrm{j}"
        if real_str is None:
            sign = "-" if imag_part < 0 else ""
            imag_str = f"{sign}{imag_abs}{imag_unit}" if imag_abs else f"{sign}{imag_unit}"
        else:
            sign = "-" if imag_part < 0 else "+"
            if imag_abs:
                imag_str = f"{sign} {imag_abs}{imag_unit}"
            else:
                imag_str = f"{sign} {imag_unit}"

    if real_str is None and imag_str is None:
        content = "0"
    elif real_str is None:
        content = imag_str
    elif imag_str is None:
        content = real_str
    else:
        content = f"{real_str} {imag_str}"

    return rf"\({content}\)"


def _format_real_text(value: float, unit: Optional[str] = None) -> str:
    """Format a real value into a concise human-readable string."""
    if value is None or not np.isfinite(value):
        return "—"

    abs_val = abs(value)
    if abs_val != 0 and (abs_val >= 1e4 or abs_val <= 1e-3):
        formatted = f"{value:.3e}"
    else:
        formatted = f"{value:.4f}".rstrip('0').rstrip('.')

    return f"{formatted} {unit}" if unit else formatted


def _format_latex_real(value: float, unit: Optional[str] = None) -> str:
    """Return a LaTeX inline string for a real number with an optional unit."""
    if value is None or not np.isfinite(value):
        return "—"
    content = _format_real_latex(value)
    if unit:
        return rf"\({content}\;{unit}\)"
    return rf"\({content}\)"


def _analysis_item(title: str, value: str, detail: str, level: str = "info") -> dict:
    return {"title": title, "value": value, "detail": detail, "level": level}


def _make_nyquist_data(num, den, poles, gm_db, pm):
    """Compute Nyquist contour samples and qualitative analysis information."""
    base_w = _make_freq_vector(num, den)
    w_positive = np.concatenate(([0.0], base_w))
    s_positive = 1j * w_positive
    resp_positive = _evaluate_transfer(num, den, s_positive)

    # Mirror the contour for negative frequencies.
    w_negative = -w_positive[-2::-1]
    s_negative = 1j * w_negative
    resp_negative = _evaluate_transfer(num, den, s_negative)

    nyquist_data = {
        "positive": {
            "frequencies": w_positive.tolist(),
            "real": np.real(resp_positive).tolist(),
            "imag": np.imag(resp_positive).tolist(),
        },
        "negative": {
            "frequencies": w_negative.tolist(),
            "real": np.real(resp_negative).tolist(),
            "imag": np.imag(resp_negative).tolist(),
        },
        "critical_point": {"real": -1.0, "imag": 0.0},
    }

    low_freq = resp_positive[0]
    high_freq = resp_positive[-1]

    def _complex_dict(value: complex, frequency: float) -> dict:
        if value is None or not np.isfinite(value):
            return {"frequency": float(frequency), "real": None, "imag": None}
        return {
            "frequency": float(frequency),
            "real": float(np.real(value)),
            "imag": float(np.imag(value)),
        }

    nyquist_data["low_freq"] = _complex_dict(low_freq, w_positive[0])
    nyquist_data["high_freq"] = _complex_dict(high_freq, w_positive[-1])

    analysis_items = []

    # Open-loop pole information
    rhp_poles = sum(1 for p in poles if np.real(p) > 0)
    if rhp_poles == 0:
        analysis_items.append(
            _analysis_item(
                "Open-loop pole distribution",
                "Stable (no RHP poles)",
                "All open-loop poles lie in the left half-plane, so the Nyquist contour is not required to encircle the critical point for unity-feedback stability.",
                level="success",
            )
        )
    else:
        analysis_items.append(
            _analysis_item(
                "Open-loop pole distribution",
                f"{rhp_poles} pole(s) in RHP",
                r"Right-half-plane poles demand the Nyquist contour encircle \((-1,0\mathrm{j})\) the same number of times for closed-loop stability.",
                level="warning",
            )
        )

    # Minimum distance to -1+j0 (critical point)
    distances = np.abs(resp_positive + 1)
    with np.errstate(invalid='ignore'):
        finite_mask = np.isfinite(distances)
    if np.any(finite_mask):
        idx_min = np.nanargmin(distances)
        d_min = distances[idx_min]
        w_at_min = w_positive[idx_min]
        analysis_items.append(
            _analysis_item(
                "Distance to critical point",
                _format_latex_real(d_min),
                (
                    r"The Nyquist locus comes closest to the critical point at {freq} with \(|L(\mathrm{{j}}\omega)+1| = {distance}\)."
                ).format(freq=_format_latex_real(w_at_min, 'rad/s'), distance=_format_latex_real(d_min)),
                level="info" if d_min > 0.2 else "warning",
            )
        )

    # Real-axis crossings for ω > 0
    crossings = []
    imag_vals = np.imag(resp_positive)
    real_vals = np.real(resp_positive)
    for i in range(len(imag_vals) - 1):
        y1, y2 = imag_vals[i], imag_vals[i + 1]
        if not (np.isfinite(y1) and np.isfinite(y2)):
            continue
        if y1 == 0:
            crossings.append((w_positive[i], real_vals[i]))
        if y1 * y2 < 0:
            # Linear interpolation for the crossing
            t = -y1 / (y2 - y1)
            if 0 <= t <= 1:
                w_cross = w_positive[i] + t * (w_positive[i + 1] - w_positive[i])
                real_cross = real_vals[i] + t * (real_vals[i + 1] - real_vals[i])
                crossings.append((w_cross, real_cross))
    if crossings:
        crossings.sort(key=lambda pair: pair[0])
        crossing_text = ", ".join(
            f"{_format_latex_real(rc, 'rad/s')} → {_format_latex_real(rr)}"
            for rc, rr in crossings
        )
        detail = (
            r"Real-axis crossings (\(\Im\{{L(\mathrm{{j}}\omega)\}}=0\)) occur at: {points}. Real parts to the left of \(-1\) typically imply additional phase margin."
        ).format(points=crossing_text)
    else:
        detail = r"The Nyquist locus does not cross the real axis for positive frequencies, indicating the phase never reaches \(\pm180^\circ\)."
    analysis_items.append(
        _analysis_item(
            "Real-axis intercepts",
            f"{len(crossings)} crossing(s)",
            detail,
            level="info",
        )
    )

    # Unity-feedback closed-loop pole estimate
    closed_loop_den = np.polyadd(den, num)
    closed_loop_poles = np.roots(closed_loop_den)
    unstable_closed = sum(1 for p in closed_loop_poles if np.real(p) >= 0)
    if unstable_closed == 0:
        cl_value = "Predicted stable"
        cl_detail = "Closed-loop poles (assuming unity feedback) lie in the left half-plane, consistent with zero Nyquist encirclements of the critical point."
        level = "success"
    else:
        cl_value = "Predicted unstable"
        cl_detail = (
            r"{count} closed-loop pole(s) fall in the right half-plane for unity feedback. The Nyquist contour must encircle \((-1,0\mathrm{{j}})\) appropriately to recover stability."
        ).format(count=unstable_closed)
        level = "warning"
    analysis_items.append(
        _analysis_item(
            "Unity-feedback verdict",
            cl_value,
            cl_detail,
            level=level,
        )
    )

    # Gain/phase margin recap
    gm_text = _format_latex_real(gm_db, 'dB') if gm_db is not None else "—"
    pm_text = _format_latex_real(pm, r'^\circ') if pm is not None else "—"
    analysis_items.append(
        _analysis_item(
            "Classical margins",
            f"GM: {gm_text} / PM: {pm_text}",
            r"Margins read from the Bode plot appear here as a reminder of how far the Nyquist locus stays from the critical point along the real axis (gain margin) and around \(-1\) (phase margin).",
            level="info",
        )
    )

    return nyquist_data, analysis_items

def _make_freq_vector(num, den, override=None):
    """Return frequency vector for Bode plot.

    If ``override`` is a ``(w_min, w_max)`` tuple, it is used directly.
    Otherwise, the range is chosen based on the magnitudes of the finite
    poles and zeros of ``num/den``.  Dense clusters (<1.5 decades wide)
    are padded symmetrically to cover at least two decades around the
    median corner frequency.
    """
    if override is not None:
        try:
            w_min, w_max = float(override[0]), float(override[1])
            if w_min > 0 and w_max > w_min:
                return np.logspace(np.log10(w_min), np.log10(w_max), 500)
        except Exception:
            pass  # fall back to auto-scaling if invalid

    sys = control.TransferFunction(num, den)
    # use control library helpers to extract poles and zeros
    poles = control.poles(sys)
    zeros = control.zeros(sys)
    w_list = np.abs(np.concatenate([poles, zeros]))
    w_list = w_list[np.isfinite(w_list) & (w_list > 0)]

    if w_list.size:
        w_min = 0.1 * w_list.min()
        w_max = 10 * w_list.max()
    else:
        w_min, w_max = 1e-2, 1e2

    if np.log10(w_max) - np.log10(w_min) < 1.5:
        center = np.median(w_list) if w_list.size else np.sqrt(w_min * w_max)
        w_min = min(w_min, center / 10)
        w_max = max(w_max, center * 10)

    return np.logspace(np.log10(w_min), np.log10(w_max), 500)


@bode_plot_bp.route('/', methods=['GET', 'POST'])
def bode_plot():
    error = ""
    warning = ""
    # Default inputs; here we use a coefficient list for H(s) = (s+1)/(s^2+2)
    default_num = "[1, 1]"
    default_den = "[1, 0, 2]"
    
    user_num = default_num
    user_den = default_den
    
    # Default: treat as coefficient list.
    num = [1, 1]
    den = [1, 0, 2]
    # For display, if the user is using coefficients, we generate a formatted polynomial.
    num_disp = format_polynomial(num)
    den_disp = format_polynomial(den)
    
    # Build the transfer function string using LaTeX fraction command
    function_str = f"H(s) = \\frac{{{num_disp}}}{{{den_disp}}}"
    

    if request.method == 'POST':
        submit_action = request.form.get('submit_action', 'bode')
    else:
        submit_action = request.args.get('view', 'bode') or 'bode'
    show_nyquist = submit_action == 'nyquist'

    
    if request.method == 'GET':
        return render_template(
            "bode_plot.html",
            bode_data=None,
            error=None,
            warning=None,
            default_num=default_num,
            default_den=default_den,
            function_str=None,
            zeros=None,
            poles=None,
            pz_pairs=None,
            gm=None,
            pm=None,
            wg=None,
            wp=None,
            pz_plot=None,
            nyquist_data=None,
            nyquist_analysis=None,
            show_nyquist=show_nyquist,
            active_action=submit_action,
        )
    
    if request.method == 'POST':
        user_num = request.form.get('numerator', default_num)
        user_den = request.form.get('denominator', default_den)
        try:
            num, num_raw = parse_poly_input(user_num)
        except Exception as e:
            error = "Error parsing numerator: " + str(e)
        if not error:
            try:
                den, den_raw = parse_poly_input(user_den)
            except Exception as e:
                error = "Error parsing denominator: " + str(e)
        if not error:
            if num_raw is None:
                num_disp = format_polynomial(num)
            else:
                num_disp = num_raw  # display as typed
            if den_raw is None:
                den_disp = format_polynomial(den)
            else:
                den_disp = den_raw
            function_str = f"H(s) = \\frac{{{num_disp}}}{{{den_disp}}}"
        else:
            return render_template(
                "bode_plot.html",
                error=error,
                bode_data=None,
                default_num=user_num,
                default_den=user_den,
                function_str=None,
                zeros=None,
                poles=None,
                pz_pairs=None,
                gm=None,
                pm=None,
                wg=None,
                wp=None,
                pz_plot=None,
                nyquist_data=None,
                nyquist_analysis=None,
                show_nyquist=show_nyquist,
                active_action=submit_action,
            )
    
    # Allow manual frequency range via optional form fields
    rng = None
    if request.method == 'POST':
        w_min_str = request.form.get('w_min')
        w_max_str = request.form.get('w_max')
    else:
        w_min_str = request.args.get('w_min')
        w_max_str = request.args.get('w_max')
    if w_min_str and w_max_str:
        rng = (w_min_str, w_max_str)

    # Frequency vector scaled to poles/zeros (or user override)
    w = _make_freq_vector(num, den, rng)
    s = 1j * w
    H = np.polyval(num, s) / np.polyval(den, s)
    
    mag = np.abs(H)
    magnitude_db = 20 * np.log10(np.maximum(mag, np.finfo(float).tiny))
    phase = np.unwrap(np.angle(H))
    phase_deg = np.degrees(phase)
    
    # Poles and zeros
    # Poles and zeros
    zeros = np.roots(num)
    poles = np.roots(den)

    # (5) Friendly warnings (non-fatal)
    deg_num = len(num) - 1
    deg_den = len(den) - 1
    if deg_num >= deg_den:
        warning += ("Warning: Non-proper transfer function (deg(num) ≥ deg(den)). "
                    "Magnitude may not settle at high frequency.\n")
    if np.any(np.real(poles) > 0):
        warning += "System has right-half-plane pole(s): open-loop unstable.\n"
    if np.any(np.real(zeros) > 0):
        warning += "Non-minimum-phase zero(s) detected (RHP zero).\n"

    # Gain and phase margins
    sys = control.TransferFunction(num, den)
    def finite_or_none(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            return [finite_or_none(v) for v in value]
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    try:
        gm, pm, wg, wp = control.margin(sys)
    except Exception:
        gm = pm = wg = wp = None
    gm_db = None
    if gm is not None and gm > 0 and np.isfinite(gm):
        gm_db = 20 * np.log10(gm)

    try:
        bandwidth = control.bandwidth(sys)
        if np.isnan(bandwidth) or np.isinf(bandwidth):
            bandwidth = None
    except Exception:
        bandwidth = None
        
    corner_freqs = _corner_frequencies(poles, zeros)

    bode_data = {
        "omega": w.tolist(),
        "magnitude_db": magnitude_db.tolist(),
        "phase_deg": phase_deg.tolist(),
        "gain_margin_db": finite_or_none(gm_db),
        "phase_margin_deg": finite_or_none(pm),
        "gain_cross_freq": finite_or_none(wg),
        "phase_cross_freq": finite_or_none(wp),
        "bandwidth": finite_or_none(bandwidth),
        "corner_frequencies": corner_freqs,
    }

    pz_plot = {
        "zeros": [
            {"re": float(np.real(z)), "im": float(np.imag(z))}
            for z in zeros if np.isfinite(z)
        ],
        "poles": [
            {"re": float(np.real(p)), "im": float(np.imag(p))}
            for p in poles if np.isfinite(p)
        ],
    }
    
    # Prepare pole/zero strings for display
    zero_list = [format_complex(z) for z in zeros]
    pole_list = [format_complex(p) for p in poles]
    pz_pairs = list(zip_longest(zero_list, pole_list, fillvalue=""))

    nyquist_plot = None
    nyquist_analysis = None
    if show_nyquist:
        nyquist_plot, nyquist_analysis = _make_nyquist_data(num, den, poles, gm_db, pm)


    return render_template(
        "bode_plot.html",
        bode_data=bode_data,
        error=error,
        warning=(warning if warning else None),
        default_num=user_num,
        default_den=user_den,
        function_str=function_str,
        zeros=zero_list,
        poles=pole_list,
        pz_pairs=pz_pairs,
        gm=gm,
        pm=pm,
        wg=wg,
        wp=wp,
        pz_plot=pz_plot,
        nyquist_data=nyquist_plot,
        nyquist_analysis=nyquist_analysis,
        show_nyquist=show_nyquist,
        active_action=submit_action,
    )

    

@bode_plot_bp.route('/download_csv')
def download_csv():
    # Read the same numerator/denominator strings from query params
    num_str = request.args.get('numerator', '')
    den_str = request.args.get('denominator', '')
    try:
        num, _ = parse_poly_input(num_str)
        den, _ = parse_poly_input(den_str)
    except Exception as e:
        return f"Error parsing inputs: {e}", 400
    # Optional manual range
    rng = None
    w_min_str = request.args.get('w_min')
    w_max_str = request.args.get('w_max')
    if w_min_str and w_max_str:
        rng = (w_min_str, w_max_str)

    # Compute frequency response
    w = _make_freq_vector(num, den, rng)
    s = 1j * w
    H = np.polyval(num, s) / np.polyval(den, s)
    mag = 20 * np.log10(np.abs(H))
    ph  = np.angle(H, deg=True)

    # Build CSV in memory
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Frequency (rad/s)', 'Magnitude (dB)', 'Phase (deg)'])
    for ω, m, p in zip(w, mag, ph):
        cw.writerow([f"{ω:.6g}", f"{m:.6g}", f"{p:.6g}"])
    output = si.getvalue().encode('utf-8')

    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename="bode_data.csv"'}
    )

@bode_plot_bp.route('/download_png')
def download_png():
    # Read the same numerator/denominator strings
    num_str = request.args.get('numerator', '')
    den_str = request.args.get('denominator', '')
    try:
        num, num_raw = parse_poly_input(num_str)
        den, den_raw = parse_poly_input(den_str)
    except Exception as e:
        return f"Error parsing inputs: {e}", 400
    # Optional manual range
    rng = None
    w_min_str = request.args.get('w_min')
    w_max_str = request.args.get('w_max')
    if w_min_str and w_max_str:
        rng = (w_min_str, w_max_str)

    # Compute response
    w = _make_freq_vector(num, den, rng)
    s = 1j * w
    H = np.polyval(num, s) / np.polyval(den, s)
    mag = 20 * np.log10(np.abs(H))
    ph  = np.angle(H, deg=True)

    # Build LaTeX string for the transfer function to display on the plot
    if num_raw is None:
        num_disp = format_polynomial(num)
    else:
        num_disp = num_raw
    if den_raw is None:
        den_disp = format_polynomial(den)
    else:
        den_disp = den_raw
    function_str = rf"$H(s) = \frac{{{num_disp}}}{{{den_disp}}}$"


    # Re‑generate Bode plot at high DPI
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    ax1.semilogx(w, mag)
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which='both', linestyle='--')
    ax2.semilogx(w, ph)
    ax2.set_ylabel("Phase (°)")
    ax2.set_xlabel("Frequency (rad/s)")
    ax2.grid(True, which='both', linestyle='--')
    fig.suptitle(function_str, y=0.98)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])

    buf = BytesIO()
    # Save at 300 DPI for high‑resolution
    plt.savefig(buf, format='png', dpi=300)
    plt.close(fig)

    return Response(
        buf.getvalue(),
        mimetype='image/png',
        headers={'Content-Disposition': 'attachment; filename="bode_plot.png"'}
    )

