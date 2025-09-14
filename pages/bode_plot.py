from flask import Blueprint, render_template, request, Response
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64
from ast import literal_eval
import re
import sympy as sp
import csv
import io
import control
from itertools import zip_longest

bode_plot_bp = Blueprint('bode_plot', __name__, template_folder='templates')

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

def format_complex(val: complex) -> str:
    """Format a complex number with a small imaginary threshold."""
    if abs(val.imag) < 1e-6:
        return f"{val.real:.3g}"
    sign = "+" if val.imag >= 0 else "-"
    return f"{val.real:.3g}{sign}{abs(val.imag):.3g}j"

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


@bode_plot_bp.route('/bode_plot', methods=['GET', 'POST'])
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
    
    if request.method == 'GET':
        return render_template(
            "bode_plot.html",
            plot_url=None,
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
                plot_url=None,
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
    
    magnitude = 20 * np.log10(np.abs(H))
    phase = np.angle(H, deg=True)
    
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
    gm, pm, wg, wp = control.margin(sys)

    # (7) Pole–Zero map (mini plot)
    figpz, axpz = plt.subplots(figsize=(3, 3))
    axpz.axhline(0, color='#cccccc'); axpz.axvline(0, color='#cccccc')
    if zeros.size:
        axpz.scatter(np.real(zeros), np.imag(zeros), marker='o', label='zeros')
    if poles.size:
        axpz.scatter(np.real(poles), np.imag(poles), marker='x', label='poles')
    axpz.set_xlabel('Re{s}'); axpz.set_ylabel('Im{s}')
    axpz.set_title('Pole–Zero Map')
    axpz.grid(True, linestyle='--', alpha=0.4)
    bufpz = BytesIO()
    plt.tight_layout()
    figpz.savefig(bufpz, format='png', dpi=150)
    bufpz.seek(0)
    pz_img = base64.b64encode(bufpz.getvalue()).decode('utf-8')
    plt.close(figpz)


    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.semilogx(w, magnitude)
    ax1.set_title("Bode Plot - Magnitude")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both", linestyle="--")
    
    ax2.semilogx(w, phase)
    ax2.set_title("Bode Plot - Phase")
    ax2.set_xlabel("Frequency (rad/s)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.grid(True, which="both", linestyle="--")
    
    # Annotate crossover frequencies on the plots
    legend_lines = []
    legend_labels = []
    if np.isfinite(wp):
        phase_line = ax1.axvline(wp, color="r", linestyle="--")
        ax2.axvline(wp, color="r", linestyle="--")
        legend_lines.append(phase_line)
        legend_labels.append("Phase crossover")
    if np.isfinite(wg):
        gain_line = ax1.axvline(wg, color="orange", linestyle="--")
        ax2.axvline(wg, color="orange", linestyle="--")
        legend_lines.append(gain_line)
        legend_labels.append("Gain crossover")
    if legend_lines:
        ax1.legend(legend_lines, legend_labels, loc="best")
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf8")
    plt.close(fig)
    
    # Prepare pole/zero strings for display
    zero_list = [format_complex(z) for z in zeros]
    pole_list = [format_complex(p) for p in poles]
    pz_pairs = list(zip_longest(zero_list, pole_list, fillvalue=""))

    return render_template(
        "bode_plot.html",
        plot_url=image_base64,
        error=error,
        warning=(warning if warning else None),
        default_num=user_num,
        default_den=user_den,
        function_str=function_str,
        zeros=zero_list,
        poles=pole_list,
        pz_pairs=pz_pairs,
        pz_img=pz_img,
        gm=gm,
        pm=pm,
        wg=wg,
        wp=wp,
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.semilogx(w, mag)
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which='both', linestyle='--')
    ax2.semilogx(w, ph)
    ax2.set_ylabel("Phase (°)")
    ax2.set_xlabel("Frequency (rad/s)")
    ax2.grid(True, which='both', linestyle='--')
    fig.suptitle(function_str, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    buf = BytesIO()
    # Save at 300 DPI for high‑resolution
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close(fig)

    return Response(
        buf.getvalue(),
        mimetype='image/png',
        headers={'Content-Disposition': 'attachment; filename="bode_plot.png"'}
    )

