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
        s = sp.symbols('s')
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


@bode_plot_bp.route('/bode_plot', methods=['GET', 'POST'])
def bode_plot():
    error = ""
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
    
    # Generate frequency axis (0.1 to 100 rad/s) and compute H(s) with s = jω.
    w = np.logspace(-1, 2, 500)
    s = 1j * w
    H = np.polyval(num, s) / np.polyval(den, s)
    
    magnitude = 20 * np.log10(np.abs(H))
    phase = np.angle(H, deg=True)
    
    # Poles and zeros
    zeros = np.roots(num)
    poles = np.roots(den)

    # Gain and phase margins
    sys = control.TransferFunction(num, den)
    gm, pm, wg, wp = control.margin(sys)

    
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

    # Compute frequency response
    w = np.logspace(-1, 2, 500)
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
        num, _ = parse_poly_input(num_str)
        den, _ = parse_poly_input(den_str)
    except Exception as e:
        return f"Error parsing inputs: {e}", 400

    # Compute response
    w = np.logspace(-1, 2, 500)
    s = 1j * w
    H = np.polyval(num, s) / np.polyval(den, s)
    mag = 20 * np.log10(np.abs(H))
    ph  = np.angle(H, deg=True)

    # Re‑generate Bode plot at high DPI
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    ax1.semilogx(w, mag); ax1.set_ylabel("Magnitude (dB)"); ax1.grid(True, which='both', linestyle='--')
    ax2.semilogx(w, ph);  ax2.set_ylabel("Phase (°)"); ax2.set_xlabel("Frequency (rad/s)"); ax2.grid(True, which='both', linestyle='--')
    plt.tight_layout()

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

