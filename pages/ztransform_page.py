# pages/ztransform_page.py

from flask import Blueprint, render_template, request
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp

from utils.eval_helpers import safe_parse_expr

ztransform_bp = Blueprint("ztransform", __name__)

_SEQUENCE_HELP = (
    "Enter a sequence in n such as 0.5^n*Heaviside(n) or "
    "Piecewise((1, (n>=0)&(n<=5)), (0, True)); allowed are numbers, n, z, j, e, pi, "
    "Heaviside, DiracDelta, KroneckerDelta, Piecewise, And, Or, Not, sin, cos, exp, "
    "sqrt, Abs, + - * / ^, comparisons and parentheses."
)


def parse_sequence(expr_str: str, n: sp.Symbol, z: sp.Symbol) -> sp.Expr:
    """Parse the free-form sequence x[n] without executing Python (sympify would)."""
    local_dict = {
        'n': n,
        'z': z,
        'j': sp.I, 'I': sp.I, 'e': sp.E, 'E': sp.E, 'pi': sp.pi, 'oo': sp.oo,
        'Heaviside': sp.Heaviside,
        'DiracDelta': sp.DiracDelta,
        'KroneckerDelta': sp.KroneckerDelta,
        'Piecewise': sp.Piecewise,
        'And': sp.And, 'Or': sp.Or, 'Not': sp.Not,
        'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'sqrt': sp.sqrt, 'Abs': sp.Abs,
        'Rational': sp.Rational,
    }
    return safe_parse_expr(
        expr_str, local_dict=local_dict, allow_relational=True, help_text=_SEQUENCE_HELP
    )

@ztransform_bp.route("/", methods=["GET", "POST"])
def ztransform():
    error = None
    z_expr_latex = None
    plot_data = None

    # 1) Read form inputs
    seqType = request.form.get("seqType", "custom")
    expr_str = request.form.get("expr", "")
    n_min   = request.form.get("n_min", "0")
    n_max   = request.form.get("n_max", "10")
    do_plot = (request.form.get("plot") == "on")

    # Parse integer bounds
    try:
        n0 = int(n_min)
        n1 = int(n_max)
    except:
        n0, n1 = 0, 10

    # Define symbols
    n = sp.symbols('n', integer=True)
    z = sp.symbols('z', complex=True)


    # 2) Build the sequence expression based on the chosen template
    try:
        if seqType == 'impulses':
            expr = 0
            # read up to 4 impulse terms
            for i in range(4):
                c = request.form.get(f"imp_c{i}")
                k = request.form.get(f"imp_k{i}")
                if c and k:
                    expr += float(c) * sp.DiracDelta(n - int(k))

        elif seqType == 'exp':
            a = float(request.form.get("a", 1))
            expr = a**n * sp.Heaviside(n)

        elif seqType == 'damped_cos':
            a     = float(request.form.get("a", 1))
            omega = float(request.form.get("omega", 1))
            expr = a**n * sp.cos(omega*n) * sp.Heaviside(n)

        elif seqType == 'pulse':
            n1 = int(request.form.get("n_start", 0))
            n2 = int(request.form.get("n_end", 5))
            expr = sp.Piecewise((1, (n>=n1)&(n<=n2)), (0, True))

        else:  # custom free-form
            expr = parse_sequence(expr_str, n, z)

        # 3) Symbolic Z-transform
        if seqType == 'impulses':
            # finite sum over known impulse positions
            Xz = sp.summation(expr * z**(-n), (n, n0, n1))
        else:
            # infinite sum for most other templates
            Xz = sp.summation(expr * z**(-n), (n, n0, sp.oo))

        Xz_s = sp.simplify(Xz)
        z_expr_latex = sp.latex(Xz_s)

        # 4) Optional numeric unit-circle plot
        if do_plot:
            # build numeric x[n]
            n_vals = np.arange(n0, n1+1)
            x_vals = np.array([float(expr.subs(n, int(k))) for k in n_vals], dtype=float)

            # ω grid on [0,2π]
            ω = np.linspace(0, 2*np.pi, 512)
            Z = np.exp(1j * ω)

            # compute X(e^{jω})
            Xw = np.array([np.sum(x_vals * Z[j]**(-n_vals)) for j in range(len(ω))], dtype=complex)
            mag   = np.abs(Xw)
            phase = np.angle(Xw)
            if mag.max() > 0:
                mag /= mag.max()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), layout="constrained")
            ax1.plot(ω, mag)
            ax1.set_title(r'|X(e^{\mathrm{j}\omega})| (normalized)')
            ax1.set_xlabel(r'$\omega$ (rad/sample)'); ax1.grid(True)

            ax2.plot(ω, phase)
            ax2.set_title(r'∠X(e^{\mathrm{j}\omega})')
            ax2.set_xlabel(r'$\omega$ (rad/sample)'); ax2.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plot_data = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "ztransform.html",
        error=error,
        z_expr_latex=z_expr_latex,
        plot_data=plot_data,
        expr=expr_str,
        n_min=n_min,
        n_max=n_max,
        do_plot=do_plot
    )
