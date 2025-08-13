from __future__ import annotations


import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from flask import Blueprint, render_template, request, abort
import sympy as sp
from utils.laplace_utils import (
    parse_poly,
    coeffs_to_poly,
    inverse_laplace_expr,
    step_response_expr,
    pretty_latex,
    factor_rational,
    eval_expression,
)
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["text.usetex"] = True # TeX-like font
import numpy as np
import io
import base64


def _parse_poly(txt: str) -> np.ndarray:
    """Helper function used for testing."""
    return parse_poly(txt)


def _inverse_laplace_expr(num, den) -> sp.Expr:
    """Helper function used for testing."""
    return inverse_laplace_expr(num, den)[0]

logger = logging.getLogger(__name__)


inverse_laplace_bp = Blueprint("inverse_laplace", __name__)
@inverse_laplace_bp.route("/", methods=["GET", "POST"])

def inverse_laplace():
    num_txt = request.form.get("numerator", "[1]")
    den_txt = request.form.get("denominator", "[1, 1]")
    response_type = request.form.get("response_type", "step")
    time_signal_ltx = None
    plot_url = None
    title = None
    sf_ltx = None
    pfd_ltx = None
    q_ltx = None
    r_ltx = None
    error = None
    num_error = None
    den_error = None
    
    if request.method == "POST":
        try:
            num = parse_poly(num_txt)
        except Exception as exc:
            logger.exception("parse numerator error")
            num_error = f"{type(exc).__name__}: {exc}"
        try:
            den = parse_poly(den_txt)

        except Exception as exc:
            logger.exception("parse denominator error")
            den_error = f"{type(exc).__name__}: {exc}"

        if not num_error and not den_error:
            try:
                num_expr = coeffs_to_poly(num)
                den_expr = coeffs_to_poly(den)
                sf_ltx = pretty_latex(num_expr/ den_expr, simplify_expr=False)
                sf_fac_ltx = pretty_latex(factor_rational(num_expr, den_expr), simplify_expr=False)

                def worker() -> tuple[sp.Expr, sp.Expr]:
                    if response_type == "step":
                        return step_response_expr(num, den)
                    return inverse_laplace_expr(num, den)
                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(worker)
                    expr_t, expr_pfd = fut.result(timeout=5)

                time_signal_ltx = pretty_latex(expr_t, simplify_expr=False)
                pfd_ltx = pretty_latex(expr_pfd, simplify_expr=False) if expr_pfd else None
                if sf_ltx == pfd_ltx:
                    pfd_ltx = None
                if sf_ltx != sf_fac_ltx:
                    sf_ltx = rf"{sf_ltx} = {sf_fac_ltx}"

                # determine plotting range based on poles
                poles = np.unique(np.roots(den))
                if len(poles) > 0:
                    # Use the slowest pole to choose a reasonable time span
                    tau_candidates = []
                    for pole in poles:
                        try:
                            pole_real = pole.real
                        except AttributeError:
                            pole_real = float(pole)
                        if np.isclose(pole_real, 0, atol=1e-4): # choose by length of period
                            pole_real = pole.imag / (2*np.pi)
                        if np.isclose(pole_real, 0, atol=1e-4):
                            continue
                        denom = max(abs(pole_real), 1e-3)
                        tau_candidates.append(1. / denom)
                    tau = max(tau_candidates) # max tau of all non-zero poles
                else:
                    tau = 1.0
                t_vals = np.linspace(0, 5 * tau, 1000)

                t = sp.symbols("t", real=True)
                y = eval_expression(expr_t, t_vals, t)
                if response_type == "step":
                    title = r"Step Response"
                    time_signal_ltx  = r"s(t)=" + time_signal_ltx
                else:
                    title = r"Impulse Response"
                    time_signal_ltx = r"h(t)=" + time_signal_ltx
                
                t_signal_name_ltx = time_signal_ltx.split("=")[0]
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.axhline(0, color='k', linewidth=0.9)
                ax.plot(t_vals, y.real, label=rf"$\mathrm{{Re}}\{{{t_signal_name_ltx}\}}$")
                ax.plot(t_vals, y.imag, label=rf"$\mathrm{{Im}}\{{{t_signal_name_ltx}\}}$", linestyle="--")
                ax.set_xlim(t_vals[0], t_vals[-1])
                ax.set_xlabel("$t$")
                ax.set_ylabel("$"+t_signal_name_ltx+"$")
                ax.grid()
                ax.legend()
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                plt.close(fig)
                plot_url = base64.b64encode(buf.getvalue()).decode("utf8")
            except TimeoutError:
                logger.exception("symbolic inverse laplace timeout")
                abort(408, "Expression too complex, try numeric impulse")
            except Exception as exc:
                logger.exception("inverse laplace failed")
                error = f"{type(exc).__name__}: {exc}"
    return render_template(
        "inverse_laplace.html",
        default_num=num_txt,
        default_den=den_txt,
        time_signal_latex=time_signal_ltx,
        plot_url=plot_url,
        title=title,
        sf_latex=sf_ltx,
        pfd_latex=pfd_ltx,
        q_latex=q_ltx,
        r_latex=r_ltx,
        error=error,
        num_error=num_error,
        den_error=den_error,
    )