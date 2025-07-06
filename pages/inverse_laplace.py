from __future__ import annotations


import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from flask import Blueprint, render_template, request, abort
import sympy as sp
from laplace_utils import (
    parse_poly,
    coeffs_to_poly,
    inverse_laplace_expr,
    step_response_expr,
    _pretty_latex,
    poly_long_division,
)
from scipy.signal import lti, step as step_func, impulse
from flask import current_app, url_for
import matplotlib.pyplot as plt
import numpy as np
import uuid
import os


def _parse_poly(txt: str) -> np.ndarray:
    return parse_poly(txt)


def _inverse_laplace_expr(num, den) -> sp.Expr:
    return inverse_laplace_expr(num, den)

logger = logging.getLogger(__name__)


inverse_laplace_bp = Blueprint("inverse_laplace", __name__)
@inverse_laplace_bp.route("/", methods=["GET", "POST"])

def inverse_laplace():
    num_txt = request.form.get("numerator", "[1]")
    den_txt = request.form.get("denominator", "[1, 1]")
    response_type = request.form.get("response_type", "step")
    latex_expr = None
    plot_url = None
    title = None
    tf_ltx = None
    q_ltx = None
    r_ltx = None
    error = None
    if request.method == "POST":
        try:
            num = parse_poly(num_txt)
            den = parse_poly(den_txt)
        except ValueError as exc:
            logger.exception("invalid input")
            abort(400, str(exc))
        except Exception as exc:
            logger.exception("parse error")
            error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                num_expr = coeffs_to_poly(num)
                den_expr = coeffs_to_poly(den)
                tf_ltx = sp.latex(num_expr / den_expr)
                if len(num) - 1 >= len(den) - 1:
                    q_coeffs, r_coeffs = poly_long_division(num, den)
                    q_ltx = sp.latex(coeffs_to_poly(q_coeffs))
                    r_ltx = sp.latex(coeffs_to_poly(r_coeffs) / den_expr)

                def worker() -> sp.Expr:
                    if response_type == "step":
                        return step_response_expr(num, den)
                    return inverse_laplace_expr(num, den)
                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(worker)
                    expr = fut.result(timeout=10)

                latex_expr = _pretty_latex(expr)

                sys = lti(num, den)
                poles = getattr(sys, "poles", [])
                if len(poles) > 0:
                    # Use the slowest pole to choose a reasonable time span
                    tau_candidates = []
                    for pole in poles:
                        try:
                            pole_real = pole.real
                        except AttributeError:
                            pole_real = float(pole)
                        denom = max(abs(pole_real), 1e-3)
                        tau_candidates.append(1.0 / denom)
                    tau = max(tau_candidates)
                else:
                    tau = 1.0
                t_vals = np.linspace(0, 5 * tau, 1000)
                if response_type == "step":
                    tout, y = step_func(sys, T=t_vals)
                    title = "Step response y(t)"
                else:
                    tout, y = impulse(sys, T=t_vals)
                    title = "Impulse response h(t)"
                plt.figure(figsize=(5, 3))
                plt.plot(tout, y)
                plt.xlabel("t")
                plt.ylabel(title.split()[0])
                plt.tight_layout()
                fname = f"{uuid.uuid4().hex}.png"
                plot_dir = os.path.join(current_app.static_folder, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, fname)
                plt.savefig(plot_path)
                plt.close()
                plot_url = url_for("static", filename=f"plots/{fname}")
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
        latex_expr=latex_expr,
        plot_url=plot_url,
        title=title,
        tf_latex=tf_ltx,
        q_latex=q_ltx,
        r_latex=r_ltx,
        error=error,
    )