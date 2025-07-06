from __future__ import annotations


import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from flask import Blueprint, render_template, request, abort
import sympy as sp
from laplace_utils import (
    parse_poly,
    coeffs_to_poly,
    inverse_laplace_expr,
    impulse_response,
    poly_long_division,
)

logger = logging.getLogger(__name__)


inverse_laplace_bp = Blueprint("inverse_laplace", __name__)
@inverse_laplace_bp.route("/", methods=["GET", "POST"])

def inverse_laplace():
    num_txt = request.form.get("numerator", "[1]")
    den_txt = request.form.get("denominator", "[1, 1]")
    dt_txt = request.form.get("dt", "1")
    causal = request.form.get("causal", "on") == "on"
    ht_ltx = None
    tf_ltx = None
    seq = None
    q_ltx = None
    r_ltx = None
    error = None
    if request.method == "POST":
        try:
            num = parse_poly(num_txt)
            den = parse_poly(den_txt)
            dt = float(dt_txt)
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
                    return inverse_laplace_expr(num, den, causal=causal)

                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(worker)
                    expr = fut.result(timeout=10)
                ht_ltx = sp.latex(expr).replace("\\theta", "\\varepsilon")
                seq = impulse_response(num, den, N=10, dt=dt)
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
        default_dt=dt_txt,
        default_causal=causal,
        tf_latex=tf_ltx,
        ht_latex=ht_ltx,
        seq=seq,
        q_latex=q_ltx,
        r_latex=r_ltx,
        error=error,
    )