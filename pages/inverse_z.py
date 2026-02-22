from flask import Blueprint, render_template, request
import numpy as np
import sympy as sp
from utils.z_utils import (
    parse_input,
    inverse_z_expr,
    roc_latex,
    eval_expression,
    DiscreteLatexPrinter,
)
import utils.sympy_utils as sp_utils


inverse_z_bp = Blueprint('inverse_z', __name__)
@inverse_z_bp.route('/', methods=['GET','POST'])
def inverse_z():
    num_txt = request.form.get('numerator','[1,0]')
    den_txt = request.form.get('denominator','[1]')
    roc_type = request.form.get('roc_type','causal')
    sf_ltx = sf_parts_ltx = hk_ltx = roc_ltx = seq = seq_title = seq_data = error_msg = None


    if request.method == 'POST':
        try:
            ltx_printer = DiscreteLatexPrinter()
            expr2ltx = lambda expr: ltx_printer.doprint(expr, simplify_expr=False)
            z = sp.symbols('z', complex=True)
            k = sp.symbols('k', integer=True)

            H = sp.simplify(parse_input(num_txt) / parse_input(den_txt))
            num, den = H.as_numer_denom()
            try:
                num_coeffs = num.as_poly(z).all_coeffs()
                den_coeffs = den.as_poly(z).all_coeffs()
            except Exception as exc:
                raise NotImplementedError(f"No rational function input: {exc}")
            # LaTeX for H(z)
            num_expr = sp_utils.coeffs_to_poly(num_coeffs, z)
            den_expr = sp_utils.coeffs_to_poly(den_coeffs, z)
            if den_expr==1:
                sf_ltx = expr2ltx(num_expr)
            else:
                sf_ltx = rf"\frac{{{expr2ltx(num_expr)}}}{{{expr2ltx(den_expr)}}}"
            hk, parts, roc_radius, exclude_origin = inverse_z_expr(num_coeffs, den_coeffs, roc_type)
            roc_ltx = roc_latex(roc_type, roc_radius, exclude_origin)
            sf_parts_ltx = ltx_printer.doprint(parts, simplify_expr=False)
            if sf_ltx == sf_parts_ltx:
                sf_parts_ltx = None
            hk_ltx = ltx_printer.doprint(hk)

            # 10 samples
            n = 10
            k_eval = np.arange(n) if roc_type == "causal" else np.arange(-n+1, 1)
            if roc_type == "causal":
                seq_title = f"First {n} samples"
            else:
                seq_title = f"Last {n} samples"
            seq_raw = eval_expression(hk, k_eval, k)
            seq_data = {
                "k": k_eval.tolist(),
                "re": [float(sp.N(sp.re(s))) for s in seq_raw],
                "im": [float(sp.N(sp.im(s))) for s in seq_raw],
            }

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            print(error_msg)
    page_title = "Inverse Z-Transform Calculator Online | Signal Processing Toolkit"
    meta_description = "Compute inverse z-transforms of rational H(z) with selectable ROC and visualize resulting discrete-time impulse response samples."

    try:
        return render_template(
        'inverse_z.html',
        default_num=num_txt,
        default_den=den_txt,
        sf_latex=sf_ltx,
        sf_parts_latex=sf_parts_ltx,
        hk_latex=hk_ltx,
        seq_title=seq_title,
        seq_data=seq_data,
        roc_type=roc_type,
        roc_latex=roc_ltx,
        error=error_msg.split(': ')[0] if error_msg else None,
        page_title=page_title,
        meta_description=meta_description,
    )
    except Exception as exc:
        print(f"Rendering error: {exc}")
        return render_template(
            'inverse_z.html',
            default_num=num_txt,
            default_den=den_txt,
            error=f"Rendering error: {exc}",
            page_title=page_title,
            meta_description=meta_description,
        )