"""
Draw a *continuous-time* Direct-Form II block diagram.

Limits
------
• Order ≤ 2 (two integrators). Higher orders fall back to a table.
• Denominator is normalised so the leading coefficient is 1.
"""

from flask import Blueprint, render_template, request
import io, base64, ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)

direct_plot_bp = Blueprint("direct_plot", __name__, template_folder="../templates")


# reusable parser setup
_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,   # lets "(s+3)(s+1)" work
    convert_xor                            # lets "^" mean exponent
)

def _str_to_coeffs(txt: str):
    txt = txt.strip()
    if txt.startswith("["):                      # Python list, e.g. [1, 0.5, 0]
        return np.asarray(ast.literal_eval(txt), dtype=float)

    # factorised form, e.g. (s+3)(s+1)^2  or  0.5(s+2)
    expr = parse_expr(
        txt,
        local_dict={"s": sp.symbols("s", complex=True)},
        transformations=_TRANSFORMS,
        evaluate=False        # keep it symbolic until we expand below
    )
    coeffs = sp.Poly(sp.expand(expr), sp.symbols("s", complex=True)).all_coeffs()
    return np.asarray(coeffs, dtype=float)


def _normalise(num, den):
    """force leading a₀ = 1 and return float coeff arrays."""
    den = np.asarray(den, dtype=float)
    num = np.asarray(num, dtype=float)
    num /= den[0]
    den /= den[0]
    return num, den

def _neg_fmt(coef: float) -> str:
    """Format -coef without redundant minus signs."""
    return f"{-coef:g}"

def _fmt_inv(a0: float) -> str:
    """Return latex  for 1/a0 with clean signs."""
    if np.isclose(a0, 1.0):   return "1"
    if np.isclose(a0, -1.0):  return "-1"
    val = abs(float(a0))
    # integers get an integer denominator, others use g-format
    if np.isclose(val, round(val)):
        d = int(round(val))
        return rf"-\frac{{1}}{{{d}}}" if a0 < 0 else rf"\frac{{1}}{{{d}}}"
    return rf"-\frac{{1}}{{{val:g}}}" if a0 < 0 else rf"\frac{{1}}{{{val:g}}}"


def _as_int_or_float(x):
    xi = int(round(float(x)))
    return xi if np.isclose(x, xi) else float(x)


def _coeffs_to_poly_expr(coeffs, s):
    """
    Baue eine SymPy-Polynomausdruck termweise,
    so dass ganze Zahlen als sp.Integer und nicht-ganze
    Zahlen als sp.Float erscheinen.
    """
    expr = 0
    n = len(coeffs)
    for k, c in enumerate(coeffs):
        power = n - k - 1
        if np.isclose(c, round(c)):
            coeff_sym = sp.Integer(int(round(c)))
        else:
            coeff_sym = sp.Float(c)
        expr += coeff_sym * s ** power
    return expr


# ─────────────────── diagram drawing ───────────────────────────────────────
def _circle(ax, xy, r=0.17):
    """Draw an adder node (circle with a plus sign)."""
    ax.add_patch(Circle(xy, r, fill=False, lw=1.4, zorder=3))
    ax.text(xy[0], xy[1], "+", ha="center", va="center", fontsize=12, zorder=3)

def _box(ax, xy, w=0.6, h=0.35, text=""):
    x, y = xy
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False, lw=1.4, zorder=2))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=10, zorder=2)


def _dot(ax, xy):
    ax.add_patch(Circle(xy, 0.04, color="k", zorder=3))


def _arrow(ax, src, dst):
    ax.annotate(
        "",
        xy=dst,
        xytext=src,
        arrowprops=dict(arrowstyle="->", lw=1.2, shrinkA=1, shrinkB=1),
        zorder=1,

    )


def _draw_df2(ax, b: np.ndarray, a: np.ndarray):
    """
    Draw the *order-2* analog Direct-Form II structure.

    Block co-ordinates are hard-wired → rock-solid geometry.
    """
    # ── lay-out constants ────────────────────────────────────────────────
    X_L, X_GL, X_INT, X_GR, X_Y = 0.0, 1.3, 2.6, 4.5, 6.0
    Y_TOP, DY = 2.0, 1.0
    r_add = 0.17

    # ── adders on the left (Σ) ───────────────────────────────────────────
    adders = [(X_L, Y_TOP - i * DY) for i in range(3)]  # Σ₀, Σ₁, Σ₂
    for xy in adders:
        _circle(ax, xy, r_add)
    ax.text(X_L - 0.6, Y_TOP + 0.15, r"$x(t)$", ha="left", fontsize=11)
    _arrow(ax, (X_L - 0.45, Y_TOP), (X_L - r_add, Y_TOP))

    # vertical spine (adder outputs)
    _arrow(ax, (X_L, Y_TOP - r_add), (X_L, Y_TOP - DY + r_add))
    _arrow(ax, (X_L, Y_TOP - DY - r_add), (X_L, Y_TOP - 2 * DY + r_add))

    # ── feedback gains (–a₁, –a₂) – draw only if they exist ─────────
    fb_boxes = [(X_GL, Y_TOP - DY), (X_GL, Y_TOP - 2 * DY)]
    for k, (x, y) in enumerate(fb_boxes, start=1):
        if k < len(a):                                   # bounds check
            _box(ax, (x, y), text=rf"${_neg_fmt(a[k])}$")
            _arrow(ax, (X_INT, y), (x + 0.3, y))         # from state → gain
            _arrow(ax, (x - 0.3, y), (X_L + r_add, y))   # from gain → adder

    # ── feed-forward gain b₀ ────────────────────────────────────────────
    _box(ax, (X_GL, Y_TOP), text=rf"${b[0]:g}$")
    _arrow(ax, (X_L + r_add, Y_TOP), (X_GL - 0.3, Y_TOP))
    _arrow(ax, (X_GL + 0.3, Y_TOP), (X_INT, Y_TOP))

    # ── integrators and state nodes ─────────────────────────────────────
    int_boxes = [(X_INT, Y_TOP - 0.5 * DY), (X_INT, Y_TOP - 1.5 * DY)]
    state_nodes = [(X_INT, Y_TOP - DY), (X_INT, Y_TOP - 2 * DY)]

    # first vertical drop from node 0 to INT₁
    _dot(ax, (X_INT, Y_TOP))
    for i, (bx, by) in enumerate(int_boxes):
        _box(ax, (bx, by), text=r"$\int$")
        _dot(ax, state_nodes[i])
        # arrows: down into integrator, then out to next node
        _arrow(ax, (X_INT, Y_TOP - i * DY), (X_INT, by + 0.18))
        _arrow(ax, (X_INT, by - 0.18), state_nodes[i])

    # (second feedback-gain loop removed – handled above)



    # ── feed-forward gains b₁, b₂ (optional) ─────────
    ff_specs = [(1, Y_TOP - DY), (2, Y_TOP - 2 * DY)]
    for idx, y in ff_specs:
        if idx < len(b):
            _box(ax, (X_GR, y), text=rf"${b[idx]:g}$")
            _arrow(ax, (X_INT, y), (X_GR - 0.3, y))
            _arrow(ax, (X_GR + 0.3, y), (X_Y - r_add, y))

    # connection from node 0 straight to first output adder
    _arrow(ax, (X_INT, Y_TOP), (X_Y - r_add, Y_TOP))

    # ── output adder chain and 1/a₀ gain ────────────────────────────────
    out_adders = [(X_Y, Y_TOP - i * DY) for i in range(3)]
    for xy in out_adders:
        _circle(ax, xy, r_add)
    _arrow(ax, (X_Y, Y_TOP - r_add), (X_Y, Y_TOP - DY + r_add))
    _arrow(ax, (X_Y, Y_TOP - DY - r_add), (X_Y, Y_TOP - 2 * DY + r_add))

    _box(ax, (X_Y + 0.9, Y_TOP), text=f"$\\frac{{1}}{{{a[0]:g}}}$")
    _arrow(ax, (X_Y + r_add, Y_TOP), (X_Y + 0.9 - 0.3, Y_TOP))
    _arrow(ax, (X_Y + 0.9 + 0.3, Y_TOP), (X_Y + 1.7, Y_TOP))
    ax.text(X_Y + 1.8, Y_TOP + 0.15, r"$y(t)$", ha="left", fontsize=11)


    # ── cosmetics ───────────────────────────────────────────────────────
    ax.set_xlim(-0.8, X_Y + 2.0)
    ax.set_ylim(-0.4, Y_TOP + 0.6)
    ax.axis("off")


def _draw_df1(ax, b: np.ndarray, a: np.ndarray):
    """Direct‑Form I (order ≤2), zero‑tap skipping, no overlaps, clean 1/a0."""
    X_L, X_X, X_S, X_A, X_ONE, X_Y = 0.0, 1.8, 3.4, 4.9, 6.1, 7.4
    Y0, DY = 2.0, 1.2
    r = 0.17

    def _nz(x):  # non‑zero?
        return not np.isclose(x, 0.0)

    # labels
    ax.text(X_L - 0.6, Y0 + 0.15, r"$x(t)$", ha="left", fontsize=11)
    ax.text(X_Y + 0.7, Y0 + 0.15, r"$y(t)$", ha="left", fontsize=11)

    # input
    _arrow(ax, (X_L - 0.45, Y0), (X_L + r, Y0))
    _dot(ax, (X_L + r, Y0))

    # ── three stacked adders: Σ0 (top, main), Σ1 (mid), Σ2 (bottom)
    sig = [(X_S, Y0 - i*DY) for i in range(3)]
    for xy in sig:
        _circle(ax, xy, r)
    # chain Σ2→Σ1→Σ0 vertically
    _arrow(ax, (X_S, Y0 - 2*DY + r), (X_S, Y0 - DY - r))
    _arrow(ax, (X_S, Y0 - DY + r),  (X_S, Y0 - r))

    # ── 1/a0 block to the RIGHT of Σ0 (so feedback lines never cross it)
    _box(ax, (X_ONE, Y0), text=f"${_fmt_inv(a[0])}$")
    _arrow(ax, (X_S + r, Y0), (X_ONE - 0.3, Y0))

    # output adder and y(t)
    _circle(ax, (X_Y, Y0), r)
    _arrow(ax, (X_ONE + 0.3, Y0), (X_Y - r, Y0))
    _arrow(ax, (X_Y + r, Y0), (X_Y + 0.6, Y0))

    # ── FEED‑FORWARD (x‑chain): b0→Σ0, b1→Σ1, b2→Σ2
    # b0
    if _nz(b[0]):
        _box(ax, (X_S - 1.0, Y0), text=rf"${b[0]:g}$")
        _arrow(ax, (X_L + r, Y0), (X_S - 1.0 - 0.3, Y0))
        _arrow(ax, (X_S - 1.0 + 0.3, Y0), (X_S - r, Y0))
    else:
        # still route input to the x‑chain even if b0=0
        _arrow(ax, (X_L + r, Y0), (X_X, Y0))

    # x‑integrators (produce states for b1,b2)
    _arrow(ax, (X_L + r, Y0), (X_X, Y0))
    x_int = [(X_X, Y0 - 0.5*DY), (X_X, Y0 - 1.5*DY)]
    x_state = [(X_X, Y0 - DY), (X_X, Y0 - 2*DY)]
    for i, (bx, by) in enumerate(x_int):
        _box(ax, (bx, by), text=r"$\int$")
        _dot(ax, x_state[i])
        _arrow(ax, (bx, Y0 - i*DY), (bx, by + 0.18))
        _arrow(ax, (bx, by - 0.18), x_state[i])

    # b1 → Σ1, b2 → Σ2
    for idx in (1, 2):
        if idx < len(b) and _nz(b[idx]):
            y_t = Y0 - idx*DY
            _box(ax, (X_S - 1.0, y_t), text=rf"${b[idx]:g}$")
            _arrow(ax, x_state[idx-1], (X_S - 1.0 - 0.3, y_t))
            _arrow(ax, (X_S - 1.0 + 0.3, y_t), (X_S - r, y_t))

    # ── FEEDBACK (y‑chain): −a1→Σ1, −a2→Σ2 (never into Σ0)
    _arrow(ax, (X_S + r, Y0), (X_A, Y0))  # tap Σ0 into y‑chain
    _dot(ax, (X_A, Y0))
    y_int = [(X_A, Y0 - 0.5*DY), (X_A, Y0 - 1.5*DY)]
    y_state = [(X_A, Y0 - DY), (X_A, Y0 - 2*DY)]
    for i, (bx, by) in enumerate(y_int):
        _box(ax, (bx, by), text=r"$\int$")
        _dot(ax, y_state[i])
        _arrow(ax, (bx, Y0 - i*DY), (bx, by + 0.18))
        _arrow(ax, (bx, by - 0.18), y_state[i])

    for idx in (1, 2):
        if idx < len(a) and _nz(a[idx]):
            y_t = Y0 - idx*DY
            _box(ax, (X_ONE - 0.8, y_t), text=rf"${_neg_fmt(a[idx])}$")
            _arrow(ax, y_state[idx-1], (X_ONE - 0.8 - 0.3, y_t))  # state → gain
            _arrow(ax, (X_ONE - 0.8 + 0.3, y_t), (X_S - r, y_t))  # gain → Σ(level)

    # canvas
    ax.set_xlim(-0.8, X_Y + 1.2)
    ax.set_ylim(-0.6, Y0 + 0.6)
    ax.axis("off")





def _draw_df3(ax, b: np.ndarray, a: np.ndarray):
    """Transposed DF‑II (order ≤2) with fully connected left spine."""
    X_L, X_GL, X_INT, X_GR, X_Y = 0.0, 1.3, 2.6, 4.5, 6.0
    Y0, DY = 2.0, 1.0
    r = 0.17

    # input adder and label
    _circle(ax, (X_L, Y0), r)
    ax.text(X_L - 0.6, Y0 + 0.15, r"$x(t)$", ha="left", fontsize=11)
    _arrow(ax, (X_L - 0.45, Y0), (X_L - r, Y0))

    # left vertical spine with junctions (so lower taps are visibly connected)
    spine_x = X_L + r
    ax.plot([spine_x, spine_x], [Y0, Y0 - 2*DY], lw=1.2, color="k", zorder=1)
    for y in (Y0, Y0 - DY, Y0 - 2*DY):
        _dot(ax, (spine_x, y))

    # feed‑forward: b0, b1, b2
    _box(ax, (X_GL, Y0), text=rf"${b[0]:g}$")
    _arrow(ax, (spine_x, Y0), (X_GL - 0.3, Y0))
    _arrow(ax, (X_GL + 0.3, Y0), (X_INT, Y0))

    for idx, y in ((1, Y0 - DY), (2, Y0 - 2*DY)):
        if idx < len(b):
            _box(ax, (X_GL, y), text=rf"${b[idx]:g}$")
            _arrow(ax, (spine_x, y), (X_GL - 0.3, y))
            _arrow(ax, (X_GL + 0.3, y), (X_INT, y))

    # integrator chain and state nodes
    ints = [(X_INT, Y0 - 0.5*DY), (X_INT, Y0 - 1.5*DY)]
    states = [(X_INT, Y0 - DY),    (X_INT, Y0 - 2*DY)]
    _dot(ax, (X_INT, Y0))
    for i, (bx, by) in enumerate(ints):
        _box(ax, (bx, by), text=r"$\int$")
        _dot(ax, states[i])
        _arrow(ax, (X_INT, Y0 - i*DY), (X_INT, by + 0.18))
        _arrow(ax, (X_INT, by - 0.18), states[i])

    # feedback gains to single output adder (route neatly to its rim)
    for idx, y in ((1, Y0 - DY), (2, Y0 - 2*DY)):
        if idx < len(a):
            _box(ax, (X_GR, y), text=rf"${_neg_fmt(a[idx])}$")
            _arrow(ax, states[idx-1], (X_GR - 0.3, y))
            _arrow(ax, (X_GR + 0.3, y), (X_Y - r, Y0))

    # 1/a0 path into output adder
    _box(ax, (X_GR, Y0), text=f"$\\frac{{1}}{{{a[0]:g}}}$")
    _arrow(ax, (X_INT, Y0), (X_GR - 0.3, Y0))
    _arrow(ax, (X_GR + 0.3, Y0), (X_Y - r, Y0))

    # output adder and y(t)
    _circle(ax, (X_Y, Y0), r)
    _arrow(ax, (X_Y + r, Y0), (X_Y + 0.6, Y0))
    ax.text(X_Y + 0.7, Y0 + 0.15, r"$y(t)$", ha="left", fontsize=11)

    ax.set_xlim(-0.8, X_Y + 1.3)
    ax.set_ylim(-0.4, Y0 + 0.6)
    ax.axis("off")




def _make_diagram(num, den, form: str):
    """Return base64 PNG of the chosen direct form diagram."""
    order = len(den) - 1
    if order > 2:
        return None  # fall back to plain text table

    fig, ax = plt.subplots(figsize=(7, 3))
    
    if form == "1":
        _draw_df1(ax, num, den)
    elif form == "3":
        _draw_df3(ax, num, den)
    else:
        _draw_df2(ax, num, den)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─────────────────── Flask route ───────────────────────────────────────────
@direct_plot_bp.route("/", methods=["GET", "POST"])
def direct_plot():
    form = request.form
    num_txt = form.get("numerator", "[-1, 8, 14]")   # demo defaults = your example
    den_txt = form.get("denominator", "[-1, 6, -10]")
    form_sel = form.get("direct_form", "2")

    diagram64 = table_txt = tf_ltx = error = None
    if request.method == "POST":
        try:
            num = _str_to_coeffs(num_txt)
            den = _str_to_coeffs(den_txt)

            # diagram or fallback
            diagram64 = _make_diagram(num, den, form_sel)
            if diagram64 is None:
                table_txt = (
                    "Order > 2 – diagram omitted.<br>"
                    f"<b>b:</b> {np.round(num,3)}<br>"
                    f"<b>a:</b> {np.round(den,3)}"
                )

            # LaTeX pretty-print.  If a coefficient is very close to an integer,
            # display it as an integer to avoid "1.0" style output.
            s = sp.symbols("s", complex=True)

            # SymPy expects highest-power first; our arrays are lowest-power first
            num_sym = _coeffs_to_poly_expr(num, s)
            den_sym = _coeffs_to_poly_expr(den, s)

            tf_ltx = r"\displaystyle H(s)=\frac{%s}{%s}" % (
                sp.latex(num_sym),
                sp.latex(den_sym),
            )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    return render_template(
        "direct_plot.html",
        default_num=num_txt,
        default_den=den_txt,
        selected_form=form_sel,   
        tf_latex=tf_ltx,
        diagram_url=diagram64,
        table_fallback=table_txt,
        error=error,
    )
