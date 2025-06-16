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

direct_plot_bp = Blueprint("direct_plot", __name__, template_folder="../templates")


# ─────────────────── helpers ────────────────────────────────────────────────
def _str_to_coeffs(txt: str):
    txt = txt.strip()
    if txt.startswith("["):                            # Python list
        return np.asarray(ast.literal_eval(txt), dtype=float)
    # factorised polynomial, e.g. (s+3)(s+1)
    s = sp.symbols("s")
    coeffs = sp.Poly(sp.expand(txt), s).all_coeffs()
    return np.asarray(coeffs, dtype=float)


def _normalise(num, den):
    """force leading a₀ = 1 and return float coeff arrays."""
    den = np.asarray(den, dtype=float)
    num = np.asarray(num, dtype=float)
    num /= den[0]
    den /= den[0]
    return num, den


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

    # ── gain blocks left (feedback –a₁, –a₂) ─────────────────────────────
    fb_boxes = [(X_GL, Y_TOP - DY), (X_GL, Y_TOP - 2 * DY)]
    for k, (x, y) in enumerate(fb_boxes, start=1):
        _box(ax, (x, y), text=rf"$-{a[k]:g}$")
        _arrow(ax, (X_INT, y), (x + 0.3, y))
        _arrow(ax, (x - 0.3, y), (X_L + r_add, y))

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

    # --- feedback gains -------------------------------------------------
    for k, (x, y) in enumerate(fb_boxes, start=1):
        if k < len(a):                 # only draw if a_k exists
            _box(ax, (x, y), text=rf"$-{a[k]:g}$")
            _arrow(ax, (X_INT, y), (x + 0.3, y))
            _arrow(ax, (x - 0.3, y), (X_L + r_add, y))

    # --- feed-forward gains --------------------------------------------
    ff_specs = [(1, Y_TOP - DY), (2, Y_TOP - 2*DY)]
    for idx, y in ff_specs:
        if idx < len(b):
            _box(ax, (X_GR, y), text=rf"${b[idx]:g}$")
            _arrow(ax, (X_INT, y), (X_GR - 0.3, y))
            _arrow(ax, (X_GR + 0.3, y), (X_Y - r_add, Y_TOP))

    # connection from node 0 straight to output adder
    _arrow(ax, (X_INT, Y_TOP), (X_Y - r_add, Y_TOP))

    # ── output adder and label ──────────────────────────────────────────
    _circle(ax, (X_Y, Y_TOP), r_add)
    _arrow(ax, (X_Y + r_add, Y_TOP), (X_Y + 0.6, Y_TOP))
    ax.text(X_Y + 0.7, Y_TOP + 0.15, r"$y(t)$", ha="left", fontsize=11)

    # ── cosmetics ───────────────────────────────────────────────────────
    ax.set_xlim(-0.8, X_Y + 1.3)
    ax.set_ylim(-0.4, Y_TOP + 0.6)
    ax.axis("off")


def _draw_df1(ax, b: np.ndarray, a: np.ndarray):
    """Draw Direct-Form I with two integrator chains (order ≤2)."""
    X_L, X_XINT, X_ADD, X_YINT, X_Y = 0.0, 1.8, 3.8, 5.6, 7.0
    Y_TOP, DY = 2.0, 1.2
    r_add = 0.17

    # output adder
    _circle(ax, (X_ADD, Y_TOP), r_add)
    _arrow(ax, (X_ADD + r_add, Y_TOP), (X_Y, Y_TOP))
    ax.text(X_Y + 0.2, Y_TOP + 0.15, r"$y(t)$", ha="left", fontsize=11)

    # input and branching
    ax.text(X_L - 0.6, Y_TOP + 0.15, r"$x(t)$", ha="left", fontsize=11)
    _arrow(ax, (X_L - 0.45, Y_TOP), (X_L + r_add, Y_TOP))
    _dot(ax, (X_L + r_add, Y_TOP))

    # b0 gain directly to adder
    _box(ax, (X_ADD - 1.0, Y_TOP), text=rf"${b[0]:g}$")
    _arrow(ax, (X_L + r_add, Y_TOP), (X_ADD - 1.0 - 0.3, Y_TOP))
    _arrow(ax, (X_ADD - 1.0 + 0.3, Y_TOP), (X_ADD - r_add, Y_TOP))

    # --- X integrator chain and b1,b2 gains ---
    int_x = [(X_XINT, Y_TOP - 0.5*DY), (X_XINT, Y_TOP - 1.5*DY)]
    states_x = [(X_XINT, Y_TOP - DY), (X_XINT, Y_TOP - 2*DY)]
    _arrow(ax, (X_L + r_add, Y_TOP), (X_XINT, Y_TOP))
    for i, (bx, by) in enumerate(int_x):
        _box(ax, (bx, by), text=r"$\int$")
        _dot(ax, states_x[i])
        _arrow(ax, (bx, Y_TOP - i*DY), (bx, by + 0.18))
        _arrow(ax, (bx, by - 0.18), states_x[i])
        idx = i + 1
        if idx < len(b):
            _box(ax, (X_ADD - 1.0, by), text=rf"${b[idx]:g}$")
            _arrow(ax, states_x[i], (X_ADD - 1.0 - 0.3, by))
            _arrow(ax, (X_ADD - 1.0 + 0.3, by), (X_ADD - r_add, Y_TOP))

    # --- Y integrator chain and feedback gains ---
    _arrow(ax, (X_ADD + r_add, Y_TOP), (X_YINT, Y_TOP))
    _dot(ax, (X_YINT, Y_TOP))
    int_y = [(X_YINT, Y_TOP - 0.5*DY), (X_YINT, Y_TOP - 1.5*DY)]
    states_y = [(X_YINT, Y_TOP - DY), (X_YINT, Y_TOP - 2*DY)]
    for i, (bx, by) in enumerate(int_y):
        _box(ax, (bx, by), text=r"$\int$")
        _dot(ax, states_y[i])
        _arrow(ax, (bx, Y_TOP - i*DY), (bx, by + 0.18))
        _arrow(ax, (bx, by - 0.18), states_y[i])
        idx = i + 1
        if idx < len(a):
            _box(ax, (X_ADD + 1.0, by), text=rf"$-{a[idx]:g}$")
            _arrow(ax, states_y[i], (X_ADD + 1.0 - 0.3, by))
            _arrow(ax, (X_ADD + 1.0 + 0.3, by), (X_ADD - r_add, Y_TOP))

    ax.set_xlim(-0.8, X_Y + 1.0)
    ax.set_ylim(-0.6, Y_TOP + 0.6)
    ax.axis("off")


def _draw_df3(ax, b: np.ndarray, a: np.ndarray):
    """Approximate transposed Direct-Form II."""
    X_L, X_GL, X_INT, X_GR, X_Y = 0.0, 1.3, 2.6, 4.5, 6.0
    Y_TOP, DY = 2.0, 1.0
    r_add = 0.17

    # input adder
    _circle(ax, (X_L, Y_TOP), r_add)
    ax.text(X_L - 0.6, Y_TOP + 0.15, r"$x(t)$", ha="left", fontsize=11)
    _arrow(ax, (X_L - 0.45, Y_TOP), (X_L - r_add, Y_TOP))

    # feed-forward gains on left
    _box(ax, (X_GL, Y_TOP), text=rf"${b[-1]:g}$")
    _arrow(ax, (X_L + r_add, Y_TOP), (X_GL - 0.3, Y_TOP))
    _arrow(ax, (X_GL + 0.3, Y_TOP), (X_INT, Y_TOP))

    ff_specs = [(1, Y_TOP - DY), (2, Y_TOP - 2*DY)]
    for idx, y in ff_specs:
        if idx < len(b):
            coef = b[-(idx+1)]
            _box(ax, (X_GL, y), text=rf"${coef:g}$")
            _arrow(ax, (X_L + r_add, y), (X_GL - 0.3, y))
            _arrow(ax, (X_GL + 0.3, y), (X_INT, y))

    # integrator chain
    int_boxes = [(X_INT, Y_TOP - 0.5*DY), (X_INT, Y_TOP - 1.5*DY)]
    state_nodes = [(X_INT, Y_TOP - DY), (X_INT, Y_TOP - 2*DY)]
    _dot(ax, (X_INT, Y_TOP))
    for i, (bx, by) in enumerate(int_boxes):
        _box(ax, (bx, by), text=r"$\int$")
        _dot(ax, state_nodes[i])
        _arrow(ax, (X_INT, Y_TOP - i*DY), (X_INT, by + 0.18))
        _arrow(ax, (X_INT, by - 0.18), state_nodes[i])

    # feedback gains to output
    fb_specs = [(1, Y_TOP - DY), (2, Y_TOP - 2*DY)]
    for idx, y in fb_specs:
        if idx < len(a):
            coef = a[-(idx+1)]
            _box(ax, (X_GR, y), text=rf"$-{coef:g}$")
            _arrow(ax, state_nodes[idx-1], (X_GR - 0.3, y))
            _arrow(ax, (X_GR + 0.3, y), (X_Y - r_add, Y_TOP))

    _box(ax, (X_GR, Y_TOP), text=rf"$-{a[-1]:g}$")
    _arrow(ax, (X_INT, Y_TOP), (X_GR - 0.3, Y_TOP))
    _arrow(ax, (X_GR + 0.3, Y_TOP), (X_Y - r_add, Y_TOP))

    _circle(ax, (X_Y, Y_TOP), r_add)
    _arrow(ax, (X_Y + r_add, Y_TOP), (X_Y + 0.6, Y_TOP))
    ax.text(X_Y + 0.7, Y_TOP + 0.15, r"$y(t)$", ha="left", fontsize=11)

    ax.set_xlim(-0.8, X_Y + 1.3)
    ax.set_ylim(-0.4, Y_TOP + 0.6)
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
            num_orig = _str_to_coeffs(num_txt)
            den_orig = _str_to_coeffs(den_txt)
            num, den = _normalise(num_orig, den_orig)

            # diagram or fallback
            diagram64 = _make_diagram(num, den, form_sel)
            if diagram64 is None:
                table_txt = (
                    "Order > 2 – diagram omitted.<br>"
                    f"<b>b:</b> {np.round(num_orig,3)}<br>"
                    f"<b>a:</b> {np.round(den_orig,3)}"
                )

            # LaTeX pretty-print.  If a coefficient is very close to an integer,
            # display it as an integer to avoid "1.0" style output.
            s = sp.symbols("s")

            def _as_int_or_float(x):
                xi = int(round(float(x)))
                return xi if np.isclose(x, xi) else float(x)

            num_sym = sp.Poly([_as_int_or_float(c) for c in num], s).as_expr()
            den_sym = sp.Poly([_as_int_or_float(c) for c in den], s).as_expr()

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
