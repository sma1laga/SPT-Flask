from flask import Blueprint, render_template, request
import io, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.math_utils import rect, tri, step, delta, dsi

discrete_plot_functions_bp = Blueprint(
    'discrete_plot_functions',
    __name__,
    template_folder='templates/discrete'
)



@discrete_plot_functions_bp.route('/', methods=['GET', 'POST'])
def plot_functions():
    # 1) Initialize before anything else to avoid NameError
    error = None
    plot_data = None

    # ---- form fields ----
    func1_str = request.form.get("func1", "")
    func2_str = request.form.get("func2", "")
    try:
        step_val = float(request.form.get("sampling", 1.0))
        if step_val <= 0:
            raise ValueError("must be positive")
    except ValueError:
        step_val = 1.0
        error = "Sampling step must be positive."

    # Only try to build & plot on a POST with no prior error
    if request.method == "POST" and not error:
        # integer index
        # decide how far in "real time" you want to see:
        max_n = 10.0

        # compute how many integer steps that is at your current sampling:
        num_steps = int(max_n / step_val)

        # build k so that n = k * Δn will span [-max_n, +max_n]
        k = np.arange(-num_steps, num_steps + 1)
        n = k * step_val

        # 2) Bind both scaled time 'n' and raw index 'k' in the context
        ctx = {
            "n":      n,
            "k":      k,
            "np":     np,
            "rect":   rect,
            "tri":    tri,
            "step":   step,
            "delta":  delta,
            "sin":    np.sin,
            "cos":    np.cos,
            "sign":   np.sign,
            "si":     dsi,
        }

        # Evaluate function 1 (or default to zero)
        try:
            y1 = eval(func1_str, {}, ctx) if func1_str.strip() \
                 else np.zeros_like(n)
        except Exception as e:
            error = f"Function 1 error: {e}"

        # Evaluate function 2, if provided
        if not error and func2_str.strip():
            try:
                y2 = eval(func2_str, {}, ctx)
            except Exception as e:
                error = f"Function 2 error: {e}"
        else:
            y2 = None

        # If everything succeeded, draw and encode the plot
        if not error:
            fig, ax = plt.subplots()
            ax.stem(n, y1, linefmt="#003366", markerfmt="o",
                    basefmt="k-", label="F1")
            if y2 is not None:
                ax.stem(n, y2, linefmt="#336600", markerfmt="s",
                        basefmt="k-", label="F2")
            ax.set_title(f"Discrete Plot (Δn = {step_val})")
            ax.set_xlabel("n")
            ax.set_ylabel("Amplitude")
            ax.legend()
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            plt.close(fig)

    # Finally render, with error and plot_data always defined
    return render_template(
        'discrete/discrete_plot_functions.html',
        error=error,
        plot_data=plot_data,
        func1=func1_str,
        func2=func2_str,
        sampling=step_val
    )
