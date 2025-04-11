# pages/convolution.py
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Blueprint, render_template, request
# from scipy.signal import convolve  # import when needed

convolution_bp = Blueprint("convolution", __name__)

@convolution_bp.route("/", methods=["GET", "POST"])
def convolution():
    if request.method == "POST":
        func1 = request.form.get("func1", "")
        func2 = request.form.get("func2", "")

        try:
            t = np.linspace(-10, 10, 400)
            # Dummy logic
            y1 = np.sin(t) if not func1 else np.sin(t*2)
            y2 = np.cos(t) if not func2 else np.cos(t*2)
            # Real code:
            # y_conv = convolve(y1, y2, mode='same') * (t[1]-t[0])
            # For now, placeholder
            y_conv = y1 + y2

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
            ax1.plot(t, y1)
            ax1.set_title("Function 1")
            ax2.plot(t, y2)
            ax2.set_title("Function 2")
            ax3.plot(t, y_conv)
            ax3.set_title("Convolution")

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

            return render_template("convolution.html",
                                   func1=func1,
                                   func2=func2,
                                   plot_data=plot_data)
        except Exception as e:
            return render_template("convolution.html",
                                   func1=func1,
                                   func2=func2,
                                   error=str(e))
    else:
        return render_template("convolution.html")
