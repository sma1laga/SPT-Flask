# pages/fourier_page.py
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Blueprint, render_template, request

fourier_bp = Blueprint("fourier", __name__)

@fourier_bp.route("/", methods=["GET", "POST"])
def fourier():
    if request.method == "POST":
        func_str = request.form.get("func", "")
        try:
            t = np.linspace(-10, 10, 400)
            # Placeholder logic
            y = np.sin(t) if not func_str else np.sin(t*5)
            # Real logic would do FFT etc.

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))
            ax1.plot(t, y)
            ax1.set_title("Time Domain")

            # Placeholder “magnitude spectrum”
            Yf = np.fft.fft(y)
            freq = np.fft.fftfreq(len(t), d=(t[1]-t[0]))
            ax2.plot(freq, np.abs(Yf))
            ax2.set_title("Magnitude Spectrum")

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

            return render_template("fourier.html",
                                   func=func_str,
                                   plot_data=plot_data)
        except Exception as e:
            return render_template("fourier.html",
                                   func=func_str,
                                   error=str(e))
    else:
        return render_template("fourier.html")
