from flask import Blueprint, render_template, request
import numpy as np, io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

discrete_fourier_bp = Blueprint(
    'discrete_fourier', __name__
)

@discrete_fourier_bp.route('/', methods=['GET','POST'])
def fourier():
    error = None
    img_data = None
    seq = request.form.get('sequence','')
    if request.method=='POST':
        try:
            x = np.array([float(v) for v in seq.split(',')])
            X = np.fft.fft(x)
            k = np.arange(len(x))
            fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
            ax1.stem(k, np.abs(X), use_line_collection=True)
            ax1.set_ylabel("|X[k]|")
            ax2.stem(k, np.angle(X), use_line_collection=True)
            ax2.set_ylabel("∠X[k]"); ax2.set_xlabel("k")
            fig.suptitle(f"DFT of [{seq}]")
            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode()
            plt.close(fig)
        except Exception as e:
            error = str(e)
    return render_template(
        'discrete/discrete_fourier.html',
        error=error, img_data=img_data, sequence=seq
    )
