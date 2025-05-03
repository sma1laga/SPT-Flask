from flask import Blueprint, render_template, request
import numpy as np, io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

discrete_dynamic_convolution_bp = Blueprint(
    'discrete_dynamic_convolution', __name__
)

@discrete_dynamic_convolution_bp.route('/', methods=['GET','POST'])
def dynamic_convolution():
    error = None
    img_data = None
    xs = request.form.get('x','')
    hs = request.form.get('h','')
    L  = request.form.get('L','')
    if request.method=='POST':
        try:
            x = [float(v) for v in xs.split(',') if v]
            h = [float(v) for v in hs.split(',') if v]
            # simple dynamic: y[n] = sum_k x[n-k]*h[k]
            y = np.convolve(x, h)
            n = np.arange(len(y))
            fig, ax = plt.subplots()
            ax.stem(n, y, use_line_collection=True)
            ax.set_title(f"Dynamic Conv. L={L}")
            ax.set_xlabel("n"); ax.set_ylabel("y[n]")
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode()
            plt.close(fig)
        except Exception as e:
            error = str(e)
    return render_template(
        'discrete/discrete_dynamic_convolution.html',
        error=error, img_data=img_data, x=xs, h=hs, L=L
    )
