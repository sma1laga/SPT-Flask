from flask import Blueprint, render_template, request
import numpy as np, io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

discrete_convolution_bp = Blueprint(
    'discrete_convolution', __name__
)

@discrete_convolution_bp.route('/', methods=['GET','POST'])
def convolution():
    error = None
    img_data = None
    s1 = request.form.get('seq1','')
    s2 = request.form.get('seq2','')
    if request.method=='POST':
        try:
            x = [float(v) for v in s1.split(',') if v]
            h = [float(v) for v in s2.split(',') if v]
            y = np.convolve(x, h)
            n1 = np.arange(len(x))
            n2 = np.arange(len(h))
            ny = np.arange(len(y))
            fig, axes = plt.subplots(3,1, figsize=(5,6))
            for ax, data, title in zip(axes, [x,h,y], ['x[n]','h[n]','y[n]']):
                ax.stem(np.arange(len(data)), data, use_line_collection=True)
                ax.set_title(title); ax.set_xlabel("n")
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode()
            plt.close(fig)
        except Exception as e:
            error = str(e)
    return render_template(
        'discrete/discrete_convolution.html',
        error=error, img_data=img_data, seq1=s1, seq2=s2
    )
