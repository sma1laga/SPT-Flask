# utils/img.py
import io, base64
import matplotlib.pyplot as plt

def fig_to_base64(fig, close=True, dpi=110):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")
