# pages/function_definitions.py
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import Blueprint, render_template
from matplotlib.ticker import MaxNLocator

func_defs_bp = Blueprint("function_definitions", __name__, url_prefix="/function_definitions")

def make_plot(x, y, kind="line"):
    """Render x,y to a PNG and return it base64-encoded."""
    fig, ax = plt.subplots(figsize=(3,2), layout="constrained")
    if kind == "comb":
        # draw as true Dirac impulses
        ax.vlines(x, 0, y, colors="C0", linewidth=1)
        ax.plot(x, y, "C0o", markersize=4)
    elif kind == "stem":
        ax.stem(x, y, linefmt="C0-", markerfmt="C0o", basefmt=" ")
    else:
        ax.plot(x, y, linewidth=1)
        # show grid and axes
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # move left and bottom spines to zero
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    # hide the other two spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ticks on both sides
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
# make y-axis ticks “discrete” (max ~5 integer ticks)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    # optional: tighten limits a bit
    ax.set_xlim(x.min(), x.max())
    ypad = abs(y).max() * 0.1
    if kind == "comb":
        ymin = 0
    else:
        ymin = y.min()
    ax.set_ylim(ymin - ypad, y.max() + ypad)
    
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

@func_defs_bp.route("/")
def function_definitions():
    # Continuous-time signals
    continuous = [
        dict(
            name="Unit Step Function",
            definition=r"u(t)=\begin{cases}0,&t<0\\1,&t\ge0\end{cases}",
            python="lambda t: np.where(t>=0,1,0)",
            t=np.linspace(-1,1,400),
            func=lambda t: np.where(t>=0,1,0),
            kind="line"
        ),
        dict(
            name="Unit Impulse Function",
            definition=r"\delta(t)=\begin{cases}\infty,&t=0\\0,&t\ne0\end{cases},\ \int\delta(t)dt=1",
            python="lambda t: np.where(t==0,1,0)",
            # ensure 0 is exactly in t
            t=np.linspace(-1, 1, 401),
            func=lambda t: np.where(np.isclose(t,0),1,0),
            kind="stem"
        ),
        dict(
            name="Ramp Function",
            definition=r"r(t)=t\,u(t)",
            python="lambda t: t*(t>=0)",
            t=np.linspace(-1,2,400),
            func=lambda t: t*(t>=0),
            kind="line"
        ),
        dict(
            name="Exponential Function",
            definition=r"x(t)=e^{at}u(t)",
            python="lambda t, a=1: np.exp(a*t)*(t>=0)",
            t=np.linspace(-1,2,400),
            func=lambda t: np.exp(1*t)*(t>=0),
            kind="line"
        ),
        dict(
            name="Cosine Function",
            definition=r"x(t)=\cos(\omega t)",
            python="lambda t, ω=2*np.pi: np.cos(ω*t)",
            t=np.linspace(0,1,400),
            func=lambda t: np.cos(2*np.pi*t),
            kind="line"
        ),
        dict(
            name="Rectangular Pulse",
            definition=r"\mathrm{rect}\!\bigl(\tfrac{t}{T}\bigr)=\begin{cases}1,&|t|<T/2\\0,&|t|>T/2\end{cases}",
            python="lambda t, T=1: np.where(np.abs(t)<T/2,1,0)",
            t=np.linspace(-1,1,400),
            func=lambda t: np.where(np.abs(t)<0.5,1,0),
            kind="line"
        ),
        dict(
            name="Triangular Pulse",
            definition=r"\mathrm{tri}\!\bigl(\tfrac{t}{T}\bigr)=\max\bigl(1-\tfrac{|t|}{T},0\bigr)",
            python="lambda t, T=1: np.maximum(1-np.abs(t)/T,0)",
            t=np.linspace(-1.5,1.5,400),
            func=lambda t: np.maximum(1-np.abs(t),0),
            kind="line"
        ),
        dict(
            name="Sinc Function",
            definition=r"\mathrm{sinc}(t)=\frac{\sin(\pi t)}{\pi t}",
            python="lambda t: np.sinc(t)",
            t=np.linspace(-5,5,400),
            func=lambda t: np.sinc(t),
            kind="line"
        ),
        dict(
            name="Gaussian Pulse",
            definition=r"g(t)=e^{-t^2/(2\sigma^2)}",
            python="lambda t, σ=1: np.exp(-t**2/(2*σ**2))",
            t=np.linspace(-3,3,400),
            func=lambda t: np.exp(-t**2/2),
            kind="line"
        ),
        dict(
            name="Dirac Comb",
            definition=r"\mathrm{comb}_T(t)=\sum_{k=-3}^{3}\delta(t - kT)",
            python="lambda k, T=1: impulses at t=k·T for k in [-3..3]",
            t=np.arange(-3,4),             # k = -3..3
            func=lambda t: np.ones_like(t), # all impulses = 1
            kind="comb"
        ),
        dict(
            name="Signum Function",
            definition=r"\mathrm{sgn}(t)=\begin{cases}-1,&t<0\\0,&t=0\\1,&t>0\end{cases}",
            python="lambda t: np.sign(t)",
            t=np.linspace(-1,1,400),
            func=lambda t: np.sign(t),
            kind="line"
        ),
        dict(
            name="Sawtooth Wave",
            definition=r"\mathrm{saw}(t)=2\bigl(\tfrac{t}{T}-\lfloor\tfrac{t}{T}+1/2\rfloor\bigr)",
            python="lambda t, T=1: 2*(t/T-np.floor(t/T+0.5))",
            t=np.linspace(0,2,400),
            func=lambda t: 2*(t-np.floor(t+0.5)),
            kind="line"
        ),
    ]

    # Discrete-time signals
    n = np.arange(-5, 16)
    discrete = [
        dict(
            name="Unit Step Sequence",
            definition=r"u[k]=\begin{cases}1,&k\ge0\\0,&k<0\end{cases}",
            python="lambda n: np.where(n>=0,1,0)",
            n=n, func=lambda n: np.where(n>=0,1,0), kind="stem"
        ),
        dict(
            name="Unit Impulse Sequence",
            definition=r"\delta[k]=\begin{cases}1,&k=0\\0,&k\ne0\end{cases}",
            python="lambda n: np.where(n==0,1,0)",
            n=n, func=lambda n: np.where(n==0,1,0), kind="stem"
        ),
        dict(
            name="Ramp Sequence",
            definition=r"r[k]=k\,u[k]",
            python="lambda n: n*(n>=0)",
            n=n, func=lambda n: n*(n>=0), kind="stem"
        ),
        dict(
            name="Exponential Sequence",
            definition=r"x[k]=a^k\,u[k]",
            python="lambda n, a=0.9: (a**n)*(n>=0)",
            n=n, func=lambda n: (0.9**n)*(n>=0), kind="stem"
        ),
        dict(
            name="Cosine Sequence",
            definition=r"x[k]=\cos(\omega k)",
            python="lambda n: np.cos(2*np.pi*n/10)",
            n=n, func=lambda n: np.cos(2*np.pi*n/10), kind="stem"
        ),
        dict(
            name="Rectangular Window",
            definition=r"w[k]=\begin{cases}1,&0\le k<N\\0,&\text{else}\end{cases}",
            python="lambda n, N=8: np.where((n>=0)&(n<8),1,0)",
            n=n, func=lambda n: np.where((n>=0)&(n<8),1,0), kind="stem"
        ),
        dict(
            name="Triangular Window",
            definition=r"w[k]=\max\bigl(1-\tfrac{|k|}{N},0\bigr)",
            python="lambda n, N=4: np.maximum(1-np.abs(n)/4,0)",
            n=n, func=lambda n: np.maximum(1-np.abs(n)/4,0), kind="stem"
        ),
        dict(
            name="Sinc Sequence",
            definition=r"\mathrm{sinc}[k]=\frac{\sin(\pi k)}{\pi k}",
            python="lambda n: np.sinc(n)",
            n=n, func=lambda n: np.sinc(n), kind="stem"
        ),
        dict(
            name="Gaussian Sequence",
            definition=r"g[k]=e^{-k^2/(2\sigma^2)}",
            python="lambda n: np.exp(-n**2/2)",
            n=n, func=lambda n: np.exp(-n**2/2), kind="stem"
        ),
        dict(
            name="Comb Sequence",
            definition=r"\mathrm{comb}_N[k]=\sum_k\delta[k-kN]",
            python="lambda n, N=4: np.where(n%4==0,1,0)",
            n=n, func=lambda n: np.where(n%4==0,1,0), kind="stem"
        ),
        dict(
            name="Signum Sequence",
            definition=r"\mathrm{sgn}[k]=\begin{cases}-1,&k<0\\0,&k=0\\1,&k>0\end{cases}",
            python="lambda n: np.sign(n)",
            n=n, func=lambda n: np.sign(n), kind="stem"
        ),
        dict(
            name="Sawtooth Sequence",
            definition=r"\mathrm{saw}[k]=2\bigl(\tfrac{k}{N}-\lfloor\tfrac{k}{N}+1/2\rfloor\bigr)",
            python="lambda n, N=8: 2*(n/8-np.floor(n/8+0.5))",
            n=n, func=lambda n: 2*(n/8-np.floor(n/8+0.5)), kind="stem"
        ),
    ]

    # Generate plots
    for sig in continuous:
        sig["plot_data"] = make_plot(sig["t"], sig["func"](sig["t"]), kind=sig["kind"])
    for sig in discrete:
        sig["plot_data"] = make_plot(sig["n"], sig["func"](sig["n"]), kind=sig["kind"])

    return render_template(
        "function_definitions.html",
        continuous_signals=continuous,
        discrete_signals=discrete
    )
