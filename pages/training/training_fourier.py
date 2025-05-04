from flask import Blueprint, render_template, request, jsonify
import io, base64, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

training_fourier_bp = Blueprint("training_fourier", __name__)

@training_fourier_bp.route("/")
def training_fourier():
    return render_template("training_fourier.html")

@training_fourier_bp.route("/generate", methods=["POST"])
def generate_problem():
    data = request.get_json(force=True)
    difficulty = data.get("difficulty", "EASY").upper()
    direction = data.get("direction", "TIME_TO_FREQ").upper()
    return jsonify(create_fourier_problem(difficulty, direction))

@training_fourier_bp.route("/check_answer", methods=["POST"])
def check_answer():
    data = request.get_json(force=True)
    feedback = "Correct!" if data.get("selectedIndex") == data.get("correctIndex") else "Incorrect. Try again!"
    return jsonify({"feedback": feedback})


def create_fourier_problem(difficulty, direction):
    # random parameters
    w0 = random.randint(1, 3)
    shift = random.randint(-3, 3)
    scale = random.randint(1, 3)
    width = random.randint(1, 3)

    # function pool definitions
    def f_rect(t): return np.where(np.abs(t) < 0.5, 1, 0)
    def f_tri(t):  return np.maximum(1 - np.abs(t), 0)
    def f_sinc(t): return np.sinc(t)
    def f_sinc2(t): return np.sinc(t)**2
    def f_inv_t(t): return np.where(t != 0, 1/t, 0)
    def f_sign(t): return np.sign(t)
    def f_cexp(t): return np.exp(1j * w0 * t)
    def f_cos(t): return np.cos(w0 * t)
    def f_sin(t): return np.sin(w0 * t)

    pool = {
        "rect(t)": (f_rect, "Rectangular"),
        "tri(t)":  (f_tri,  "Triangular"),
        "sinc(t)":(f_sinc, "Sinc"),
        "sinc^2(t)":(f_sinc2, "Sinc^2"),
        "1/t":   (f_inv_t, "1/t"),
        "sign(t)":(f_sign, "Signum"),
        f"exp(j{w0}t)":(f_cexp, "Complex Exp"),
        f"cos({w0}t)":(f_cos, "Cosine"),
        f"sin({w0}t)":(f_sin, "Sine")
    }

    name, (func, desc) = random.choice(list(pool.items()))

    # evaluation grid
    t = np.linspace(-10, 10, 512)
    omega = np.linspace(-10, 10, 512)

    # time-domain signal
    sig = scale * func((t - shift) / width)

    # analytic transform functions
    def T_rect(w):
        x = w/2
        return np.where(w==0, 1.0, np.sin(x)/x)
    def T_tri(w):
        x = w/2
        return (np.sin(x)/x)**2
    def T_sinc(w):
        return np.where(np.abs(w) <= np.pi, 1.0, 0.0)
    def T_sinc2(w):
        return np.where(np.abs(w) <= 2*np.pi, np.pi*(1 - np.abs(w)/(2*np.pi)), 0.0)
    def T_inv_t(w):
        return -1j*np.pi*np.sign(w)
    def T_sign(w):
        return -2j/w
    def T_cexp(w):
        # delta approximations
        delta = 0.1
        return 2*np.pi * np.exp(-(w - w0)**2/(2*delta**2))/(delta*np.sqrt(2*np.pi))
    def T_cos(w):
        delta=0.1
        return np.pi*(np.exp(-(w-w0)**2/(2*delta**2))/(delta*np.sqrt(2*np.pi)) + 
                      np.exp(-(w+w0)**2/(2*delta**2))/(delta*np.sqrt(2*np.pi)))
    def T_sin(w):
        delta=0.1
        return (-1j*np.pi*np.exp(-(w-w0)**2/(2*delta**2))/(delta*np.sqrt(2*np.pi)) + 
                 1j*np.pi*np.exp(-(w+w0)**2/(2*delta**2))/(delta*np.sqrt(2*np.pi)))

    X_funcs = {
        "rect(t)": T_rect,
        "tri(t)":  T_tri,
        "sinc(t)":T_sinc,
        "sinc^2(t)":T_sinc2,
        "1/t":   T_inv_t,
        "sign(t)":T_sign,
        f"exp(j{w0}t)":T_cexp,
        f"cos({w0}t)":T_cos,
        f"sin({w0}t)":T_sin
    }

    # correct transform (with scaling & shift)
    X = scale * width * X_funcs[name](omega*width) * np.exp(-1j*omega*shift)

    # distractor transforms
    def make_X(wd, sc, sh):
        return sc * wd * X_funcs[name](omega*wd) * np.exp(-1j*omega*sh)
    X1 = make_X(width+1, scale, shift)
    X2 = make_X(width, scale+1, shift)
    X3 = make_X(width, scale, shift+1)

    options = [X, X1, X2, X3]
    idxs = list(range(4)); random.shuffle(idxs)
    shuffled = [options[i] for i in idxs]
    correct_idx = idxs.index(0)

    # plotting
    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(3,2, height_ratios=[1,1,1])
    ax0 = fig.add_subplot(gs[0,:])
    ax0.plot(t, sig.real)
    ax0.set_title(f"{desc} (shift={shift}, scale={scale}, width={width})")
    ax0.grid(True)
    ax0.set_xlim(-10,10)

    for i, Xopt in enumerate(shuffled):
        ax = fig.add_subplot(gs[1 + i//2, i%2])
        ax.plot(omega, np.abs(Xopt), label='|X|')
        ax.plot(omega, np.angle(Xopt), linestyle='--', label='∠X')
        ax.plot(omega, np.imag(Xopt), linestyle=':', label='Im{X}')
        ax.set_title(f"Option {i+1}")
        ax.grid(True)
        ax.set_xlim(-10,10)
        ax.legend()

    fig.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf,format='png'); buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode(); plt.close(fig)

    return {"plot_data": plot_data, "correctIndex": correct_idx}
