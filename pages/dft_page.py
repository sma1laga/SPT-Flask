from flask import Blueprint, render_template, request, jsonify
import io, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

discrete_fourier_bp = Blueprint("discrete_fourier", __name__)

def make_sequence(func_str, N):
    # safe math and preset functions
    safe_names = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
        'pi': math.pi, 'e': math.e, 'abs': abs, 'complex': complex,
        'rect': lambda n,N: 1.0,
        'tri': lambda n,N: (1.0 - abs((2*n)/(N-1) - 1)) if N>1 else 1.0,
        'step': lambda n,N: 1.0 if n>=0 else 0.0,
        'delta': lambda n,N: 1.0 if n==0 else 0.0,
        'sign': lambda n,N: (1.0 if n>0 else (0.0 if n==0 else -1.0))
    }
    seq = []
    for n in range(N):
        local = {'n': n, 'N': N}
        local.update(safe_names)
        try:
            val = eval(func_str, {"__builtins__": None}, local)
        except Exception as e:
            raise ValueError(f"Error at n={n}: {e}")
        seq.append(val)
    return np.array(seq, dtype=complex)

def compute_dft_plot(x, padFactor):
    N = x.size
    M = max(1, int(padFactor)) * N
    X = np.fft.fft(x, n=M)

    mag = np.abs(X)
    if mag.max() > 0:
        mag /= mag.max()
    phase = np.angle(X)
    phase[mag < 1e-6] = 0

    n_time = np.arange(N)
    k = np.arange(M)
    fig, (ax_t, ax_m, ax_p) = plt.subplots(1, 3, figsize=(14, 4))

    ax_t.stem(n_time, x.real, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax_t.set(title=f"Time-domain x[n], N={N}", xlabel="n")

    ax_m.stem(k, mag, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax_m.set(title=f"Magnitude |X[k]|, M={M}", xlabel="k")

    ax_p.stem(k, phase, linefmt='g-', markerfmt='go', basefmt='k-')
    ax_p.set(title="Phase ∠X[k]", xlabel="k", ylim=[-np.pi, np.pi])

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    return {"plot_data": img_b64, "dft_label": f"N={N}, M={M}"}

@discrete_fourier_bp.route("/", methods=["GET", "POST"])
def show_dft():
    N = 8
    func_str = "sin(2*pi*n/N)"
    padFactor = 10
    plot_data = None
    dft_label = ""

    if request.method == "POST":
        form = request.form
        try:
            N = max(1, int(form.get("N", N)))
            func_str = form.get("func", func_str).strip()
            padFactor = max(1, int(form.get("padFactor", padFactor)))
            x = make_sequence(func_str, N)
            res = compute_dft_plot(x, padFactor)
            plot_data, dft_label = res["plot_data"], res["dft_label"]
        except Exception as e:
            dft_label = f"Error: {e}"

    return render_template(
        "discrete/dft.html",
        N=N,
        func=func_str,
        padFactor=padFactor,
        plot_data=plot_data,
        dft_label=dft_label
    )

@discrete_fourier_bp.route("/update", methods=["POST"])
def update_dft():
    data = request.get_json(force=True)
    try:
        N = max(1, int(data.get("N", 8)))
        func_str = data.get("func", "sin(2*pi*n/N)").strip()
        padFactor = max(1, int(data.get("padFactor", 10)))
        x = make_sequence(func_str, N)
        res = compute_dft_plot(x, padFactor)
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(res)
