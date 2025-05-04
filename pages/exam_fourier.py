from flask import Blueprint, render_template, request, session, redirect, url_for
import time, random, io, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

exam_fourier_bp = Blueprint("exam_fourier", __name__)

# Define time-domain functions and their analytic Fourier transforms
w0_values = [1, 2, 3]
def_rect = lambda t: np.where(np.abs(t) < 0.5, 1, 0)
def_tri = lambda t: np.maximum(1 - np.abs(t), 0)
def_sinc = lambda t: np.sinc(t)
def_sinc2 = lambda t: np.sinc(t) ** 2
def_inv_t = lambda t: np.where(t != 0, 1 / t, 0)
def_sign = lambda t: np.sign(t)

def T_rect(w):
    x = w / 2
    return np.where(w == 0, 1.0, np.sin(x) / x)

def T_tri(w):
    x = w / 2
    return (np.sin(x) / x) ** 2

def T_sinc(w):
    return np.where(np.abs(w) <= np.pi, 1.0, 0.0)

def T_sinc2(w):
    return np.where(np.abs(w) <= 2 * np.pi,
                    np.pi * (1 - np.abs(w) / (2 * np.pi)),
                    0.0)

def T_inv_t(w):
    return -1j * np.pi * np.sign(w)

def T_sign(w):
    return -2j / w

@exam_fourier_bp.route("/start_exam")
def start_exam():
    # Initialize exam with 10 mixed-direction problems
    problems = []
    for _ in range(10):
        name = random.choice([
            "rect", "tri", "sinc", "sinc2",
            "inv_t", "sign", "cexp", "cos", "sin"
        ])
        direction = random.choice(["TIME_TO_FREQ", "FREQ_TO_TIME"])
        w0 = random.choice(w0_values)
        shift = random.randint(-3, 3)
        scale = random.randint(1, 3)
        width = random.randint(1, 3)
        problems.append({
            "name": name,
            "w0": w0,
            "shift": shift,
            "scale": scale,
            "width": width,
            "direction": direction,
            "correct": None
        })
    session["exam_data"] = {"start_time": time.time(), "problems": problems}
    return redirect(url_for("exam_fourier.exam_fourier"))

@exam_fourier_bp.route("/", methods=["GET", "POST"])
def exam_fourier():
    data = session.get("exam_data")
    if request.method == "GET":
        if not data:
            return render_template("exam_fourier.html", started=False)
        rendered = []
        for idx, prob in enumerate(data["problems"]):
            img, _ = _render_problem(prob)
            rendered.append({
                "idx": idx,
                "img": img,
                "direction": prob["direction"]
            })
        return render_template(
            "exam_fourier.html",
            started=True,
            problems=rendered
        )

    # POST: grade submitted answers
    answers = []
    for i in range(10):
        val = request.form.get(f"ans_{i}")
        answers.append(int(val) if val is not None else None)

    correct_count = 0
    results = []
    for i, prob in enumerate(data["problems"]):
        user_ans = answers[i]
        img, _ = _render_problem(prob, highlight=user_ans)
        if user_ans == prob.get("correct"):
            correct_count += 1
        results.append({
            "index": i,
            "img_data": img,
            "correctIndex": prob.get("correct"),
            "userIndex": user_ans
        })

    elapsed = time.time() - data.get("start_time", time.time())
    accuracy = round(correct_count / 10 * 100, 1)
    total_time = round(elapsed, 1)
    score_acc = (correct_count / 10) * 70
    time_factor = max(0, 1 - (elapsed - 60) / 300)
    score_time = time_factor * 30
    final_score = round(max(0, min(score_acc + score_time, 100)), 1)

    session.pop("exam_data", None)
    return render_template(
        "exam_fourier_result.html",
        correct_count=correct_count,
        accuracy=accuracy,
        total_time=total_time,
        final_score=final_score,
        results=results
    )

# Helper to render a single problem
def _render_problem(prob, highlight=None):
    t = np.linspace(-10, 10, 512)
    w = np.linspace(-10, 10, 512)
    name, w0 = prob["name"], prob["w0"]
    # Map names to functions
    time_funcs = {
        "rect": def_rect,
        "tri": def_tri,
        "sinc": def_sinc,
        "sinc2": def_sinc2,
        "inv_t": def_inv_t,
        "sign": def_sign,
        "cexp": lambda tt: np.exp(1j * w0 * tt),
        "cos": lambda tt: np.cos(w0 * tt),
        "sin": lambda tt: np.sin(w0 * tt)
    }
    ft_funcs = {
        "rect": T_rect,
        "tri": T_tri,
        "sinc": T_sinc,
        "sinc2": T_sinc2,
        "inv_t": T_inv_t,
        "sign": T_sign,
        # Narrow gaussian impulses for deltas
        "cexp": lambda omega: 2 * np.pi * (np.abs(omega - w0) < 1e-1),
        "cos": lambda omega: np.pi * (
            (np.abs(omega - w0) < 1e-1) |
            (np.abs(omega + w0) < 1e-1)
        ),
        "sin": lambda omega: -1j * np.pi * (np.abs(omega - w0) < 1e-1) + 1j * np.pi * (np.abs(omega + w0) < 1e-1)
    }
    # Generate time-domain signal
    sig = prob["scale"] * time_funcs[name]((t - prob["shift"]) / prob["width"])
    # Build analytic transforms
    def make_transform(wd, sc, sh):
        base = sc * wd * ft_funcs[name](w * wd)
        return base * np.exp(-1j * w * sh)
    transforms = [
        make_transform(prob["width"], prob["scale"], prob["shift"]),
        make_transform(prob["width"] + 1, prob["scale"], prob["shift"]),
        make_transform(prob["width"], prob["scale"] + 1, prob["shift"]),
        make_transform(prob["width"], prob["scale"], prob["shift"] + 1)
    ]
    # Prepare options based on direction
    if prob["direction"] == "FREQ_TO_TIME":
        opts = [
            np.real(
                np.fft.ifft(np.fft.ifftshift(X)) * len(w) * (w[1] - w[0])
            )
            for X in transforms
        ]
    else:
        opts = transforms
    # Shuffle and record correct index
    idxs = list(range(4))
    random.shuffle(idxs)
    shuffled = [opts[i] for i in idxs]
    prob["correct"] = idxs.index(0)
    # Plot
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    ax0 = fig.add_subplot(gs[0, :])
    if prob["direction"] == "TIME_TO_FREQ":
        ax0.plot(t, sig.real)
    else:
        ax0.plot(w, transforms[0].real)
    ax0.set_xlim(-10, 10)
    ax0.grid(True)
    for i, opt in enumerate(shuffled):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        if prob["direction"] == "TIME_TO_FREQ":
            ax.plot(w, np.abs(opt), label='|X|')
            ax.plot(w, np.angle(opt), linestyle='--', label='∠X')
            ax.plot(w, np.imag(opt), linestyle=':', label='Im X')
            ax.legend()
        else:
            ax.plot(t, opt)
        ax.set_xlim(-10, 10)
        ax.grid(True)
        ax.set_title(f"Option {i+1}")
        if highlight is not None and i == highlight:
            color = 'lightgreen' if i == prob['correct'] else 'lightcoral'
            ax.patch.set_facecolor(color)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return img_str, shuffled
