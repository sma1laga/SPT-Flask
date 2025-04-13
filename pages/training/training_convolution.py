# pages/training/training_convolution.py
from flask import Blueprint, render_template, request, jsonify
import io, base64, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve

training_convolution_bp = Blueprint("training_convolution", __name__)

@training_convolution_bp.route("/")
def training_convolution():
    """
    Renders the Training Convolution page.
    The page includes difficulty radio buttons, a 'Generate Problem' button,
    an image area for the 4 options, and 4 'Option' buttons for the user guess.
    """
    return render_template("training_convolution.html")


@training_convolution_bp.route("/generate", methods=["POST"])
def generate_problem():
    """
    AJAX route that receives difficulty from the user and returns:
     - A base64-encoded image showing the 2 input functions (top) 
       and 4 convolution options (bottom).
     - The index of the correct answer (0..3).
    The client can store this index to compare when user picks an option.
    """
    data = request.get_json(force=True)
    difficulty = data.get("difficulty", "EASY").upper()

    result = create_convolution_problem(difficulty)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)


@training_convolution_bp.route("/check_answer", methods=["POST"])
def check_answer():
    """
    AJAX route that receives the user's selected index and the correct index,
    returns whether user is correct or not.
    """
    data = request.get_json(force=True)
    selected_index = data.get("selectedIndex")
    correct_index = data.get("correctIndex")
    if selected_index == correct_index:
        return jsonify({"feedback": "Correct!"})
    else:
        return jsonify({"feedback": "Incorrect. Try again!"})


def create_convolution_problem(difficulty):
    """
    Generates two random input functions, computes the correct convolution,
    plus three distractors. Returns a base64 plot plus the correct answer index.
    """
    # 1) Difficulty-based function pools
    def tri(t):
        return np.maximum(1 - np.abs(t), 0)
    def step(t):
        return np.where(t >= 0, 1, 0)
    def rect(t):
        return np.where(np.abs(t) < 0.5, 1, 0)
    def delta(t):
        d = np.zeros_like(t)
        idx = np.argmin(np.abs(t))
        d[idx] = 1
        return d
    def sign(t):
        return np.sign(t)
    def two_deltas(t):
        d = np.zeros_like(t)
        idx1 = np.argmin(np.abs(t + 2))
        idx2 = np.argmin(np.abs(t - 2))
        d[idx1] = 1
        d[idx2] = 1
        return d
    def multi_deltas(t):
        d = np.zeros_like(t)
        n = random.choice([2, 3])
        indices = np.random.choice(len(t), n, replace=False)
        d[indices] = 1
        return d
    def t_rect(t):
        return t * rect(t)
    def sign_rect(t):
        return np.sign(t) * rect(t)
    def tri_sign(t):
        return tri(t) * np.sign(t)
    def neg_rect_tri(t):
        return -rect(t) * tri(t)

    if difficulty == "EASY":
        func_dict = {
            "tri(t)": tri,
            "step(t)": step,
            "rect(t)": rect,
            "delta(t)": delta
        }
    elif difficulty == "MEDIUM":
        func_dict = {
            "tri(t)": tri,
            "step(t)": step,
            "rect(t)": rect,
            "delta(t)": delta,
            "sign(t)": sign,
            "2 deltas": two_deltas,
            "t*rect(t)": t_rect
        }
    elif difficulty == "HARD":
        func_dict = {
            "tri(t)": tri,
            "step(t)": step,
            "rect(t)": rect,
            "delta(t)": delta,
            "sign(t)": sign,
            "2-3 deltas": multi_deltas,
            "sign(t)*rect(t)": sign_rect,
            "tri(t)*sign(t)": tri_sign,
            "-rect(t)*tri(t)": neg_rect_tri
        }
    else:
        # default to EASY if something else
        func_dict = {
            "tri(t)": tri,
            "step(t)": step,
            "rect(t)": rect,
            "delta(t)": delta
        }

    # 2) Time axis
    t = np.linspace(-10, 10, 400)
    dt = t[1] - t[0]

    # 3) Randomly pick 2 function names
    names = list(func_dict.keys())
    if len(names) < 2:
        return {"error": "Not enough functions."}
    chosen = random.sample(names, 2)
    func1_name, func2_name = chosen[0], chosen[1]
    func1, func2 = func_dict[func1_name], func_dict[func2_name]

    # 4) Random shift & scale parameters
    shift_choices = np.arange(-3, 3.5, 0.5)
    scale_choices = np.arange(0.5, 2.6, 0.5)
    shift1 = float(random.choice(shift_choices))
    scale1 = float(random.choice(scale_choices))
    width1 = float(random.choice(scale_choices))

    shift2 = float(random.choice(shift_choices))
    scale2 = float(random.choice(scale_choices))
    width2 = float(random.choice(scale_choices))

    # Evaluate the input signals
    f1 = scale1 * func1((t - shift1) / width1)
    f2 = scale2 * func2((t - shift2) / width2)

    # Correct convolution
    correct_conv = convolve(f1, f2, mode='same') * dt

    # Distactors
    # 1) Modify width1
    new_width1 = max(0.5, min(width1 + random.choice([-1.0, 1.0]), 2.5))
    f1_d1 = scale1 * func1((t - shift1)/new_width1)
    dconv1 = convolve(f1_d1, f2, mode='same') * dt

    # 2) Modify scale2
    new_scale2 = max(0.5, min(scale2 + random.choice([-1.0, 1.0]), 2.5))
    f2_d2 = new_scale2 * func2((t - shift2)/width2)
    dconv2 = convolve(f1, f2_d2, mode='same') * dt

    # 3) Modify shift2 or width1 plus scale
    extrashift = random.choice([-1.0, 1.0])
    new_width1_3 = max(0.5, min(width1 + extrashift, 2.5))
    new_scale2_3 = max(0.5, min(scale2 + random.choice([-1.0, 1.0]), 2.5))
    f1_d3 = scale1 * func1((t - shift1)/new_width1_3)
    f2_d3 = new_scale2_3 * func2((t - (shift2 + extrashift))/width2)
    dconv3 = convolve(f1_d3, f2_d3, mode='same') * dt

    # Shuffle them
    options = [correct_conv, dconv1, dconv2, dconv3]
    indices = list(range(4))
    random.shuffle(indices)
    shuffled = [options[i] for i in indices]
    correct_idx = indices.index(0)  # where the correct one went

    # 5) Plot
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(nrows=3, ncols=2, hspace=0.5, wspace=0.3)

    ax_f1 = fig.add_subplot(gs[0,0])
    ax_f2 = fig.add_subplot(gs[0,1])
    ax_f1.set_title(f"Input 1: {func1_name}\nshift={shift1}, scale={scale1}, width={width1}")
    ax_f2.set_title(f"Input 2: {func2_name}\nshift={shift2}, scale={scale2}, width={width2}")
    ax_f1.plot(t, f1, color="blue")
    ax_f2.plot(t, f2, color="green")
    ax_f1.grid(True)
    ax_f2.grid(True)
    ax_f1.set_xlim(t[0], t[-1])
    ax_f2.set_xlim(t[0], t[-1])

    # bottom 4 subplots
    for i in range(4):
        row = 1 + i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(f"Option {i+1}")
        ax.plot(t, shuffled[i], color="red")
        ax.grid(True)
        ax.set_xlim(t[0], t[-1])

    fig.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    # Return JSON
    return {
        "plot_data": plot_data,
        "correctIndex": correct_idx
    }
