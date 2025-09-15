# pages/training/training_convolution.py
from flask import Blueprint, render_template, request, jsonify
import io, base64, random
from functools import partial
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from scipy.signal import convolve

DELTA_EPS = 1e-2

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
        """rect(t/2)"""
        return np.where(np.abs(t) < 1, 1, 0)
    def sign(t):
        return np.sign(t)
    def delta(t):
        return 1/(np.sqrt(np.pi)*DELTA_EPS) * np.exp(-(t/DELTA_EPS)**2)
    def two_deltas(t):
        return delta(t+1) + delta(t-1)
    def multi_deltas(t, pos=None):
        if pos is None: # randomly choose positions (overlapping spikes allowed)
            n = random.choice([2, 3])
            pos = [random.choice([-1, 0, 1]) for _ in range(n)]
        d = np.zeros_like(t)
        for pos_cur in pos:
            d += delta(t-pos_cur)
        return d
    def t_rect(t):
        return t * rect(t)
    def sign_rect(t):
        return np.sign(t) * rect(t)
    def tri_sign(t):
        return tri(t) * np.sign(t)
    def neg_rect_tri(t):
        return -rect(t*2) * tri(t)

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
            "2-3 deltas": partial(multi_deltas, pos=[random.choice([-1, 0, 1]) for _ in range(random.choice([2, 3]))]),
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
    try:
        # 2) Time axis
        tmax = 10 # calculation time span
        p_per_timestep = 128 # should be power of 2, <128 can lead to scaling errors for delta!
        t = np.linspace(-tmax, tmax, 2*tmax*p_per_timestep+1)

        # 3) Randomly pick 2 function names
        names = list(func_dict.keys())
        chosen = random.choices(names, k=2) if len(names) > 1 else names*2
        # assure no double infinity functions step(t) or sign(t) and replace one of them with a random finite function
        if all(name in ["step(t)", "sign(t)"] for name in chosen):
            chosen[random.randint(0,1)] = random.choice([n for n in names if n not in ["step(t)", "sign(t)"]])
        # assure no double delta(t) and replace one of them with a random function
        if all("delta" in name for name in chosen):
            chosen[random.randint(0,1)] = random.choice([n for n in names if "delta" not in n])
        func1_name, func2_name = chosen[0], chosen[1]
        func1, func2 = func_dict[func1_name], func_dict[func2_name]

        # 4) Random shift & scale parameters
        shift_min, shift_max = -1, 1
        scale_min, scale_max = -2, 2
        width_min, width_max = 0.5, 1.5
        shift_choices = np.arange(shift_min, shift_max + 0.1, 0.5).tolist()
        scale_choices = np.arange(scale_min, scale_max + 0.1, 0.5).tolist()
        scale_choices.remove(0)  # avoid zero scale
        width_choices = np.arange(width_min, width_max + 0.1, 0.5).tolist()
        shift1 = float(random.choice(shift_choices))
        scale1 = float(random.choice(scale_choices))
        width1 = float(random.choice(width_choices))
        if "delta" in func1_name:  # delta function should not be stretched
            width1 = 1.0

        if difficulty == "EASY" and np.abs(scale1*2) % 2: # avoid double 0.5 scale
            scale_choices = np.arange(scale_min, scale_max + 0.1, 1).tolist()
            scale_choices.remove(0)
        shift2 = float(random.choice(shift_choices))
        scale2 = float(random.choice(scale_choices))
        width2 = float(random.choice(width_choices))
        if "delta" in func2_name:  # delta function should not be stretched
            width2 = 1.0

        # Evaluate the input signals
        f1 = scale1 * func1((t - shift1) / width1)
        f2 = scale2 * func2((t - shift2) / width2)

        def conv_cont(f1, f2, t_diff):
            """
            Continuous convolution of two functions with constant time span between samples.
            """
            return convolve(f1, f2, mode='same') * t_diff

        # Correct convolution
        dt = t[1] - t[0]
        correct_conv = conv_cont(f1, f2, dt)

        # Distractors (make sure, correct solution appears only once and no mathematically duplicate options are created)
        non_stretchable = ["step(t)", "delta(t)", "sign(t)", "2-3 deltas"]
        # 1) Modify width1
        if func1_name not in non_stretchable:
            width1_remaining = [w for w in width_choices if w != width1]
            new_width1 = random.choice(width1_remaining)
            f1_d1 = scale1 * func1((t - shift1) / new_width1)
            dconv1 = conv_cont(f1_d1, f2, dt)
        elif func2_name not in non_stretchable:
            width2_remaining = [w for w in width_choices if w != width2]
            new_width2 = random.choice(width2_remaining)
            f2_d1 = scale2 * func2((t - shift2) / new_width2)
            dconv1 = conv_cont(f1, f2_d1, dt)
        else: # modify shift1
            shift1_remaining = [s for s in shift_choices if s != shift1]
            new_shift1 = random.choice(shift1_remaining)
            f1_d1 = scale1 * func1((t - new_shift1) / width1)
            dconv1 = conv_cont(f1_d1, f2, dt)

        # 2) Modify scale2
        scale2_remaining = [s for s in scale_choices if s != scale2]
        new_scale2 = random.choice(scale2_remaining)
        f2_d2 = new_scale2 * func2((t - shift2) / width2)
        dconv2 = conv_cont(f1, f2_d2, dt)

        # 3) Modify shift1 if possible else scale2
        if not all(func in non_stretchable for func in [func1_name, func2_name]):
            shift1_remaining = [s for s in shift_choices if s != shift1]
            new_shift1 = random.choice(shift1_remaining)
            f1_d3 = scale1 * func1((t - new_shift1) / width1)
            dconv3 = conv_cont(f1_d3, f2, dt)
        else:
            scale2_remaining = [s for s in scale2_remaining if s != new_scale2]
            new_scale2 = random.choice(scale2_remaining)
            f2_d3 = new_scale2 * func2((t - shift2) / width2)
            dconv3 = conv_cont(f1, f2_d3, dt)

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
        ax_f1.set_title("$x(t)$")
        ax_f2.set_title("$h(t)$")
        corr_fac = np.sqrt(np.pi) * DELTA_EPS
        ax_f1.plot(t, f1 * (corr_fac if "delta" in func1_name else 1), color="tab:blue", lw=2, zorder=3)
        ax_f2.plot(t, f2 * (corr_fac if "delta" in func2_name else 1), color="tab:orange", lw=2, zorder=3)

        # bottom 4 subplots
        for i in range(4):
            row, col = 1 + i // 2, i % 2
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"$y_{i+1}(t)$")
            ax.plot(t, shuffled[i], color="tab:green", lw=2, zorder=3)

        # settings for all axes
        xlim = 5.5
        xlim_mask = np.abs(t) <= xlim
        for ax in fig.get_axes():
            ax.set_xlabel("$t$")
            ax.axhline(0, color='dimgray', zorder=2)
            ax.axvline(0, color='dimgray', zorder=2)
            ax.margins(y=0.2)
            ax.set_xlim(-xlim, xlim)
            # select ylim only based on values in t = -xlim...xlim
            ydata_visible = ax.get_lines()[0].get_ydata()[xlim_mask]
            y_abs_max = np.abs(ydata_visible).max()
            y_span = ydata_visible.max() - ydata_visible.min()
            if y_abs_max <= 2.1:
                ylim_abs = max(y_abs_max * 1.1, 1.1)
                ax.set_ylim(-ylim_abs, ylim_abs)
            else:
                ax.set_ylim(ydata_visible.min() - 0.05 * y_span, ydata_visible.max() + 0.05 * y_span)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            if y_span <= 8:
                ax.yaxis.set_major_locator(MultipleLocator(1))
                ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            else:
                ax.yaxis.set_major_locator(MultipleLocator(5))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(which='both', zorder=0)

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plot_data = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)

    except Exception as e:
        print(e)
    # Return JSON
    return {
        "plot_data": plot_data,
        "correctIndex": correct_idx
    }
