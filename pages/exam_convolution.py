from flask import Blueprint, render_template, request, session, redirect, url_for
import time, random
import numpy as np
import io, base64
import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.signal import convolve

DELTA_EPS = 1e-2

exam_convolution_bp = Blueprint("exam_convolution", __name__)

@exam_convolution_bp.route("/start_exam")
def start_exam():
    """
    Creates 10 random param sets (only numeric seeds + correctIndex=None)
    and saves them to session. Also stores start_time.
    Then redirects to the main exam page.
    """
    problems = []
    for _ in range(10):
        problems.append(create_problem_params())
    session["exam_convolution_data"] = {
        "start_time": time.time(),
        "problems": problems
    }
    return redirect(url_for("exam_convolution.exam_convolution"))


@exam_convolution_bp.route("/", methods=["GET", "POST"])
def exam_convolution():
    """
    GET:
      - If no exam in session, show 'Start Exam' button.
      - If exam in session, generate base64 images on the fly for each question
        (storing only an 'img_data' in a local list for the template),
        set correctIndex in each problem so we can check answers on POST,
        but do NOT store the large base64 strings in the session.
    POST:
      - User finalizes answers => compute final score => show exam_convolution_result
    """
    if request.method == "GET":
        exam_data = session.get("exam_convolution_data")
        if not exam_data:
            # No exam => show "Start Exam" button
            return render_template("exam_convolution.html", started=False)

        # exam in progress => for each problem, generate an image & correctIndex
        problems_for_template = []
        for i, problem in enumerate(exam_data["problems"]):
            # generate the base64 figure, sets problem["correctIndex"]
            base64_img = generate_problem_plot(problem, highlight_index=None)
            problems_for_template.append({
                "index": i,
                "img_data": base64_img
            })

        # Now store the updated correctIndex in session (small int), 
        # but do NOT store base64 images => keep session small
        session["exam_convolution_data"] = exam_data

        # pass problems_for_template so the template can see images & radio buttons
        return render_template("exam_convolution.html",
                               started=True,
                               problems_for_template=problems_for_template)

    else:
        # POST => user finalizing answers
        try:
            exam_data = session.get("exam_convolution_data")
            if not exam_data:
                return "No exam found in session. Please start again."

            # read user answers
            user_answers = []
            for i in range(10):
                ans_str = request.form.get(f"answer_{i}")
                if ans_str is None:
                    user_answers.append(None)
                else:
                    user_answers.append(int(ans_str))

            correct_count = 0
            for i in range(10):
                problem = exam_data["problems"][i]
                correct_idx = problem.get("correctIndex")
                if correct_idx is None:
                    return f"Error: problem {i} has no correctIndex key!"
                if user_answers[i] == correct_idx:
                    correct_count += 1

            accuracy = correct_count / 10.0
            start_time = exam_data["start_time"]
            end_time = time.time()
            total_time = end_time - start_time

            # simple formula => 70% accuracy, 30% time factor
            base_accuracy_score = accuracy * 70
            # under 7.5min for full time score, no time score after 20min
            time_factor = np.clip(1 - (total_time - 60*7.5)/(60*12.5), 0, 1)
            time_score = time_factor * 30
            final_score = base_accuracy_score + time_score
            final_score = np.clip(final_score, 0, 100)

            # We'll build a results_for_template so we can show each problem figure again,
            # now color-coded for correctIndex vs userIndex
            results_for_template = []
            for i in range(10):
                problem = exam_data["problems"][i]
                correct_idx = problem["correctIndex"]
                user_idx = user_answers[i]

                # re-generate the figure, highlight the correct option in green,
                # and if user_idx != correct_idx, highlight user_idx in red
                # if user_idx == correct_idx => highlight both in green
                base64_result_img = generate_problem_plot(problem, highlight_index=user_idx)

                results_for_template.append({
                    "index": i,
                    "userIndex": user_idx,
                    "correctIndex": correct_idx,
                    "img_data": base64_result_img
                })

            # remove from session => done
            session.pop("exam_convolution_data", None)

            return render_template("exam_convolution_result.html",
                                accuracy=round(accuracy*100,1),
                                correct_count=correct_count,
                                total_time=round(total_time,1),
                                final_score=round(final_score,1),
                                results=results_for_template)
        except Exception as e:
            print(e)
            return render_template("exam_convolution.html",
                                   started=False,
                                   error_message="An error occurred while processing your exam. Please try again.")

# function definitions
def tri(t):
    return np.maximum(1 - np.abs(t), 0)
def step(t):
    return np.where(t >= 0, 1, 0)
def rect(t):
    """rect(t/2)"""
    return np.where(np.abs(t) < 1, 1, 0)
def delta(t):
    return 1/(np.sqrt(np.pi)*DELTA_EPS) * np.exp(-(t/DELTA_EPS)**2)

_func_dict = {"tri(t)": tri, "step(t)": step, "rect(t)": rect, "delta(t)": delta}
shift_min, shift_max = -1, 1
scale_min, scale_max = -2, 2
width_min, width_max = 0.5, 1.5

def create_problem_params():
    """Random seeds for two signals. correctIndex=None initially."""
    # choose functions
    chosen = random.choices(list(_func_dict.keys()), k=2)
    if all("delta" in name for name in chosen):
        fnames = list(_func_dict.keys())
        chosen[random.randint(0,1)] = random.choice([n for n in fnames if "delta" not in n])
    # choose additional function parameters
    shift_choices = np.arange(shift_min, shift_max + 0.1, 0.5).tolist()
    scale_choices = np.arange(scale_min, scale_max + 0.1, 0.5).tolist()
    scale_choices.remove(0)  # avoid zero scale
    width_choices = np.arange(width_min, width_max + 0.1, 0.5).tolist()
    shift1 = float(random.choice(shift_choices))
    scale1 = float(random.choice(scale_choices))
    width1 = float(random.choice(width_choices))
    if "delta" in chosen[0]:  # delta function should not be stretched
        width1 = 1.0

    if np.abs(scale1*2) % 2: # avoid double 0.5 scale
        scale_choices = np.arange(scale_min, scale_max + 0.1, 1).tolist()
        scale_choices.remove(0)
    shift2 = float(random.choice(shift_choices))
    scale2 = float(random.choice(scale_choices))
    width2 = float(random.choice(width_choices))
    if "delta" in chosen[1]:  # delta function should not be stretched
        width2 = 1.0

    return {
        "func1_name": chosen[0],
        "func2_name": chosen[1],
        "shift1": shift1,
        "scale1": scale1,
        "width1": width1,
        "shift2": shift2,
        "scale2": scale2,
        "width2": width2,
        "correctIndex": None
    }

def generate_problem_plot(problem, highlight_index=None):
    """
    Evaluate f1,f2 => correct_conv + 3 distractors => 6-subplot figure => base64
    Sets problem["correctIndex"] if not set. 
    highlight_index: userIndex (0..3 or None).

    If highlight_index != correct_index, highlight userIndex in red.
    If highlight_index is None => normal exam question (no color-coded highlight).
    If highlight_index is not None => final results page => color-coded subplots.
    """
    f1_name = problem["func1_name"]
    f2_name = problem["func2_name"]
    shift1  = problem["shift1"]
    scale1  = problem["scale1"]
    width1  = problem["width1"]
    shift2  = problem["shift2"]
    scale2  = problem["scale2"]
    width2  = problem["width2"]
    func1 = _func_dict[f1_name]
    func2 = _func_dict[f2_name]
    tmax = 10 # calculation time span
    p_per_timestep = 128 # should be power of 2, <128 can lead to scaling errors for delta!
    t = np.linspace(-tmax, tmax, 2*tmax*p_per_timestep+1)
    dt = t[1] - t[0]

    def conv_cont(f1, f2, t_diff):
        """
        Continuous convolution of two functions with constant time span between samples.
        """
        return convolve(f1, f2, mode='same') * t_diff

    f1 = scale1 * func1((t - shift1) / width1)
    f2 = scale2 * func2((t - shift2) / width2)
    correct_conv = conv_cont(f1, f2, dt)

    # Distractors (make sure, correct solution appears only once and no mathematically duplicate options are created)
    non_stretchable = ["delta(t)", "step(t)"]
    shift_choices = np.arange(shift_min, shift_max + 0.1, 0.5).tolist()
    scale_choices = np.arange(scale_min, scale_max + 0.1, 0.5).tolist()
    scale_choices.remove(0)  # avoid zero scale
    if np.abs(scale1*2) % 2: # avoid double 0.5 scale
        scale_choices = np.arange(scale_min, scale_max + 0.1, 1).tolist()
        scale_choices.remove(0)
    width_choices = np.arange(width_min, width_max + 0.1, 0.5).tolist()
    # 1) Modify width1
    if f1_name not in non_stretchable:
        width1_remaining = [w for w in width_choices if w != width1]
        new_width1 = random.choice(width1_remaining)
        f1_d1 = scale1 * func1((t - shift1) / new_width1)
        dconv1 = conv_cont(f1_d1, f2, dt)
    elif f2_name not in non_stretchable:
        width2_remaining = [w for w in width_choices if w != width2]
        new_width2 = random.choice(width2_remaining)
        f2_d1 = scale2 * func2((t - shift2) / new_width2)
        dconv1 = conv_cont(f1, f2_d1, dt)
    else: # modify shift1 if both not stretchable
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
    if not all(func in non_stretchable for func in [f1_name, f2_name]):
        shift1_remaining = [s for s in shift_choices if s != shift1]
        new_shift1 = random.choice(shift1_remaining)
        f1_d3 = scale1 * func1((t - new_shift1) / width1)
        dconv3 = conv_cont(f1_d3, f2, dt)
    else:
        scale2_remaining = [s for s in scale2_remaining if s != new_scale2]
        new_scale2 = random.choice(scale2_remaining)
        f2_d3 = new_scale2 * func2((t - shift2) / width2)
        dconv3 = conv_cont(f1, f2_d3, dt)

    idxs = list(range(4))
    random.shuffle(idxs)

    # if the problem does not have correctIndex yet, set it
    if problem["correctIndex"] is None:
        c_idx = idxs.index(0)
        problem["correctIndex"] = c_idx
    else:
        c_idx = problem["correctIndex"]
    
    options = [correct_conv, dconv1, dconv2, dconv3]
    options_shuffled = [options[i] for i in idxs]

    # Debugging code in case of duplicate options:
    # if any(np.allclose(options_shuffled[i], options_shuffled[j]) for i in range(4) for j in range(i+1,4)):
    #     print("Warning: duplicate options detected!")
    #     print(f"f1: {f1_name}, shift1: {shift1}, scale1: {scale1}, width1: {width1}")
    #     print(f"f2: {f2_name}, shift2: {shift2}, scale2: {scale2}, width2: {width2}")
    #     for i in range(4):
    #         for j in range(i+1,4):
    #             if np.allclose(options_shuffled[i], options_shuffled[j]):
    #                 print(f"Options {i} and {j} are duplicates.")
    #                 # print(f"Indices: {idxs[i]}, {idxs[j]}")
    #                 print(shift_choices)
    #                 print(shift1_remaining)
    #                 print(new_shift1)
    
    # Now create the figure
    rcParams['text.parse_math'] = True
    rcParams['text.usetex'] = True
    
    fig = plt.figure(figsize=(8,8))
    gs = fig.add_gridspec(nrows=3, ncols=2, hspace=0.5, wspace=0.3)

    # top row => f1, f2
    axf1 = fig.add_subplot(gs[0,0])
    axf2 = fig.add_subplot(gs[0,1])
    axf1.set_title("$x(t)$")
    axf2.set_title("$h(t)$")
    corr_fac = np.sqrt(np.pi) * DELTA_EPS
    if "delta" in f1_name: # correct delta scaling (calculation with narrow Gaussian)
        f1 *= corr_fac
    if "delta" in f2_name:
        f2 *= corr_fac
    axf1.plot(t, f1, color="blue", lw=2, zorder=3)
    axf2.plot(t, f2, color="green", lw=2, zorder=3)

    # bottom 4 subplots
    for i in range(4):
        if highlight_index is not None: # highlight logic if highlight_index != None
            if i == highlight_index: # option picked by user
                if i == c_idx: # correct
                    color = "green"
                    lw = 3.0
                else: # incorrect
                    color = "red"
                    lw = 3.0
            elif i == c_idx: # user did not pick correct option
                color = "red"
                lw = 2
            else: # user did not pick incorrect option
                color = "green"
                lw = 2
        else: # appearance during exam
            color="black"
            lw = 2
        
        row, col = 1 + i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(f"$y_{i+1}(t)$")
        ax.plot(t, options_shuffled[i], color=color, lw=lw, zorder=3)

    # settings for all axes
    xlim = 5.5
    xlim_mask = np.abs(t) <= xlim
    for i, ax in enumerate(fig.get_axes()):
        ax.set_xlabel("$t$")
        ax.axhline(0, color='dimgray', zorder=2)
        ax.axvline(0, color='dimgray', zorder=2)
        ax.set_xlim(-xlim, xlim)
        # select ylim only based on values in t = -xlim...xlim
        ydata_visible = ax.get_lines()[0].get_ydata()[xlim_mask]
        y_abs_max = np.abs(ydata_visible).max()
        y_span = ydata_visible.max() - ydata_visible.min()
        if y_abs_max <= 2.1: # plot symmetric y range
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
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    return base64_img
