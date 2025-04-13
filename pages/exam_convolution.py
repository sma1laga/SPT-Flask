from flask import Blueprint, render_template, request, session, redirect, url_for
import time, random
import numpy as np
import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve

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
        time_factor = max(0, 1 - (total_time - 60)/300)
        time_score = time_factor * 30
        final_score = base_accuracy_score + time_score
        final_score = max(0, min(final_score, 100))

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
            base64_result_img = generate_problem_plot(problem, highlight_index=user_idx, correct_index=correct_idx)

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

def create_problem_params():
    """Random seeds for two signals. correctIndex=None initially."""
    def tri(t):
        return np.maximum(1 - np.abs(t),0)
    def step(t):
        return np.where(t>=0,1,0)
    def rect(t):
        return np.where(np.abs(t)<0.5,1,0)
    def delta(t):
        d=np.zeros_like(t)
        idx=np.argmin(np.abs(t))
        d[idx]=1
        return d

    func_dict = {"tri(t)": tri,"step(t)": step,"rect(t)": rect,"delta(t)": delta}

    chosen = random.sample(list(func_dict.keys()),2)
    shift1 = float(random.choice(np.arange(-2,2.5,0.5)))
    scale1 = float(random.choice(np.arange(0.5,2.1,0.5)))
    shift2 = float(random.choice(np.arange(-2,2.5,0.5)))
    scale2 = float(random.choice(np.arange(0.5,2.1,0.5)))

    return {
        "func1_name": chosen[0],
        "func2_name": chosen[1],
        "shift1": shift1,
        "scale1": scale1,
        "shift2": shift2,
        "scale2": scale2,
        "correctIndex": None
    }

def generate_problem_plot(problem, highlight_index=None, correct_index=None):
    """
    Evaluate f1,f2 => correct_conv + 3 distractors => 6-subplot figure => base64
    Sets problem["correctIndex"] if not set. 
    highlight_index: userIndex (0..3 or None). 
    correct_index: if provided, we'll color that subplot green. 
      If highlight_index != correct_index, highlight userIndex in red.

    If highlight_index is None => normal exam question (no color-coded highlight).
    If highlight_index is not None => final results page => color-coded subplots.
    """
    def tri(t):
        return np.maximum(1 - np.abs(t),0)
    def step(t):
        return np.where(t>=0,1,0)
    def rect(t):
        return np.where(np.abs(t)<0.5,1,0)
    def delta(t):
        d=np.zeros_like(t)
        idx=np.argmin(np.abs(t))
        d[idx]=1
        return d

    func_dict = {"tri(t)": tri,"step(t)": step,"rect(t)": rect,"delta(t)": delta}

    f1_name = problem["func1_name"]
    f2_name = problem["func2_name"]
    shift1  = problem["shift1"]
    scale1  = problem["scale1"]
    shift2  = problem["shift2"]
    scale2  = problem["scale2"]

    t = np.linspace(-5,5,200)
    dt = t[1]-t[0]

    def convme(a,b):
        return convolve(a,b, mode='same')*dt

    y1 = scale1*func_dict[f1_name](t - shift1)
    y2 = scale2*func_dict[f2_name](t - shift2)
    correct_conv = convme(y1,y2)

    # distractors
    y1d1 = (scale1 + random.choice([-0.5,0.5]))*func_dict[f1_name](t - shift1)
    dconv1= convme(y1d1, y2)
    y2d2 = (scale2 + random.choice([-0.5,0.5]))*func_dict[f2_name](t - shift2)
    dconv2= convme(y1, y2d2)
    y1d3 = (scale1 + random.choice([-0.5,0.5]))*func_dict[f1_name](t - shift1)
    y2d3 = (scale2 + random.choice([-0.5,0.5]))*func_dict[f2_name](t - shift2)
    dconv3= convme(y1d3,y2d3)

    options = [correct_conv, dconv1, dconv2, dconv3]
    idxs = list(range(4))
    random.shuffle(idxs)

    # if the problem does not have correctIndex yet, set it
    if problem["correctIndex"] is None:
        c_idx = idxs.index(0)
        problem["correctIndex"] = c_idx
    else:
        c_idx = problem["correctIndex"]

    # we now know c_idx is the correct index
    # place the correct conv in that c_idx, etc.
    # but to show the same random shuffle each time, we do:
    #  actually let's do the same shuffle each time by seeding:
    # or simpler approach => we'll do a smaller random again, but
    # typically you'd store the shuffle in problem as well.
    # We'll just re-shuffle so that the correctIndex is c_idx => do a partial approach:

    # We'll build a new 'shuffled' array where the correct conv is at c_idx
    # and the others fill the remaining slots
    new_options = [None,None,None,None]
    new_options[c_idx] = correct_conv
    distracts = [dconv1, dconv2, dconv3]
    d_i = 0
    for j in range(4):
        if new_options[j] is None:
            new_options[j] = distracts[d_i]
            d_i+=1

    # Now create the figure
    fig = plt.figure(figsize=(8,8))
    gs = fig.add_gridspec(nrows=3,ncols=2,hspace=0.5,wspace=0.3)

    # top row => f1, f2
    axf1 = fig.add_subplot(gs[0,0])
    axf2 = fig.add_subplot(gs[0,1])
    axf1.plot(t,y1, color="blue")
    axf2.plot(t,y2, color="green")
    axf1.set_title(f"Input1: {f1_name}\nshift={shift1},scale={scale1}")
    axf2.set_title(f"Input2: {f2_name}\nshift={shift2},scale={scale2}")
    axf1.grid(True)
    axf2.grid(True)
    axf1.set_xlim(t[0],t[-1])
    axf2.set_xlim(t[0],t[-1])

    # bottom => 4 subplots
    ax_o1 = fig.add_subplot(gs[1,0])
    ax_o2 = fig.add_subplot(gs[1,1])
    ax_o3 = fig.add_subplot(gs[2,0])
    ax_o4 = fig.add_subplot(gs[2,1])
    option_axes = [ax_o1, ax_o2, ax_o3, ax_o4]

    # highlight logic if highlight_index != None
    for i, ax in enumerate(option_axes):
        color = "red"
        lw = 1.5
        if highlight_index is not None and i == highlight_index and i == c_idx:
            # user picked i, which is correct => let's do green
            color = "green"
            lw=3.0
        elif highlight_index is not None and i == highlight_index and i != c_idx:
            # user picked i incorrectly => red
            color = "red"
            lw=3.0
        elif highlight_index is not None and i == c_idx:
            # correct index is i, user didn't pick it => green
            color = "green"
            lw=2.5
        else:
            color="gray"
            lw=1.5

        ax.plot(t,new_options[i], color=color, linewidth=lw)
        ax.set_title(f"Option {i+1}")
        ax.grid(True)
        ax.set_xlim(t[0], t[-1])

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    return base64_img
