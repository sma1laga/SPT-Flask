"""
Blueprint and problem generator for the processing‑chain training module.

This module exposes a simple Flask blueprint that serves a template for the
processing chain training page and provides endpoints to generate practice
problems on demand.  Problems are generated at multiple difficulty levels;
this module currently implements the ``EASY`` and ``MEDIUM`` tiers.  Easy
problems consist of three serial operations (multiplication, Hilbert transform
and filtering) arranged in one of three predefined layouts.  Medium problems
extend this idea with a branching structure that introduces sampling and
derivative blocks alongside the original operations.  A random input spectrum
(rectangular or triangular) is chosen and the operations are parameterised
randomly.  The server then computes the correct frequency‑domain output after
each labelled connection as well as plausible distractors by varying one
parameter at a time.  Results are returned to the client as base64‑encoded PNG
images along with the correct answer index for each connection.

The client can display the diagram and the plots and let the user pick
their answer for every letter.  A simple check endpoint is also provided to
verify whether a given selection matches the stored correct index.

Attribution: this implementation is inspired by the existing convolution
training module in the repository.  The mathematical operations here are
implemented directly rather than relying on the existing chain_blocks
helpers because those helpers aren’t available in this standalone context.

"""

from __future__ import annotations

import io
import base64
import random
from typing import Dict, List, Tuple

import numpy as np
from flask import Blueprint, jsonify, render_template, request
import matplotlib

# Use a non‑interactive backend so that plots work without a display
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

_EDGE_COLOR = "#050505"
_BLOCK_LINEWIDTH = 1.85
_CONNECTOR_LINEWIDTH = 1.55
_ARROW_LINEWIDTH = 1.65
_SPLITTER_RADIUS = 0.08


def _arrow_props() -> Dict[str, object]:
    """Return a consistent arrow style for block diagrams."""

    return dict(arrowstyle="->", lw=_ARROW_LINEWIDTH, color=_EDGE_COLOR, shrinkA=0, shrinkB=0)

def _draw_connector(
    ax: plt.Axes,
    points: List[Tuple[float, float]],
    *,
    arrow_props: Dict[str, object],
    linewidth: float | None = None,
) -> None:
    """Draw a polyline connector that terminates in an arrow.

    The connector is defined by ``points`` representing the successive
    waypoints of the path.  Straight line segments are drawn between each
    waypoint using :func:`matplotlib.axes.Axes.plot`, while the final segment
    is rendered with :func:`matplotlib.axes.Axes.annotate` so that an arrow
    head is placed at the end of the path.  This helper keeps arrows aligned
    to horizontal/vertical segments which makes block diagrams easier to
    follow than the previous diagonal arrows.
    """

    if len(points) < 2:
        return

    line_width = linewidth if linewidth is not None else _CONNECTOR_LINEWIDTH
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=_EDGE_COLOR,
            lw=line_width,
        )

    final_start, final_end = points[-2], points[-1]
    final_props = dict(arrow_props)
    final_props.setdefault("lw", line_width)
    ax.annotate("", xy=final_end, xytext=final_start, arrowprops=final_props)



# Define a blueprint for the processing chain training.  Do not specify a URL
# prefix here – the main application registers this blueprint with a
# ``url_prefix`` of ``/training/processing_chain``.
training_processing_chain_bp = Blueprint(
    "training_processing_chain", __name__
)


@training_processing_chain_bp.route("/", methods=["GET"])
def training_processing_chain() -> str:
    """
    Serve the processing chain training page.

    This view mirrors the naming convention used by other training modules
    (for example the convolution training), where the function name
    matches the blueprint name.  Some templates and navigation links
    refer to the endpoint ``training_processing_chain`` directly, so
    keeping the function name consistent avoids ``url_for`` lookup errors.
    """
    return render_template("training_processing_chain.html")


@training_processing_chain_bp.route("/generate", methods=["POST"])
def generate_problem() -> Tuple[Dict[str, str], int]:
    """
    Generate a new processing chain problem.

    The client sends JSON with a ``difficulty`` field.  For the ``EASY`` and
    ``MEDIUM`` difficulties a random layout and random operation parameters are
    selected that match the respective block catalogue.
    The returned JSON has the following structure:

    ``diagram`` – a base64‑encoded PNG image of the block diagram.
    ``letters`` – a list of objects, one per labelled connection.  Each object
      contains the letter name, a list of three base64 image strings and the
      index (0,1,2) of the correct image in that list.

    If an unsupported difficulty is requested an error message is returned.
    """
    data = request.get_json(force=True) or {}
    difficulty: str = str(data.get("difficulty", "EASY")).upper()

    try:
        if difficulty == "EASY":
            problem = _create_easy_problem()
        elif difficulty == "MEDIUM":
            problem = _create_medium_problem()
        elif difficulty == "HARD":
            problem = _create_hard_problem()
        else:
            return (
                jsonify({"error": "Only EASY, MEDIUM and HARD difficulties are implemented."}),
                400,
            )
    except Exception as exc:
        return jsonify({"error": f"Failed to create problem: {exc}"}), 500
    return jsonify(problem)


@training_processing_chain_bp.route("/check_answer", methods=["POST"])
def check_answer() -> Tuple[Dict[str, str], int]:
    """
    Compare a user’s selected index against the correct index.

    The client should send JSON with ``selectedIndex`` and ``correctIndex``
    integers.  If they match a positive feedback string is returned,
    otherwise a generic try‑again message is sent.  This endpoint is kept
    simple to mirror the convolution training module and can be reused
    client‑side for each individual letter.
    """
    data = request.get_json(force=True) or {}
    sel = data.get("selectedIndex")
    cor = data.get("correctIndex")
    try:
        sel_int = int(sel)
        cor_int = int(cor)
    except Exception:
        return jsonify({"feedback": "Invalid data."}), 400
    if sel_int == cor_int:
        return jsonify({"feedback": "Correct!"})
    return jsonify({"feedback": "Incorrect. Try again!"})


def _create_easy_problem() -> Dict[str, object]:
    """
    Construct a single EASY difficulty processing chain problem

    This selects one of three simple layout patterns, randomly chooses a
    rectangular or triangular input spectrum with random amplitude and width,
    and assigns random parameters to the multiplication, Hilbert and filter
    blocks.  For each connection letter the correct output and two
    distractors are generated.  The entire diagram is also drawn as a
    separate image.

    Returns a dictionary ready for JSON serialisation as described in
    :func:`generate_problem`.
    """
    # freq. grid for evaluating spectra
    W = 10.0
    n_points = 1024
    w = np.linspace(-W, W, n_points)

    # 1) Choose input spectrum
    input_shape = random.choice(["rect", "tri"])
    amp_choices = [0.5, 1.0, 1.5, 2.0]
    width_choices = [1.0, 2.0, 3.0]
    amp = random.choice(amp_choices)
    width = random.choice(width_choices)
    if input_shape == "rect":
        base = _rect_spectrum(w, width)
    else:
        base = _tri_spectrum(w, width)
    x_sig = amp * base
    amp_text, amp_latex = _format_complex_gain(amp)
    input_expr = f"{amp_text}*{input_shape}(w/{width})"
    width_fmt = format(width, "g")
    operator_name = "rect" if input_shape == "rect" else "tri"
    latex_shape = (
        rf"\operatorname{{{operator_name}}}\left(\frac{{\omega}}{{{width_fmt}}}\right)"
    )
    input_expr_latex = f"{amp_latex} \\cdot {latex_shape}"

    # 2) Choose operations and layout
    # Define three layouts: sequences of operation names; letters always refer
    # to the output of the block at the same index (0→A, 1→B, 2→C).
    layouts: List[Tuple[str, str, str]] = [
        ("Multiplication", "Hilbert", "Filter"),
        ("Hilbert", "Multiplication", "Filter"),
        ("Multiplication", "Filter", "Hilbert"),
    ]
    op_sequence = random.choice(layouts)

    # Randomly parameterise each operation
    mul_param = _random_multiplication_param()
    hil_param = None  # Hilbert has no tunable parameter
    fil_param = _random_filter_param()

    # Map operation names to parameter values
    params = {
        "Multiplication": mul_param,
        "Hilbert": hil_param,
        "Filter": fil_param,
    }

    # Compute the correct spectra step by step and generate distractors
    letter_results: List[Dict[str, object]] = []
    signal_labels = ["a(t)", "b(t)", "c(t)"]
    signal_labels_latex = ["a(t)", "b(t)", "c(t)"]
    freq_titles = [r"$A(j\omega)$", r"$B(j\omega)$", r"$C(j\omega)$"]
    prev_sig = x_sig.copy()
    for i, op_name in enumerate(op_sequence):
        # Apply correct operation
        param = params[op_name]
        next_sig = _apply_operation(prev_sig, op_name, param, w)
        correct = next_sig.copy()

        # Generate two distractors by varying the current operation only
        distractors = []
        if op_name == "Multiplication":
            # two new multiplication params
            for _ in range(2):
                alt_param = _random_multiplication_param(exclude=mul_param)
                alt_sig = _apply_operation(prev_sig, op_name, alt_param, w)
                distractors.append(alt_sig)
        elif op_name == "Hilbert":
            # distractor 1: skip Hilbert entirely
            alt_sig1 = prev_sig.copy()
            # distractor 2: apply an incorrect Hilbert phase (sign error)
            alt_sig2 = _apply_incorrect_hilbert(prev_sig, w)
            distractors.extend([alt_sig1, alt_sig2])
        elif op_name == "Filter":
            # generate two different filters
            for _ in range(2):
                alt_param = _random_filter_param(exclude=fil_param)
                alt_sig = _apply_operation(prev_sig, "Filter", alt_param, w)
                distractors.append(alt_sig)
        else:
            # fallback: copy signal twice
            distractors.extend([correct.copy(), correct.copy()])

        # Shuffle options and determine correct index
        options = _ensure_option_diversity([correct] + distractors, w)
        indices = list(range(3))
        random.shuffle(indices)
        shuffled = [options[j] for j in indices]
        correct_index = indices.index(0)

        # Encode each option as base64 image
        encoded_imgs = [
            _plot_spectrum(w, sig, title=f"{freq_titles[i]}") for sig in shuffled
        ]
        letter_results.append(
            {
                "letter": chr(ord("A") + i),
                "signalLabel": signal_labels[i],
                "signalLabelLatex": signal_labels_latex[i],
                "images": encoded_imgs,
                "correctIndex": correct_index,
            }
        )
        prev_sig = next_sig

    # Build diagram metadata and image
    diagram_ops = []
    for idx, op_name in enumerate(op_sequence):
        letter = chr(ord("A") + idx)
        signal_name = f"{chr(ord('a') + idx)}(t)"
        signal_latex = signal_labels_latex[idx]
        name_latex = _operation_name_latex(op_name)
        param_latex = _operation_parameter_label_latex(op_name, params[op_name])
        summary_latex = (
            rf"\mathbf{{{letter}}}:\; {signal_latex}\; \text{{after}}\; {name_latex}"
        )
        if param_latex:
            summary_latex += rf"\;({param_latex})"
        diagram_ops.append(
            {
                "letter": letter,
                "signal": signal_name,
                "name": op_name,
                "parameter": _operation_parameter_label(op_name, params[op_name]),
                "signalLatex": signal_latex,
                "nameLatex": name_latex,
                "parameterLatex": param_latex,
                "summaryLatex": summary_latex,
            }
        )

    diagram_img = _draw_diagram(op_sequence, params)
    input_plot = _plot_spectrum(w, x_sig, title=r"$X(j\omega)$")


    return {
        "diagram": diagram_img,
        "diagramOperations": diagram_ops,
        "inputExpression": input_expr,
        "inputPlot": input_plot,
        # convert tuple to list for JSON serialisation
        "operations": list(op_sequence),
        "letters": letter_results,
        "inputExpressionLatex": input_expr_latex,
    }

def _create_medium_problem() -> Dict[str, object]:
    """Construct a MEDIUM difficulty processing chain problem."""

    W = 10.0
    n_points = 2048
    w = np.linspace(-W, W, n_points)

    # Input spectrum identical to EASY mode for consistency
    input_shape = random.choice(["rect", "tri"])
    amp = random.choice([0.5, 1.0, 1.5, 2.0])
    width = random.choice([1.0, 2.0, 3.0])
    base = _rect_spectrum(w, width) if input_shape == "rect" else _tri_spectrum(w, width)
    # Allow the medium difficulty input to be purely real or purely imaginary.
    phase_choice = random.choice([1.0, -1.0, 1j, -1j])
    complex_amp = phase_choice * amp
    coeff_text, coeff_latex = _format_complex_gain(complex_amp)
    x_sig = complex_amp * base
    input_expr = f"{coeff_text}*{input_shape}(w/{width})"
    width_fmt = format(width, "g")
    operator_name = "rect" if input_shape == "rect" else "tri"
    latex_shape = rf"\operatorname{{{operator_name}}}\left(\frac{{\omega}}{{{width_fmt}}}\right)"
    input_expr_latex = rf"{coeff_latex} \cdot {latex_shape}"
    input_plot = _plot_spectrum(w, x_sig, title=r"$X(j\omega)$")

    layout_builder = random.choice(
        [
            lambda: _build_medium_layout_sampling_branches(w, x_sig),
            lambda: _build_medium_layout_multiplication_split(w, x_sig),
        ]
    )
    layout_data = layout_builder()
    layout_data.update(
        {
            "inputExpression": input_expr,
            "inputExpressionLatex": input_expr_latex,
            "inputPlot": input_plot,
        }
    )
    return layout_data

def _create_hard_problem() -> Dict[str, object]:
    """Construct a HARD difficulty processing chain problem."""

    W = 12.0
    n_points = 4096
    w = np.linspace(-W, W, n_points)

    input_shape = random.choice(["rect", "tri"])
    width = random.choice([1.0, 1.5, 2.0, 3.0])
    base = _rect_spectrum(w, width) if input_shape == "rect" else _tri_spectrum(w, width)
    coeff = _random_complex_input_coeff()
    coeff_text, coeff_latex = _format_complex_gain(coeff)
    x_sig = coeff * base

    width_fmt = format(width, "g")
    operator_name = "rect" if input_shape == "rect" else "tri"
    latex_shape = rf"\operatorname{{{operator_name}}}\left(\frac{{\omega}}{{{width_fmt}}}\right)"
    input_expr = f"{coeff_text}*{input_shape}(w/{width_fmt})"
    input_expr_latex = rf"{coeff_latex} \cdot {latex_shape}"
    input_plot = _plot_spectrum(w, x_sig, title=r"$X(j\omega)$")

    layout_builder = random.choice(
        [
            lambda: _build_hard_layout_split_modulation(w, x_sig),
            lambda: _build_hard_layout_real_imag_mixer(w, x_sig),
            lambda: _build_hard_layout_real_imag_sampling_chain(w, x_sig),
            lambda: _build_hard_layout_complex_split_sampling(w, x_sig),
        ]
    )
    layout_data = layout_builder()
    layout_data.update(
        {
            "inputExpression": input_expr,
            "inputExpressionLatex": input_expr_latex,
            "inputPlot": input_plot,
        }
    )
    return layout_data



def _build_medium_layout_sampling_branches(
    w: np.ndarray, x_sig: np.ndarray
) -> Dict[str, object]:
    """Medium layout with two independent branches before a post-processing block."""

    branch_ops: List[Tuple[str, str | None]] = []
    branch_outputs: List[np.ndarray] = []
    branch_signals = ["a(t)", "b(t)"]
    branch_signals_latex = ["a(t)", "b(t)"]
    for _ in range(2):
        op_name = random.choice(["Multiplication", "Sampling"])
        if op_name == "Multiplication":
            param = _random_multiplication_param()
        else:
            param = _random_sampling_param()
        branch_ops.append((op_name, param))
        branch_outputs.append(_apply_operation(x_sig, op_name, param, w))

    # Combine via addition
    sum_sig = branch_outputs[0] + branch_outputs[1]

    # Post-addition block: derivative, Hilbert or another multiplication
    post_choices = ["Derivative", "Hilbert", "Multiplication"]
    post_op = random.choice(post_choices)
    if post_op == "Multiplication":
        post_param = _random_multiplication_param()
    else:
        post_param = None
    post_sig = _apply_operation(sum_sig, post_op, post_param, w)

    # Final filter
    filter_param = _random_filter_param()
    final_sig = _apply_operation(post_sig, "Filter", filter_param, w)

    # Prepare letters and spectra
    letter_results: List[Dict[str, object]] = []
    letter_specs = [
        ("A", branch_signals[0], branch_signals_latex[0], branch_outputs[0]),
        ("B", branch_signals[1], branch_signals_latex[1], branch_outputs[1]),
        ("C", "c(t)", "c(t)", sum_sig),
        ("D", "d(t)", "d(t)", post_sig),
        ("E", "e(t)", "e(t)", final_sig),
    ]
    freq_titles = {
        "A": r"$A(j\omega)$",
        "B": r"$B(j\omega)$",
        "C": r"$C(j\omega)$",
        "D": r"$D(j\omega)$",
        "E": r"$E(j\omega)$",
    }

    for idx, (letter, label, label_latex, correct_sig) in enumerate(letter_specs):
        if letter in ("A", "B"):
            op_name, param = branch_ops[idx]
            if op_name == "Multiplication":
                distractors = []
                for _ in range(2):
                    alt_param = _random_multiplication_param(exclude=param)
                    distractors.append(_apply_operation(x_sig, "Multiplication", alt_param, w))
            else:
                distractors = []
                for _ in range(2):
                    alt_param = _random_sampling_param(exclude=param)
                    distractors.append(_apply_operation(x_sig, "Sampling", alt_param, w))
        elif letter == "C":
            distractors = [
                branch_outputs[0] - branch_outputs[1],
                0.5 * (branch_outputs[0] + branch_outputs[1]),
            ]
        elif letter == "D":
            if post_op == "Derivative":
                distractors = [
                    1j * sum_sig,
                    -1j * w * sum_sig,
                ]
            elif post_op == "Hilbert":
                distractors = [sum_sig.copy(), _apply_incorrect_hilbert(sum_sig, w)]
            else:
                distractors = []
                for _ in range(2):
                    alt_param = _random_multiplication_param(exclude=post_param)
                    distractors.append(_apply_operation(sum_sig, "Multiplication", alt_param, w))
        else:  # letter == "E"
            distractors = []
            for _ in range(2):
                alt_param = _random_filter_param(exclude=filter_param)
                distractors.append(_apply_operation(post_sig, "Filter", alt_param, w))

        options = _ensure_option_diversity([correct_sig] + distractors, w)
        indices = list(range(len(options)))
        random.shuffle(indices)
        shuffled = [options[i] for i in indices]
        correct_index = indices.index(0)
        encoded_imgs = [
            _plot_spectrum(w, sig, title=freq_titles.get(letter, r"$S(j\omega)$"))
            for sig in shuffled
        ]
        letter_results.append(
            {
                "letter": letter,
                "signalLabel": label,
                "signalLabelLatex": label_latex,
                "images": encoded_imgs,
                "correctIndex": correct_index,
            }
        )

    diagram_ops = []
    for idx, letter in enumerate(["A", "B"]):
        op_name, param = branch_ops[idx]
        name_latex = _operation_name_latex(op_name)
        param_label = _operation_parameter_label(op_name, param)
        param_label_latex = _operation_parameter_label_latex(op_name, param)
        summary = rf"\mathbf{{{letter}}}:\; {branch_signals_latex[idx]}\; \text{{after}}\; {name_latex}"
        if param_label_latex:
            summary += rf"\;({param_label_latex})"
        diagram_ops.append(
            {
                "letter": letter,
                "signal": branch_signals[idx],
                "name": op_name,
                "parameter": param_label,
                "signalLatex": branch_signals_latex[idx],
                "nameLatex": name_latex,
                "parameterLatex": param_label_latex,
                "summaryLatex": summary,
            }
        )

    diagram_ops.append(
        {
            "letter": "C",
            "signal": "c(t)",
            "name": "Addition",
            "parameter": None,
            "signalLatex": "c(t)",
            "nameLatex": _operation_name_latex("Addition"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("C", "c(t)", "Addition", None),
        }
    )

    diagram_ops.append(
        {
            "letter": "D",
            "signal": "d(t)",
            "name": post_op,
            "parameter": _operation_parameter_label(post_op, post_param),
            "signalLatex": "d(t)",
            "nameLatex": _operation_name_latex(post_op),
            "parameterLatex": _operation_parameter_label_latex(post_op, post_param),
            "summaryLatex": _build_summary_latex("D", "d(t)", post_op, post_param),
        }
    )

    diagram_ops.append(
        {
            "letter": "E",
            "signal": "e(t)",
            "name": "Filter",
            "parameter": _operation_parameter_label("Filter", filter_param),
            "signalLatex": "e(t)",
            "nameLatex": _operation_name_latex("Filter"),
            "parameterLatex": _operation_parameter_label_latex("Filter", filter_param),
            "summaryLatex": _build_summary_latex("E", "e(t)", "Filter", filter_param),
        }
    )

    diagram_img = _draw_medium_diagram_sampling(branch_ops, post_op, post_param, filter_param)


    return {
        "diagram": diagram_img,
        "diagramOperations": diagram_ops,
        "operations": [
            {"type": op, "param": param} for op, param in branch_ops
        ]
        + [
            {"type": "Addition", "param": None},
            {"type": post_op, "param": post_param},
            {"type": "Filter", "param": filter_param},
        ],
        "letters": letter_results,
    }


def _build_medium_layout_multiplication_split(
    w: np.ndarray, x_sig: np.ndarray
) -> Dict[str, object]:
    """Medium layout with an initial multiplication followed by two branches."""

    mul_param = _random_multiplication_param()
    after_mul = _apply_operation(x_sig, "Multiplication", mul_param, w)

    branch_ops: List[Tuple[str, str | None]] = []
    branch_outputs: List[np.ndarray] = []
    branch_signals = ["b(t)", "c(t)"]
    branch_signals_latex = ["b(t)", "c(t)"]
    for _ in range(2):
        op_name = random.choice(["Hilbert", "Derivative"])
        param = None
        branch_ops.append((op_name, param))
        branch_outputs.append(_apply_operation(after_mul, op_name, param, w))

    sum_sig = branch_outputs[0] + branch_outputs[1]

    filter_param = _random_filter_param()
    final_sig = _apply_operation(sum_sig, "Filter", filter_param, w)

    letter_specs = [
        ("A", "a(t)", "a(t)", after_mul),
        ("B", branch_signals[0], branch_signals_latex[0], branch_outputs[0]),
        ("C", branch_signals[1], branch_signals_latex[1], branch_outputs[1]),
        ("D", "d(t)", "d(t)", sum_sig),
        ("E", "e(t)", "e(t)", final_sig),
    ]
    freq_titles = {
        "A": r"$A(j\omega)$",
        "B": r"$B(j\omega)$",
        "C": r"$C(j\omega)$",
        "D": r"$D(j\omega)$",
        "E": r"$E(j\omega)$",
    }

    letter_results: List[Dict[str, object]] = []
    for idx, (letter, label, label_latex, correct_sig) in enumerate(letter_specs):
        if letter == "A":
            distractors = []
            for _ in range(2):
                alt_param = _random_multiplication_param(exclude=mul_param)
                distractors.append(_apply_operation(x_sig, "Multiplication", alt_param, w))
        elif letter in ("B", "C"):
            branch_idx = idx - 1
            op_name, _ = branch_ops[branch_idx]
            if op_name == "Hilbert":
                distractors = [after_mul.copy(), _apply_incorrect_hilbert(after_mul, w)]
            else:  # Derivative
                distractors = [1j * after_mul, -1j * w * after_mul]
        elif letter == "D":
            distractors = [
                branch_outputs[0] - branch_outputs[1],
                0.5 * (branch_outputs[0] + branch_outputs[1]),
            ]
        else:  # letter == "E"
            distractors = []
            for _ in range(2):
                alt_param = _random_filter_param(exclude=filter_param)
                distractors.append(_apply_operation(sum_sig, "Filter", alt_param, w))

        options = _ensure_option_diversity([correct_sig] + distractors, w)
        indices = list(range(len(options)))
        random.shuffle(indices)
        shuffled = [options[i] for i in indices]
        correct_index = indices.index(0)
        encoded_imgs = [
            _plot_spectrum(w, sig, title=freq_titles.get(letter, r"$S(j\omega)$"))
            for sig in shuffled
        ]
        letter_results.append(
            {
                "letter": letter,
                "signalLabel": label,
                "signalLabelLatex": label_latex,
                "images": encoded_imgs,
                "correctIndex": correct_index,
            }
        )

    diagram_ops = [
        {
            "letter": "A",
            "signal": "a(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", mul_param),
            "signalLatex": "a(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", mul_param),
            "summaryLatex": _build_summary_latex("A", "a(t)", "Multiplication", mul_param),
        }
    ]

    for idx, letter in enumerate(["B", "C"]):
        op_name, param = branch_ops[idx]
        signal = branch_signals[idx]
        signal_latex = branch_signals_latex[idx]
        diagram_ops.append(
            {
                "letter": letter,
                "signal": signal,
                "name": op_name,
                "parameter": _operation_parameter_label(op_name, param),
                "signalLatex": signal_latex,
                "nameLatex": _operation_name_latex(op_name),
                "parameterLatex": _operation_parameter_label_latex(op_name, param),
                "summaryLatex": _build_summary_latex(letter, signal_latex, op_name, param),
            }
        )

    diagram_ops.append(
        {
            "letter": "D",
            "signal": "d(t)",
            "name": "Addition",
            "parameter": None,
            "signalLatex": "d(t)",
            "nameLatex": _operation_name_latex("Addition"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("D", "d(t)", "Addition", None),
        }
    )

    diagram_ops.append(
        {
            "letter": "E",
            "signal": "e(t)",
            "name": "Filter",
            "parameter": _operation_parameter_label("Filter", filter_param),
            "signalLatex": "e(t)",
            "nameLatex": _operation_name_latex("Filter"),
            "parameterLatex": _operation_parameter_label_latex("Filter", filter_param),
            "summaryLatex": _build_summary_latex("E", "e(t)", "Filter", filter_param),
        }
    )

    diagram_img = _draw_medium_diagram_multiplication_split(mul_param, branch_ops, filter_param)

    operations = [
        {"type": "Multiplication", "param": mul_param},
        {"type": branch_ops[0][0], "param": branch_ops[0][1]},
        {"type": branch_ops[1][0], "param": branch_ops[1][1]},
        {"type": "Addition", "param": None},
        {"type": "Filter", "param": filter_param},
    ]

    return {
        "diagram": diagram_img,
        "diagramOperations": diagram_ops,
        "operations": operations,
        "letters": letter_results,
    }


def _build_hard_layout_split_modulation(
    w: np.ndarray, x_sig: np.ndarray
) -> Dict[str, object]:
    """Hard layout with cascaded and parallel modulation stages."""

    mul_primary = _random_multiplication_param()
    sig_a = _apply_operation(x_sig, "Multiplication", mul_primary, w)

    sig_b = _apply_operation(sig_a, "Hilbert", None, w)
    sig_c = _apply_operation(sig_b, "Derivative", None, w)

    branch_param = _random_multiplication_param(exclude=mul_primary)
    sig_d = _apply_operation(sig_a, "Multiplication", branch_param, w)

    sum_sig = sig_c + sig_d

    filter_param = _random_filter_param()
    sig_f = _apply_operation(sum_sig, "Filter", filter_param, w)

    freq_titles = {
        "A": r"$A(j\omega)$",
        "B": r"$B(j\omega)$",
        "C": r"$C(j\omega)$",
        "D": r"$D(j\omega)$",
        "E": r"$E(j\omega)$",
        "F": r"$Y(j\omega)$",
    }

    letter_specs = [
        ("A", "a(t)", "a(t)", sig_a),
        ("B", "b(t)", "b(t)", sig_b),
        ("C", "c(t)", "c(t)", sig_c),
        ("D", "d(t)", "d(t)", sig_d),
        ("E", "e(t)", "e(t)", sum_sig),
        ("F", "y(t)", "y(t)", sig_f),
    ]

    letter_results: List[Dict[str, object]] = []
    for letter, label, label_latex, correct_sig in letter_specs:
        if letter == "A":
            distractors = []
            for _ in range(2):
                alt_param = _random_multiplication_param(exclude=mul_primary)
                distractors.append(_apply_operation(x_sig, "Multiplication", alt_param, w))
        elif letter == "B":
            distractors = [sig_a.copy(), _apply_incorrect_hilbert(sig_a, w)]
        elif letter == "C":
            distractors = [1j * sig_b, -1j * w * sig_b]
        elif letter == "D":
            distractors = []
            for _ in range(2):
                alt_param = _random_multiplication_param(exclude=branch_param)
                distractors.append(_apply_operation(sig_a, "Multiplication", alt_param, w))
        elif letter == "E":
            distractors = [sig_c - sig_d, 0.5 * (sig_c + sig_d)]
        else:  # F
            distractors = []
            for _ in range(2):
                alt_param = _random_filter_param(exclude=filter_param)
                distractors.append(_apply_operation(sum_sig, "Filter", alt_param, w))

        options = _ensure_option_diversity([correct_sig] + distractors, w)
        indices = list(range(len(options)))
        random.shuffle(indices)
        shuffled = [options[i] for i in indices]
        correct_index = indices.index(0)
        encoded_imgs = [
            _plot_spectrum(w, sig, title=freq_titles.get(letter, r"$S(j\omega)$"))
            for sig in shuffled
        ]
        letter_results.append(
            {
                "letter": letter,
                "signalLabel": label,
                "signalLabelLatex": label_latex,
                "images": encoded_imgs,
                "correctIndex": correct_index,
            }
        )

    diagram_ops = [
        {
            "letter": "A",
            "signal": "a(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", mul_primary),
            "signalLatex": "a(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", mul_primary),
            "summaryLatex": _build_summary_latex("A", "a(t)", "Multiplication", mul_primary),
        },
        {
            "letter": "B",
            "signal": "b(t)",
            "name": "Hilbert",
            "parameter": None,
            "signalLatex": "b(t)",
            "nameLatex": _operation_name_latex("Hilbert"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("B", "b(t)", "Hilbert", None),
        },
        {
            "letter": "C",
            "signal": "c(t)",
            "name": "Derivative",
            "parameter": None,
            "signalLatex": "c(t)",
            "nameLatex": _operation_name_latex("Derivative"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("C", "c(t)", "Derivative", None),
        },
        {
            "letter": "D",
            "signal": "d(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", branch_param),
            "signalLatex": "d(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", branch_param),
            "summaryLatex": _build_summary_latex("D", "d(t)", "Multiplication", branch_param),
        },
        {
            "letter": "E",
            "signal": "e(t)",
            "name": "Addition",
            "parameter": None,
            "signalLatex": "e(t)",
            "nameLatex": _operation_name_latex("Addition"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("E", "e(t)", "Addition", None),
        },
        {
            "letter": "F",
            "signal": "y(t)",
            "name": "Filter",
            "parameter": _operation_parameter_label("Filter", filter_param),
            "signalLatex": "y(t)",
            "nameLatex": _operation_name_latex("Filter"),
            "parameterLatex": _operation_parameter_label_latex("Filter", filter_param),
            "summaryLatex": _build_summary_latex("F", "y(t)", "Filter", filter_param),
        },
    ]

    diagram_img = _draw_hard_diagram_split_modulation(mul_primary, branch_param, filter_param)

    operations = [
        {"type": "Multiplication", "param": mul_primary},
        {"type": "Hilbert", "param": None},
        {"type": "Derivative", "param": None},
        {"type": "Multiplication", "param": branch_param},
        {"type": "Addition", "param": None},
        {"type": "Filter", "param": filter_param},
    ]

    return {
        "diagram": diagram_img,
        "diagramOperations": diagram_ops,
        "operations": operations,
        "letters": letter_results,
    }


def _draw_hard_diagram_split_modulation(
    mul_param: str | None, branch_param: str | None, filter_param: str | None
) -> str:
    """Draw the HARD layout with a cascaded Hilbert/derivative branch."""

    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    ax.axis("off")

    arrow_props = _arrow_props()

    y_mid = 1.6
    top_y = 2.8
    bot_y = 0.6
    input_x = 0.85
    mul_x = 2.35
    split_x = 3.55
    top_hilbert_x = 5.05
    top_derivative_x = 6.9
    bottom_mul_x = 5.45
    adder_x = 8.55
    filter_x = 10.55
    output_x = 11.95

    def prepare_block(info: Dict[str, object], centre: float, y_pos: float) -> Dict[str, object]:
        info = dict(info)
        info["centre"] = centre
        info["y"] = y_pos
        if info["shape"] == "circle":
            radius = info["radius"]
            info["left"] = centre - radius
            info["right"] = centre + radius
            info["top"] = y_pos + radius
            info["bottom"] = y_pos - radius
        else:
            width = info["width"]
            height = info["height"]
            info["left"] = centre - width / 2.0
            info["right"] = centre + width / 2.0
            info["top"] = y_pos + height / 2.0
            info["bottom"] = y_pos - height / 2.0
        return info

    mul_block = prepare_block(_block_render_info("Multiplication", mul_param), mul_x, y_mid)
    hilbert_block = prepare_block(_block_render_info("Hilbert", None), top_hilbert_x, top_y)
    derivative_block = prepare_block(_block_render_info("Derivative", None), top_derivative_x, top_y)
    bottom_block = prepare_block(_block_render_info("Multiplication", branch_param), bottom_mul_x, bot_y)
    adder_block = prepare_block(_block_render_info("Addition", None), adder_x, y_mid)
    filter_block = prepare_block(_block_render_info("Filter", filter_param), filter_x, y_mid)

    blocks = [mul_block, hilbert_block, derivative_block, bottom_block, adder_block, filter_block]

    def draw_block(block: Dict[str, object]) -> None:
        if block["shape"] == "circle":
            circle = plt.Circle((block["centre"], block["y"]), block["radius"], fill=False, lw=_BLOCK_LINEWIDTH, color=_EDGE_COLOR)
            ax.add_patch(circle)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                block["y"],
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 16),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                ax.annotate(
                    param_label,
                    xy=(block["centre"], block["y"] + block["radius"] * 0.35),
                    xytext=(block["centre"], block["y"] + block["radius"] + 0.6),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    arrowprops=_arrow_props(),
                )
        else:
            rect = plt.Rectangle(
                (block["left"], block["bottom"]),
                block["width"],
                block["height"],
                fill=False,
                lw=_BLOCK_LINEWIDTH,
                color=_EDGE_COLOR,
                joinstyle="round",
            )
            ax.add_patch(rect)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                block["y"] + block.get("label_y_offset", 0.18),
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 11),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                ax.text(
                    block["centre"],
                    block["y"] - block.get("param_y_offset", 0.18),
                    param_label,
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    for block in blocks:
        draw_block(block)

    # Input to multiplier
    ax.annotate("", xy=(mul_block["left"], y_mid), xytext=(input_x, y_mid), arrowprops=arrow_props)
    ax.text((input_x + mul_block["left"]) / 2.0, y_mid + 0.5, "$x(t)$", ha="center", va="center", fontsize=12)

    # Multiplier to split (letter A)
    ax.annotate("", xy=(split_x, y_mid), xytext=(mul_block["right"], y_mid), arrowprops=arrow_props)
    mid_a_x = (mul_block["right"] + split_x) / 2.0
    ax.text(mid_a_x, y_mid + 0.45, "A", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_a_x, y_mid - 0.6, "$a(t)$", ha="center", va="center", fontsize=11)

    split_circle = plt.Circle((split_x, y_mid), _SPLITTER_RADIUS, color=_EDGE_COLOR)
    ax.add_patch(split_circle)

    # Branch to top Hilbert
    ax.plot([split_x, split_x], [y_mid, top_y - 0.25], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.annotate("", xy=(hilbert_block["left"], top_y), xytext=(split_x, top_y), arrowprops=arrow_props)

    # Hilbert to derivative (letter B)
    ax.annotate("", xy=(derivative_block["left"], top_y), xytext=(hilbert_block["right"], top_y), arrowprops=arrow_props)
    mid_b_x = (hilbert_block["right"] + derivative_block["left"]) / 2.0
    ax.text(mid_b_x, top_y + 0.35, "B", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_b_x, top_y - 0.55, "$b(t)$", ha="center", va="center", fontsize=11)

    # Derivative to adder (letter C)
    top_route_x = derivative_block["right"] + 0.6
    connector_points = [
        (derivative_block["right"], top_y),
        (top_route_x, top_y),
        (top_route_x, y_mid + 0.55),
        (adder_block["left"], y_mid + 0.55),
    ]
    _draw_connector(ax, connector_points, arrow_props=arrow_props)
    mid_c_x = (top_route_x + adder_block["left"]) / 2.0
    ax.text(mid_c_x, y_mid + 0.9, "C", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_c_x, y_mid + 0.25, "$c(t)$", ha="center", va="center", fontsize=11)

    # Branch to bottom multiplier
    ax.plot([split_x, split_x], [y_mid, bot_y + 0.25], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.annotate("", xy=(bottom_block["left"], bot_y), xytext=(split_x, bot_y), arrowprops=arrow_props)

    # Bottom multiplier to adder (letter D)
    bottom_route_x = bottom_block["right"] + 0.6
    connector_points = [
        (bottom_block["right"], bot_y),
        (bottom_route_x, bot_y),
        (bottom_route_x, y_mid - 0.55),
        (adder_block["left"], y_mid - 0.55),
    ]
    _draw_connector(ax, connector_points, arrow_props=arrow_props)
    mid_d_x = (bottom_route_x + adder_block["left"]) / 2.0
    ax.text(mid_d_x, y_mid - 0.95, "D", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_d_x, y_mid - 1.45, "$d(t)$", ha="center", va="center", fontsize=11)

    # Adder to filter (letter E)
    ax.annotate("", xy=(filter_block["left"], y_mid), xytext=(adder_block["right"], y_mid), arrowprops=arrow_props)
    mid_e_x = (adder_block["right"] + filter_block["left"]) / 2.0
    ax.text(mid_e_x, y_mid + 0.4, "E", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_e_x, y_mid - 0.5, "$e(t)$", ha="center", va="center", fontsize=11)

    # Filter to output (letter F)
    ax.annotate("", xy=(output_x, y_mid), xytext=(filter_block["right"], y_mid), arrowprops=arrow_props)
    mid_f_x = (filter_block["right"] + output_x) / 2.0
    ax.text(mid_f_x, y_mid + 0.4, "F", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_f_x, y_mid - 0.5, "$y(t)$", ha="center", va="center", fontsize=11)

    ax.text(output_x + 0.6, y_mid, "$y(t)$", ha="center", va="center", fontsize=12)

    ax.set_xlim(0.35, output_x + 1.05)
    ax.set_ylim(bot_y - 0.6, top_y + 0.8)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=220)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _format_complex_gain(value: complex) -> Tuple[str, str]:
    """Return plain-text and LaTeX strings for a (possibly complex) gain."""

    real_part = float(np.real(value))
    imag_part = float(np.imag(value))
    tol = 1e-9
    real_zero = abs(real_part) < tol
    imag_zero = abs(imag_part) < tol

    if real_zero and imag_zero:
        return "0", "0"

    if imag_zero:
        scalar = format(real_part, "g")
        return scalar, scalar

    j_unit_text = "j"
    j_unit_latex = r"\mathrm{j}"
    imag_mag = abs(imag_part)
    if np.isclose(imag_mag, 1.0):
        magnitude = ""
    else:
        magnitude = format(imag_mag, "g")

    if real_zero:
        sign = "-" if imag_part < 0 else ""
        return f"{sign}{magnitude}{j_unit_text}", f"{sign}{magnitude}{j_unit_latex}"

    real_str = format(real_part, "g")
    sign = "+" if imag_part > 0 else "-"
    return (
        f"{real_str}{sign}{magnitude}{j_unit_text}",
        f"{real_str}{sign}{magnitude}{j_unit_latex}",
    )


def _random_complex_input_coeff() -> complex:
    """Return a random complex coefficient for HARD problems."""

    mode = random.choice(["real", "imag", "complex"])
    magnitudes = [0.5, 1.0, 1.5, 2.0]
    if mode == "real":
        return random.choice([1, -1]) * random.choice(magnitudes)
    if mode == "imag":
        return 1j * random.choice([1, -1]) * random.choice(magnitudes)
    real_part = random.choice([1, -1]) * random.choice(magnitudes)
    imag_part = random.choice([1, -1]) * random.choice(magnitudes)
    return complex(real_part, imag_part)



def _rect_spectrum(w: np.ndarray, width: float) -> np.ndarray:
    """Return a rectangular spectrum of the specified width."""
    return np.where(np.abs(w) <= width / 2.0, 1.0, 0.0)


def _tri_spectrum(w: np.ndarray, width: float) -> np.ndarray:
    """Return a triangular spectrum of the specified width."""
    half = width / 2.0
    return np.maximum(1.0 - np.abs(w) / half, 0.0)


def _random_multiplication_param(exclude: str | None = None) -> str:
    """
    Choose a random multiplication parameter string.  Optionally exclude
    ``exclude`` from the draw to avoid returning the same value twice.

    @Paul here highly adaptable, thought this makes sense? Adapt if needed..

    Supported formats (drawn uniformly):
      * ``constant:K`` with K ∈ {±0.5, ±1, ±1.5, ±2}
      * ``imaginary[:K]`` where K ∈ {0.5, 1, 1.5, 2}
      * ``linear:A`` with A ∈ {±0.5, ±1, ±2}
      * ``sin:A,w0`` (always real) with A ∈ {0.5,1,1.5}, w0 ∈ {1,2,3}
      * ``cos:A,w0`` same parameter ranges as sin
      * ``exponential:K,sign,w0`` with K ∈ {0.5,1,2}, sign ∈ {+,-}, w0 ∈ {1,2,3}
    """
    choices = []
    # constant
    for K in [0.5, 1, 1.5, 2]:
        for sgn in [1, -1]:
            choices.append(f"constant:{sgn*K}")
    # imaginary
    for K in [0.5, 1, 1.5, 2]:
        choices.append(f"imaginary:{K}")
    # linear
    for A in [0.5, 1.0, 2.0]:
        for sgn in [1, -1]:
            choices.append(f"linear:{sgn*A}")
    # sin and cos
    # note: avoid using unicode variable names (e.g. ω0) to prevent syntax
    # errors on some systems.  Use the ASCII identifier w0 instead.
    for func in ["sin", "cos"]:
        for A in [0.5, 1.0, 1.5]:
            for w0 in [1, 2, 3]:
                choices.append(f"{func}:{A},{w0}")
                choices.append(f"{func}:j,{A},{w0}")
    # exponential
    for K in [0.5, 1.0, 2.0]:
        for sign in ["+", "-"]:
            for w0 in [1, 2, 3]:
                choices.append(f"exponential:{K},{sign},{w0}")

    if exclude is not None and exclude in choices:
        choices = [c for c in choices if c != exclude]
    return random.choice(choices)


def _random_filter_param(exclude: str | None = None) -> str:
    """
    Choose a random filter parameter string.  Optionally exclude a given
    parameter from the draw.

    Supported filters:
      * ``lowpass:c`` with c ∈ {1, 2, 3}
      * ``highpass:c`` with c ∈ {1, 2, 3}
      * ``bandpass:lo,hi`` with lo,hi chosen from {1,2,3,4} and lo < hi
    """
    options: List[str] = []
    for c in [1.0, 2.0, 3.0]:
        options.append(f"lowpass:{c}")
        options.append(f"highpass:{c}")
    # bandpass: choose from [1,2,3,4] for (lo, hi) such that lo < hi
    bp_vals = [1.0, 2.0, 3.0, 4.0]
    for lo in bp_vals:
        for hi in bp_vals:
            if lo < hi:
                options.append(f"bandpass:{lo},{hi}")
    if exclude is not None and exclude in options:
        options = [o for o in options if o != exclude]
    return random.choice(options)

def _random_sampling_param(exclude: str | None = None) -> str:
    """Return a random sampling parameter string of the form ``sampling:T``."""

    choices = [f"sampling:{T}" for T in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]]
    if exclude is not None and exclude in choices:
        choices = [c for c in choices if c != exclude]
    return random.choice(choices)



def _apply_operation(signal: np.ndarray, op_name: str, param: str | None, w: np.ndarray) -> np.ndarray:
    """
    Apply a frequency‑domain operation to the signal ``signal``.

    ``op_name`` must be one of ``Multiplication``, ``Hilbert`` or ``Filter``.
    ``param`` should be a string encoded according to the rules described
    above.  For ``Hilbert`` the parameter is ignored.  This function
    interprets the parameter and applies the appropriate spectral transform.
    """
    if op_name == "Multiplication":
        return _apply_multiplication(signal, param, w)
    elif op_name == "Sampling":
        return _apply_sampling(signal, param, w)
    elif op_name == "Hilbert":
        return _apply_hilbert(signal, w)
    elif op_name == "Derivative":
        return _apply_derivative(signal, w)
    elif op_name == "Real":
        return _apply_real(signal)
    elif op_name == "Imag":
        return _apply_imag(signal)
    elif op_name == "Filter":
        return _apply_filter(signal, param, w)
    raise ValueError(f"Unsupported operation: {op_name}")


def _apply_multiplication(signal: np.ndarray, param: str | None, w: np.ndarray) -> np.ndarray:
    """
    Multiply the spectrum by the factor described in ``param``.

    This mirrors (in a simplified way) the ``apply_multiplication`` function
    from the project’s chain_transforms module.  Several categories are
    recognised.  See :func:`_random_multiplication_param` for supported
    formats.
    """
    if not param:
        return signal
    p = param.strip().lower()
    # constant
    if p.startswith("constant:"):
        K = float(p.split(":")[1])
        return K * signal
    # imaginary
    if p.startswith("imaginary"):
        parts = p.split(":")
        K = float(parts[1]) if len(parts) > 1 else 1.0
        return 1j * K * signal
    # linear
    if p.startswith("linear:"):
        A = float(p.split(":")[1])
        return A * w * signal
    # sin/cos
    if p.startswith("sin:") or p.startswith("cos:"):
        is_sin = p.startswith("sin:")
        raw_tokens = [t.strip() for t in p.split(":", 1)[1].split(",") if t.strip()]
        jflag = False
        if raw_tokens and raw_tokens[0] == "j":
            jflag = True
            raw_tokens = raw_tokens[1:]
        A = float(raw_tokens[0]) if raw_tokens else 1.0
        # parse the frequency shift value (use ASCII name w0 instead of unicode omega)
        w0 = float(raw_tokens[1]) if len(raw_tokens) > 1 else 1.0
        # implement spectral shift for multiplication by sin/cos in time
        # Y(jω) = A/2 * [X(j(ω-w0)) ± X(j(ω+w0))]
        # sin uses difference with factor 1/(2j); cos uses sum with factor 1/2
        dw = w[1] - w[0]
        # compute integer shift in bins
        shift_bins = int(round(w0 / dw))
        # shift helper (wrap around using np.roll)
        def shift(arr: np.ndarray, bins: int) -> np.ndarray:
            return np.roll(arr, -bins)
        x_p = shift(signal, shift_bins)
        x_m = shift(signal, -shift_bins)
        if is_sin:
            y = (A / (2j)) * (x_p - x_m)
        else:
            y = (A / 2.0) * (x_p + x_m)
        return 1j * y if jflag else y
    # exponential
    if p.startswith("exponential:"):
        _, rest = p.split(":")
        items = rest.split(",")
        if len(items) == 3:
            K = float(items[0])
            sign = items[1].strip()
            w0 = float(items[2])
        else:
            K = 1.0
            sign = items[0].strip()
            w0 = float(items[1])
        sign_val = +1.0 if sign.startswith("+") else -1.0
        dw = w[1] - w[0]
        bins = int(round(sign_val * w0 / dw))
        y = np.roll(signal, -bins)
        return K * y
    # fallback: treat parameter as a Python expression in w
    safe = {"np": np, "w": w, "j": 1j}
    try:
        factor = eval(param, {"__builtins__": {}}, safe)
        return signal * factor
    except Exception:
        return signal

def _apply_sampling(signal: np.ndarray, param: str | None, w: np.ndarray) -> np.ndarray:
    """Apply an ideal sampling comb with spacing ``T`` in frequency."""

    T = 1.0
    if param:
        p = param.strip().lower()
        if p.startswith("sampling:"):
            try:
                T = float(p.split(":", 1)[1])
            except ValueError:
                T = 1.0
    T = max(abs(T), 1e-3)
    tol = T / 10.0
    comb_centres = np.round(w / T) * T
    mask = np.where(np.abs(w - comb_centres) < tol, 1.0 / T, 0.0)
    return signal * mask



def _apply_hilbert(signal: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply a Hilbert transform in the frequency domain."""
    # Hilbert transform: multiply by -j*sign(w)
    sign_w = np.sign(w)
    return -1j * sign_w * signal


def _apply_derivative(signal: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Differentiate a signal in the frequency domain."""

    return 1j * w * signal


def _apply_real(signal: np.ndarray) -> np.ndarray:
    """Return the real component of a spectrum."""

    return np.real(signal)


def _apply_imag(signal: np.ndarray) -> np.ndarray:
    """Return the imaginary component of a spectrum."""

    return np.imag(signal)


def _apply_incorrect_hilbert(signal: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return a plausible but incorrect Hilbert transform variant."""

    sign_w = np.sign(w)
    # flip the sign convention to emulate a common mistake
    return 1j * sign_w * signal



def _apply_filter(signal: np.ndarray, param: str | None, w: np.ndarray) -> np.ndarray:
    """
    Apply a simple ideal filter defined by ``param``.

    ``param`` has the form ``mode:values`` where mode is one of
    ``lowpass``, ``highpass`` or ``bandpass``.  Values are parsed as
    comma‑separated floats.  The returned signal is multiplied by a mask
    that implements the desired magnitude response.
    """
    if not param:
        param = "lowpass:1"  # default
    p = param.lower()
    mode, rest = p.split(":")
    # magnitude mask initialised to zeros
    mask = np.zeros_like(signal, dtype=float)
    if mode == "lowpass":
        c = float(rest)
        mask = np.abs(w) <= c
    elif mode == "highpass":
        c = float(rest)
        mask = np.abs(w) >= c
    elif mode == "bandpass":
        # bandpass: lo,hi
        lo_str, hi_str = rest.split(",")
        lo = float(lo_str)
        hi = float(hi_str)
        mask = (np.abs(w) >= lo) & (np.abs(w) <= hi)
    else:
        mask = np.ones_like(w, dtype=float)
    return signal * mask


def _plot_spectrum(w: np.ndarray, sig: np.ndarray, title: str) -> str:
    """
    Plot the real and imaginary parts of ``sig`` versus ``w`` and return a
    base64 encoded PNG.  Axis limits are chosen based on the maximum
    amplitude in the visible range +-5 for readability.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(w, np.real(sig), label="Re", color="tab:blue", lw=1.5)
    if np.any(np.abs(np.imag(sig)) > 1e-10):
        ax.plot(w, np.imag(sig), linestyle="dotted", label="Im", color="tab:orange", lw=1.5)
    ax.set_xlabel(r"Frequency $\omega$")
    ax.set_title(title)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.legend()
    # limit frequency to +-5 for clarity
    mask = np.abs(w) <= 5
    # compute y limits based on visible region
    visible = sig[mask]
    y_abs_max = np.max(np.abs(np.concatenate([np.real(visible), np.imag(visible)])))
    ylim = max(y_abs_max * 1.2, 0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-ylim, ylim)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _build_hard_layout_real_imag_mixer(
    w: np.ndarray, x_sig: np.ndarray
) -> Dict[str, object]:
    """Hard layout featuring sampling, real/imag splits and recombination."""

    mul_primary = _random_multiplication_param()
    sig_a = _apply_operation(x_sig, "Multiplication", mul_primary, w)

    sample_param = _random_sampling_param()
    sig_b = _apply_operation(sig_a, "Sampling", sample_param, w)
    sig_c = _apply_operation(sig_b, "Real", None, w)
    top_mul_param = _random_multiplication_param()
    sig_d = _apply_operation(sig_c, "Multiplication", top_mul_param, w)

    sig_e = _apply_operation(sig_a, "Hilbert", None, w)
    sig_f = _apply_operation(sig_e, "Imag", None, w)
    bottom_mul_param = _random_multiplication_param()
    sig_g = _apply_operation(sig_f, "Multiplication", bottom_mul_param, w)

    sum_sig = sig_d + sig_g

    filter_param = _random_filter_param()
    sig_i = _apply_operation(sum_sig, "Filter", filter_param, w)

    freq_titles = {
        "A": r"$A(j\omega)$",
        "B": r"$B(j\omega)$",
        "C": r"$C(j\omega)$",
        "D": r"$D(j\omega)$",
        "E": r"$E(j\omega)$",
        "F": r"$F(j\omega)$",
        "G": r"$G(j\omega)$",
        "H": r"$H(j\omega)$",
        "I": r"$Y(j\omega)$",
    }

    letter_specs = [
        ("A", "a(t)", "a(t)", sig_a),
        ("B", "b(t)", "b(t)", sig_b),
        ("C", "c(t)", "c(t)", sig_c),
        ("D", "d(t)", "d(t)", sig_d),
        ("E", "e(t)", "e(t)", sig_e),
        ("F", "f(t)", "f(t)", sig_f),
        ("G", "g(t)", "g(t)", sig_g),
        ("H", "h(t)", "h(t)", sum_sig),
        ("I", "y(t)", "y(t)", sig_i),
    ]

    letter_results: List[Dict[str, object]] = []
    for letter, label, label_latex, correct_sig in letter_specs:
        if letter == "A":
            distractors = []
            for _ in range(2):
                alt_param = _random_multiplication_param(exclude=mul_primary)
                distractors.append(_apply_operation(x_sig, "Multiplication", alt_param, w))
        elif letter == "B":
            distractors = []
            for _ in range(2):
                alt_param = _random_sampling_param(exclude=sample_param)
                distractors.append(_apply_operation(sig_a, "Sampling", alt_param, w))
        elif letter == "C":
            distractors = [sig_b.copy(), np.imag(sig_b)]
        elif letter == "D":
            distractors = []
            for _ in range(2):
                alt_param = _random_multiplication_param(exclude=top_mul_param)
                distractors.append(_apply_operation(sig_c, "Multiplication", alt_param, w))
        elif letter == "E":
            distractors = [sig_a.copy(), _apply_incorrect_hilbert(sig_a, w)]
        elif letter == "F":
            distractors = [np.real(sig_e), -np.imag(sig_e)]
        elif letter == "G":
            distractors = []
            for _ in range(2):
                alt_param = _random_multiplication_param(exclude=bottom_mul_param)
                distractors.append(_apply_operation(sig_f, "Multiplication", alt_param, w))
        elif letter == "H":
            distractors = [sig_d - sig_g, 0.5 * (sig_d + sig_g)]
        else:  # I
            distractors = []
            for _ in range(2):
                alt_param = _random_filter_param(exclude=filter_param)
                distractors.append(_apply_operation(sum_sig, "Filter", alt_param, w))

        options = _ensure_option_diversity([correct_sig] + distractors, w)
        indices = list(range(len(options)))
        random.shuffle(indices)
        shuffled = [options[i] for i in indices]
        correct_index = indices.index(0)
        encoded_imgs = [
            _plot_spectrum(w, sig, title=freq_titles.get(letter, r"$S(j\omega)$"))
            for sig in shuffled
        ]
        letter_results.append(
            {
                "letter": letter,
                "signalLabel": label,
                "signalLabelLatex": label_latex,
                "images": encoded_imgs,
                "correctIndex": correct_index,
            }
        )

    diagram_ops = [
        {
            "letter": "A",
            "signal": "a(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", mul_primary),
            "signalLatex": "a(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", mul_primary),
            "summaryLatex": _build_summary_latex("A", "a(t)", "Multiplication", mul_primary),
        },
        {
            "letter": "B",
            "signal": "b(t)",
            "name": "Sampling",
            "parameter": _operation_parameter_label("Sampling", sample_param),
            "signalLatex": "b(t)",
            "nameLatex": _operation_name_latex("Sampling"),
            "parameterLatex": _operation_parameter_label_latex("Sampling", sample_param),
            "summaryLatex": _build_summary_latex("B", "b(t)", "Sampling", sample_param),
        },
        {
            "letter": "C",
            "signal": "c(t)",
            "name": "Real",
            "parameter": None,
            "signalLatex": "c(t)",
            "nameLatex": _operation_name_latex("Real"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("C", "c(t)", "Real", None),
        },
        {
            "letter": "D",
            "signal": "d(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", top_mul_param),
            "signalLatex": "d(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", top_mul_param),
            "summaryLatex": _build_summary_latex("D", "d(t)", "Multiplication", top_mul_param),
        },
        {
            "letter": "E",
            "signal": "e(t)",
            "name": "Hilbert",
            "parameter": None,
            "signalLatex": "e(t)",
            "nameLatex": _operation_name_latex("Hilbert"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("E", "e(t)", "Hilbert", None),
        },
        {
            "letter": "F",
            "signal": "f(t)",
            "name": "Imag",
            "parameter": None,
            "signalLatex": "f(t)",
            "nameLatex": _operation_name_latex("Imag"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("F", "f(t)", "Imag", None),
        },
        {
            "letter": "G",
            "signal": "g(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", bottom_mul_param),
            "signalLatex": "g(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", bottom_mul_param),
            "summaryLatex": _build_summary_latex("G", "g(t)", "Multiplication", bottom_mul_param),
        },
        {
            "letter": "H",
            "signal": "h(t)",
            "name": "Addition",
            "parameter": None,
            "signalLatex": "h(t)",
            "nameLatex": _operation_name_latex("Addition"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("H", "h(t)", "Addition", None),
        },
        {
            "letter": "I",
            "signal": "y(t)",
            "name": "Filter",
            "parameter": _operation_parameter_label("Filter", filter_param),
            "signalLatex": "y(t)",
            "nameLatex": _operation_name_latex("Filter"),
            "parameterLatex": _operation_parameter_label_latex("Filter", filter_param),
            "summaryLatex": _build_summary_latex("I", "y(t)", "Filter", filter_param),
        },
    ]

    diagram_img = _draw_hard_diagram_real_imag(
        mul_primary,
        sample_param,
        top_mul_param,
        bottom_mul_param,
        filter_param,
    )

    operations = [
        {"type": "Multiplication", "param": mul_primary},
        {"type": "Sampling", "param": sample_param},
        {"type": "Real", "param": None},
        {"type": "Multiplication", "param": top_mul_param},
        {"type": "Hilbert", "param": None},
        {"type": "Imag", "param": None},
        {"type": "Multiplication", "param": bottom_mul_param},
        {"type": "Addition", "param": None},
        {"type": "Filter", "param": filter_param},
    ]

    return {
        "diagram": diagram_img,
        "diagramOperations": diagram_ops,
        "operations": operations,
        "letters": letter_results,
    }

def _build_hard_layout_real_imag_sampling_chain(
    w: np.ndarray, x_sig: np.ndarray
) -> Dict[str, object]:
    """Hard layout that splits into real/imag paths before sampling and filtering."""

    mul_primary = _random_multiplication_param()
    sig_a = _apply_operation(x_sig, "Multiplication", mul_primary, w)

    sig_b = _apply_operation(sig_a, "Real", None, w)
    top_mul_param = _random_multiplication_param()
    sig_c = _apply_operation(sig_b, "Multiplication", top_mul_param, w)

    sig_d = _apply_operation(sig_a, "Imag", None, w)
    bottom_mul_param = _random_multiplication_param()
    sig_e = _apply_operation(sig_d, "Multiplication", bottom_mul_param, w)

    sum_sig = sig_c + sig_e

    sample_param = _random_sampling_param()
    sig_g = _apply_operation(sum_sig, "Sampling", sample_param, w)

    filter_param = _random_filter_param()
    sig_h = _apply_operation(sig_g, "Filter", filter_param, w)

    freq_titles = {
        "A": r"$A(j\omega)$",
        "B": r"$B(j\omega)$",
        "C": r"$C(j\omega)$",
        "D": r"$D(j\omega)$",
        "E": r"$E(j\omega)$",
        "F": r"$F(j\omega)$",
        "G": r"$G(j\omega)$",
        "H": r"$Y(j\omega)$",
    }

    letter_specs = [
        ("A", "a(t)", "a(t)", sig_a),
        ("B", "b(t)", "b(t)", sig_b),
        ("C", "c(t)", "c(t)", sig_c),
        ("D", "d(t)", "d(t)", sig_d),
        ("E", "e(t)", "e(t)", sig_e),
        ("F", "f(t)", "f(t)", sum_sig),
        ("G", "g(t)", "g(t)", sig_g),
        ("H", "y(t)", "y(t)", sig_h),
    ]

    letter_results: List[Dict[str, object]] = []
    for letter, label, label_latex, correct_sig in letter_specs:
        if letter == "A":
            distractors = [
                _apply_operation(
                    x_sig,
                    "Multiplication",
                    _random_multiplication_param(exclude=mul_primary),
                    w,
                )
                for _ in range(2)
            ]
        elif letter == "B":
            distractors = [sig_a.copy(), np.imag(sig_a)]
        elif letter == "C":
            distractors = [
                _apply_operation(
                    sig_b,
                    "Multiplication",
                    _random_multiplication_param(exclude=top_mul_param),
                    w,
                )
                for _ in range(2)
            ]
        elif letter == "D":
            distractors = [sig_a.copy(), np.real(sig_a)]
        elif letter == "E":
            distractors = [
                _apply_operation(
                    sig_d,
                    "Multiplication",
                    _random_multiplication_param(exclude=bottom_mul_param),
                    w,
                )
                for _ in range(2)
            ]
        elif letter == "F":
            distractors = [sig_c - sig_e, 0.5 * (sig_c + sig_e)]
        elif letter == "G":
            distractors = [
                _apply_operation(
                    sum_sig,
                    "Sampling",
                    _random_sampling_param(exclude=sample_param),
                    w,
                )
                for _ in range(2)
            ]
        else:  # letter == "H"
            distractors = [
                _apply_operation(
                    sig_g,
                    "Filter",
                    _random_filter_param(exclude=filter_param),
                    w,
                )
                for _ in range(2)
            ]

        options = _ensure_option_diversity([correct_sig] + distractors, w)
        indices = list(range(len(options)))
        random.shuffle(indices)
        shuffled = [options[i] for i in indices]
        correct_index = indices.index(0)

        encoded_imgs = [
            _plot_spectrum(w, sig, title=freq_titles.get(letter, r"$S(j\omega)$"))
            for sig in shuffled
        ]
        letter_results.append(
            {
                "letter": letter,
                "signalLabel": label,
                "signalLabelLatex": label_latex,
                "images": encoded_imgs,
                "correctIndex": correct_index,
            }
        )

    diagram_ops = [
        {
            "letter": "A",
            "signal": "a(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", mul_primary),
            "signalLatex": "a(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", mul_primary),
            "summaryLatex": _build_summary_latex("A", "a(t)", "Multiplication", mul_primary),
        },
        {
            "letter": "B",
            "signal": "b(t)",
            "name": "Real",
            "parameter": None,
            "signalLatex": "b(t)",
            "nameLatex": _operation_name_latex("Real"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("B", "b(t)", "Real", None),
        },
        {
            "letter": "C",
            "signal": "c(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", top_mul_param),
            "signalLatex": "c(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", top_mul_param),
            "summaryLatex": _build_summary_latex("C", "c(t)", "Multiplication", top_mul_param),
        },
        {
            "letter": "D",
            "signal": "d(t)",
            "name": "Imag",
            "parameter": None,
            "signalLatex": "d(t)",
            "nameLatex": _operation_name_latex("Imag"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("D", "d(t)", "Imag", None),
        },
        {
            "letter": "E",
            "signal": "e(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", bottom_mul_param),
            "signalLatex": "e(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", bottom_mul_param),
            "summaryLatex": _build_summary_latex("E", "e(t)", "Multiplication", bottom_mul_param),
        },
        {
            "letter": "F",
            "signal": "f(t)",
            "name": "Addition",
            "parameter": None,
            "signalLatex": "f(t)",
            "nameLatex": _operation_name_latex("Addition"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("F", "f(t)", "Addition", None),
        },
        {
            "letter": "G",
            "signal": "g(t)",
            "name": "Sampling",
            "parameter": _operation_parameter_label("Sampling", sample_param),
            "signalLatex": "g(t)",
            "nameLatex": _operation_name_latex("Sampling"),
            "parameterLatex": _operation_parameter_label_latex("Sampling", sample_param),
            "summaryLatex": _build_summary_latex("G", "g(t)", "Sampling", sample_param),
        },
        {
            "letter": "H",
            "signal": "y(t)",
            "name": "Filter",
            "parameter": _operation_parameter_label("Filter", filter_param),
            "signalLatex": "y(t)",
            "nameLatex": _operation_name_latex("Filter"),
            "parameterLatex": _operation_parameter_label_latex("Filter", filter_param),
            "summaryLatex": _build_summary_latex("H", "y(t)", "Filter", filter_param),
        },
    ]

    diagram_img = _draw_hard_diagram_real_imag_sampling(
        mul_primary,
        top_mul_param,
        bottom_mul_param,
        sample_param,
        filter_param,
    )

    operations = [
        {"type": "Multiplication", "param": mul_primary},
        {"type": "Real", "param": None},
        {"type": "Multiplication", "param": top_mul_param},
        {"type": "Imag", "param": None},
        {"type": "Multiplication", "param": bottom_mul_param},
        {"type": "Addition", "param": None},
        {"type": "Sampling", "param": sample_param},
        {"type": "Filter", "param": filter_param},
    ]

    return {
        "diagram": diagram_img,
        "diagramOperations": diagram_ops,
        "operations": operations,
        "letters": letter_results,
    }


def _build_hard_layout_complex_split_sampling(
    w: np.ndarray, x_sig: np.ndarray
) -> Dict[str, object]:
    """Hard layout with dual multiplications, a Hilbert/derivative stage and sampling."""

    mul_primary = _random_multiplication_param()
    sig_a = _apply_operation(x_sig, "Multiplication", mul_primary, w)

    transform = random.choice(["Hilbert", "Derivative"])
    sig_b = _apply_operation(sig_a, transform, None, w)

    mul_secondary = _random_multiplication_param()
    sig_c = _apply_operation(sig_b, "Multiplication", mul_secondary, w)

    sig_d = _apply_operation(sig_c, "Real", None, w)
    sig_e = _apply_operation(sig_c, "Imag", None, w)
    sum_sig = sig_d + sig_e

    sample_param = _random_sampling_param()
    sig_g = _apply_operation(sum_sig, "Sampling", sample_param, w)

    filter_param = _random_filter_param()
    sig_h = _apply_operation(sig_g, "Filter", filter_param, w)

    freq_titles = {
        "A": r"$A(j\omega)$",
        "B": r"$B(j\omega)$",
        "C": r"$C(j\omega)$",
        "D": r"$D(j\omega)$",
        "E": r"$E(j\omega)$",
        "F": r"$F(j\omega)$",
        "G": r"$G(j\omega)$",
        "H": r"$Y(j\omega)$",
    }

    letter_specs = [
        ("A", "a(t)", "a(t)", sig_a),
        ("B", "b(t)", "b(t)", sig_b),
        ("C", "c(t)", "c(t)", sig_c),
        ("D", "d(t)", "d(t)", sig_d),
        ("E", "e(t)", "e(t)", sig_e),
        ("F", "f(t)", "f(t)", sum_sig),
        ("G", "g(t)", "g(t)", sig_g),
        ("H", "y(t)", "y(t)", sig_h),
    ]

    letter_results: List[Dict[str, object]] = []
    for letter, label, label_latex, correct_sig in letter_specs:
        if letter == "A":
            distractors = []
            for _ in range(2):
                alt_param = _random_multiplication_param(exclude=mul_primary)
                distractors.append(_apply_operation(x_sig, "Multiplication", alt_param, w))
        elif letter == "B":
            if transform == "Hilbert":
                distractors = [sig_a.copy(), _apply_incorrect_hilbert(sig_a, w)]
            else:
                distractors = [1j * sig_a, -1j * w * sig_a]
        elif letter == "C":
            distractors = []
            for _ in range(2):
                alt_param = _random_multiplication_param(exclude=mul_secondary)
                distractors.append(_apply_operation(sig_b, "Multiplication", alt_param, w))
        elif letter == "D":
            distractors = [sig_c.copy(), np.imag(sig_c)]
        elif letter == "E":
            distractors = [sig_c.copy(), np.real(sig_c)]
        elif letter == "F":
            distractors = [sig_d - sig_e, 0.5 * (sig_d + sig_e)]
        elif letter == "G":
            distractors = []
            for _ in range(2):
                alt_param = _random_sampling_param(exclude=sample_param)
                distractors.append(_apply_operation(sum_sig, "Sampling", alt_param, w))
        else:  # letter == "H"
            distractors = []
            for _ in range(2):
                alt_param = _random_filter_param(exclude=filter_param)
                distractors.append(_apply_operation(sig_g, "Filter", alt_param, w))

        options = _ensure_option_diversity([correct_sig] + distractors, w)
        indices = list(range(len(options)))
        random.shuffle(indices)
        shuffled = [options[i] for i in indices]
        correct_index = indices.index(0)

        encoded_imgs = [
            _plot_spectrum(w, sig, title=freq_titles.get(letter, r"$S(j\omega)$"))
            for sig in shuffled
        ]
        letter_results.append(
            {
                "letter": letter,
                "signalLabel": label,
                "signalLabelLatex": label_latex,
                "images": encoded_imgs,
                "correctIndex": correct_index,
            }
        )

    diagram_ops = [
        {
            "letter": "A",
            "signal": "a(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", mul_primary),
            "signalLatex": "a(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", mul_primary),
            "summaryLatex": _build_summary_latex("A", "a(t)", "Multiplication", mul_primary),
        },
        {
            "letter": "B",
            "signal": "b(t)",
            "name": transform,
            "parameter": None,
            "signalLatex": "b(t)",
            "nameLatex": _operation_name_latex(transform),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("B", "b(t)", transform, None),
        },
        {
            "letter": "C",
            "signal": "c(t)",
            "name": "Multiplication",
            "parameter": _operation_parameter_label("Multiplication", mul_secondary),
            "signalLatex": "c(t)",
            "nameLatex": _operation_name_latex("Multiplication"),
            "parameterLatex": _operation_parameter_label_latex("Multiplication", mul_secondary),
            "summaryLatex": _build_summary_latex("C", "c(t)", "Multiplication", mul_secondary),
        },
        {
            "letter": "D",
            "signal": "d(t)",
            "name": "Real",
            "parameter": None,
            "signalLatex": "d(t)",
            "nameLatex": _operation_name_latex("Real"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("D", "d(t)", "Real", None),
        },
        {
            "letter": "E",
            "signal": "e(t)",
            "name": "Imag",
            "parameter": None,
            "signalLatex": "e(t)",
            "nameLatex": _operation_name_latex("Imag"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("E", "e(t)", "Imag", None),
        },
        {
            "letter": "F",
            "signal": "f(t)",
            "name": "Addition",
            "parameter": None,
            "signalLatex": "f(t)",
            "nameLatex": _operation_name_latex("Addition"),
            "parameterLatex": None,
            "summaryLatex": _build_summary_latex("F", "f(t)", "Addition", None),
        },
        {
            "letter": "G",
            "signal": "g(t)",
            "name": "Sampling",
            "parameter": _operation_parameter_label("Sampling", sample_param),
            "signalLatex": "g(t)",
            "nameLatex": _operation_name_latex("Sampling"),
            "parameterLatex": _operation_parameter_label_latex("Sampling", sample_param),
            "summaryLatex": _build_summary_latex("G", "g(t)", "Sampling", sample_param),
        },
        {
            "letter": "H",
            "signal": "y(t)",
            "name": "Filter",
            "parameter": _operation_parameter_label("Filter", filter_param),
            "signalLatex": "y(t)",
            "nameLatex": _operation_name_latex("Filter"),
            "parameterLatex": _operation_parameter_label_latex("Filter", filter_param),
            "summaryLatex": _build_summary_latex("H", "y(t)", "Filter", filter_param),
        },
    ]

    diagram_img = _draw_hard_diagram_complex_split_sampling(
        mul_primary,
        transform,
        mul_secondary,
        sample_param,
        filter_param,
    )

    operations = [
        {"type": "Multiplication", "param": mul_primary},
        {"type": transform, "param": None},
        {"type": "Multiplication", "param": mul_secondary},
        {"type": "Real", "param": None},
        {"type": "Imag", "param": None},
        {"type": "Addition", "param": None},
        {"type": "Sampling", "param": sample_param},
        {"type": "Filter", "param": filter_param},
    ]

    return {
        "diagram": diagram_img,
        "diagramOperations": diagram_ops,
        "operations": operations,
        "letters": letter_results,
    }




def _draw_hard_diagram_real_imag(
    mul_param: str | None,
    sample_param: str | None,
    top_mul_param: str | None,
    bottom_mul_param: str | None,
    filter_param: str | None,
) -> str:
    """Draw the HARD layout that splits real and imaginary paths."""

    fig, ax = plt.subplots(figsize=(14.4, 5.3))
    ax.axis("off")

    arrow_props = _arrow_props()

    y_mid = 1.9
    top_y = 3.4
    bot_y = 0.6
    input_x = 0.9
    mul_x = 2.5
    split_x = 3.8
    top_sample_x = 5.6
    top_real_x = 8.0
    top_mul_x = 10.4
    bottom_hilbert_x = 5.6
    bottom_imag_x = 8.0
    bottom_mul_x = 10.4
    adder_x = 12.6
    filter_x = 14.8
    output_x = 16.4

    def prepare_block(info: Dict[str, object], centre: float, y_pos: float) -> Dict[str, object]:
        info = dict(info)
        info["centre"] = centre
        info["y"] = y_pos
        if info["shape"] == "circle":
            radius = info["radius"]
            info["left"] = centre - radius
            info["right"] = centre + radius
            info["top"] = y_pos + radius
            info["bottom"] = y_pos - radius
        else:
            width = info["width"]
            height = info["height"]
            info["left"] = centre - width / 2.0
            info["right"] = centre + width / 2.0
            info["top"] = y_pos + height / 2.0
            info["bottom"] = y_pos - height / 2.0
        return info

    mul_block = prepare_block(_block_render_info("Multiplication", mul_param), mul_x, y_mid)
    sample_block = prepare_block(_block_render_info("Sampling", sample_param), top_sample_x, top_y)
    real_block = prepare_block(_block_render_info("Real", None), top_real_x, top_y)
    top_mul_block = prepare_block(_block_render_info("Multiplication", top_mul_param), top_mul_x, top_y)
    hilbert_block = prepare_block(_block_render_info("Hilbert", None), bottom_hilbert_x, bot_y)
    imag_block = prepare_block(_block_render_info("Imag", None), bottom_imag_x, bot_y)
    bottom_mul_block = prepare_block(_block_render_info("Multiplication", bottom_mul_param), bottom_mul_x, bot_y)
    adder_block = prepare_block(_block_render_info("Addition", None), adder_x, y_mid)
    filter_block = prepare_block(_block_render_info("Filter", filter_param), filter_x, y_mid)

    blocks = [
        mul_block,
        sample_block,
        real_block,
        top_mul_block,
        hilbert_block,
        imag_block,
        bottom_mul_block,
        adder_block,
        filter_block,
    ]

    def draw_block(block: Dict[str, object]) -> None:
        if block["shape"] == "circle":
            circle = plt.Circle((block["centre"], block["y"]), block["radius"], fill=False, lw=_BLOCK_LINEWIDTH, color=_EDGE_COLOR)
            ax.add_patch(circle)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                block["y"],
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 16),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                param_text_y = block["y"] + block["radius"] + 0.55
                ax.text(
                    block["centre"],
                    param_text_y,
                    param_label,
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                )
                if block.get("param_connector") == "top":
                    arrow_tail_y = param_text_y - 0.18
                    arrow_head_y = block["y"] + block["radius"] * 0.98
                    ax.annotate(
                        "",
                        xy=(block["centre"], arrow_head_y),
                        xytext=(block["centre"], arrow_tail_y),
                        arrowprops=arrow_props,
                    )
        else:
            rect = plt.Rectangle(
                (block["left"], block["bottom"]),
                block["width"],
                block["height"],
                fill=False,
                lw=_BLOCK_LINEWIDTH,
                color=_EDGE_COLOR,
                joinstyle="round",
            )
            ax.add_patch(rect)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                block["y"] + block.get("label_y_offset", 0.18),
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 11),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                ax.text(
                    block["centre"],
                    block["y"] - block.get("param_y_offset", 0.18),
                    param_label,
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    for block in blocks:
        draw_block(block)

    # Input to multiplier
    ax.annotate("", xy=(mul_block["left"], y_mid), xytext=(input_x, y_mid), arrowprops=arrow_props)
    ax.text((input_x + mul_block["left"]) / 2.0, y_mid + 0.45, "$x(t)$", ha="center", va="center", fontsize=12)

    # Multiplier to split (letter A)
    ax.annotate("", xy=(split_x, y_mid), xytext=(mul_block["right"], y_mid), arrowprops=arrow_props)
    mid_a_x = (mul_block["right"] + split_x) / 2.0
    ax.text(mid_a_x, y_mid + 0.4, "A", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_a_x, y_mid - 0.5, "$a(t)$", ha="center", va="center", fontsize=11)

    split_circle = plt.Circle((split_x, y_mid), _SPLITTER_RADIUS, color=_EDGE_COLOR)
    ax.add_patch(split_circle)

    # Top branch connections
    ax.plot([split_x, split_x], [y_mid, top_y], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.annotate("", xy=(sample_block["left"], top_y), xytext=(split_x, top_y), arrowprops=arrow_props)
    ax.annotate("", xy=(real_block["left"], top_y), xytext=(sample_block["right"], top_y), arrowprops=arrow_props)
    mid_b_x = (sample_block["right"] + real_block["left"]) / 2.0
    ax.text(mid_b_x, top_y + 0.4, "B", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_b_x, top_y - 0.6, "$b(t)$", ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(top_mul_block["left"], top_y), xytext=(real_block["right"], top_y), arrowprops=arrow_props)
    mid_c_x = (real_block["right"] + top_mul_block["left"]) / 2.0
    ax.text(mid_c_x, top_y + 0.4, "C", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_c_x, top_y - 0.6, "$c(t)$", ha="center", va="center", fontsize=11)

    join_top_y = y_mid + 1.05
    top_branch_x = top_mul_block["right"] + 0.6
    connector_points = [
        (top_mul_block["right"], top_y),
        (top_branch_x, top_y),
        (top_branch_x, join_top_y),
        (adder_block["centre"], join_top_y),
        (adder_block["centre"], y_mid + adder_block["radius"] * 0.95),
    ]
    _draw_connector(ax, connector_points, arrow_props=arrow_props)
    mid_d_x = (top_branch_x + adder_block["centre"]) / 2.0
    ax.text(mid_d_x, join_top_y + 0.35, "D", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_d_x, join_top_y - 0.35, "$d(t)$", ha="center", va="center", fontsize=11)

    # Bottom branch connections
    ax.plot([split_x, split_x], [y_mid, bot_y], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.annotate("", xy=(hilbert_block["left"], bot_y), xytext=(split_x, bot_y), arrowprops=arrow_props)
    ax.annotate("", xy=(imag_block["left"], bot_y), xytext=(hilbert_block["right"], bot_y), arrowprops=arrow_props)
    mid_e_x = (hilbert_block["right"] + imag_block["left"]) / 2.0
    ax.text(mid_e_x, bot_y + 0.55, "E", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_e_x, bot_y - 0.35, "$e(t)$", ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(bottom_mul_block["left"], bot_y), xytext=(imag_block["right"], bot_y), arrowprops=arrow_props)
    mid_f_x = (imag_block["right"] + bottom_mul_block["left"]) / 2.0
    ax.text(mid_f_x, bot_y + 0.6, "F", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_f_x, bot_y - 0.35, "$f(t)$", ha="center", va="center", fontsize=11)
    join_bot_y = y_mid - 1.05
    bottom_branch_x = bottom_mul_block["right"] + 0.6
    connector_points = [
        (bottom_mul_block["right"], bot_y),
        (bottom_branch_x, bot_y),
        (bottom_branch_x, join_bot_y),
        (adder_block["centre"], join_bot_y),
        (adder_block["centre"], y_mid - adder_block["radius"] * 0.95),
    ]
    _draw_connector(ax, connector_points, arrow_props=arrow_props)
    mid_g_x = (bottom_branch_x + adder_block["centre"]) / 2.0
    ax.text(mid_g_x, join_bot_y + 0.4, "G", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_g_x, join_bot_y - 0.35, "$g(t)$", ha="center", va="center", fontsize=11)

    # Addition to filter (letter H)
    ax.annotate("", xy=(filter_block["left"], y_mid), xytext=(adder_block["right"], y_mid), arrowprops=arrow_props)
    mid_h_x = (adder_block["right"] + filter_block["left"]) / 2.0
    ax.text(mid_h_x, y_mid + 0.5, "H", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_h_x, y_mid - 0.6, "$h(t)$", ha="center", va="center", fontsize=11)

    # Filter to output (letter I)
    ax.annotate("", xy=(output_x, y_mid), xytext=(filter_block["right"], y_mid), arrowprops=arrow_props)
    mid_i_x = (filter_block["right"] + output_x) / 2.0
    ax.text(mid_i_x, y_mid + 0.5, "I", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_i_x, y_mid - 0.6, "$y(t)$", ha="center", va="center", fontsize=11)

    ax.text(output_x + 0.75, y_mid, "$y(t)$", ha="center", va="center", fontsize=12)

    ax.set_xlim(0.3, output_x + 1.0)
    ax.set_ylim(bot_y - 0.7, top_y + 0.8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=230, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _draw_hard_diagram_complex_split_sampling(
    mul_param: str | None,
    transform_name: str,
    mul_secondary: str | None,
    sample_param: str | None,
    filter_param: str | None,
) -> str:
    """Draw the HARD layout with a transform and real/imag split before sampling."""

    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.axis("off")

    arrow_props = _arrow_props()

    y_mid = 1.75
    top_y = 3.0
    bot_y = 0.6
    input_x = 0.8
    mul_x = 2.2
    transform_x = 3.7
    second_mul_x = 5.2
    split_x = 6.3
    real_x = 7.8
    imag_x = 7.8
    adder_x = 9.4
    sampling_x = 10.95
    filter_x = 13.2
    output_x = 15.0

    def prepare_block(info: Dict[str, object], centre: float, y_pos: float) -> Dict[str, object]:
        info = dict(info)
        info["centre"] = centre
        info["y"] = y_pos
        if info["shape"] == "circle":
            radius = info["radius"]
            info["left"] = centre - radius
            info["right"] = centre + radius
            info["top"] = y_pos + radius
            info["bottom"] = y_pos - radius
        else:
            width = info["width"]
            height = info["height"]
            info["left"] = centre - width / 2.0
            info["right"] = centre + width / 2.0
            info["top"] = y_pos + height / 2.0
            info["bottom"] = y_pos - height / 2.0
        return info

    mul_block = prepare_block(_block_render_info("Multiplication", mul_param), mul_x, y_mid)
    transform_block = prepare_block(_block_render_info(transform_name, None), transform_x, y_mid)
    second_mul_block = prepare_block(_block_render_info("Multiplication", mul_secondary), second_mul_x, y_mid)
    real_block = prepare_block(_block_render_info("Real", None), real_x, top_y)
    imag_block = prepare_block(_block_render_info("Imag", None), imag_x, bot_y)
    adder_block = prepare_block(_block_render_info("Addition", None), adder_x, y_mid)
    sampling_block = prepare_block(_block_render_info("Sampling", sample_param), sampling_x, y_mid)
    filter_block = prepare_block(_block_render_info("Filter", filter_param), filter_x, y_mid)

    blocks = [
        mul_block,
        transform_block,
        second_mul_block,
        real_block,
        imag_block,
        adder_block,
        sampling_block,
        filter_block,
    ]

    def draw_block(block: Dict[str, object]) -> None:
        if block["shape"] == "circle":
            circle = plt.Circle((block["centre"], block["y"]), block["radius"], fill=False, lw=_BLOCK_LINEWIDTH, color=_EDGE_COLOR)
            ax.add_patch(circle)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                block["y"],
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 16),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                param_text_y = block["y"] + block["radius"] + 0.55
                ax.text(
                    block["centre"],
                    param_text_y,
                    param_label,
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                )
                if block.get("param_connector") == "top":
                    arrow_tail_y = param_text_y - 0.18
                    arrow_head_y = block["y"] + block["radius"] * 0.98
                    ax.annotate(
                        "",
                        xy=(block["centre"], arrow_head_y),
                        xytext=(block["centre"], arrow_tail_y),
                        arrowprops=arrow_props,
                    )
        else:
            rect = plt.Rectangle(
                (block["left"], block["bottom"]),
                block["width"],
                block["height"],
                fill=False,
                lw=_BLOCK_LINEWIDTH,
                color=_EDGE_COLOR,
                joinstyle="round",
            )
            ax.add_patch(rect)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                block["y"] + block.get("label_y_offset", 0.18),
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 11),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                ax.text(
                    block["centre"],
                    block["y"] - block.get("param_y_offset", 0.18),
                    param_label,
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    for block in blocks:
        draw_block(block)

    # Input to first multiplier
    ax.annotate("", xy=(mul_block["left"], y_mid), xytext=(input_x, y_mid), arrowprops=arrow_props)
    ax.text((input_x + mul_block["left"]) / 2.0, y_mid + 0.45, "$x(t)$", ha="center", va="center", fontsize=12)

    # Multiplier to transform (letter A)
    ax.annotate("", xy=(transform_block["left"], y_mid), xytext=(mul_block["right"], y_mid), arrowprops=arrow_props)
    mid_a_x = (mul_block["right"] + transform_block["left"]) / 2.0
    ax.text(mid_a_x, y_mid + 0.45, "A", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_a_x, y_mid - 0.55, "$a(t)$", ha="center", va="center", fontsize=11)

    # Transform to second multiplier (letter B)
    ax.annotate("", xy=(second_mul_block["left"], y_mid), xytext=(transform_block["right"], y_mid), arrowprops=arrow_props)
    mid_b_x = (transform_block["right"] + second_mul_block["left"]) / 2.0
    ax.text(mid_b_x, y_mid + 0.45, "B", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_b_x, y_mid - 0.55, "$b(t)$", ha="center", va="center", fontsize=11)

    # Second multiplier to splitter (letter C)
    ax.annotate("", xy=(split_x, y_mid), xytext=(second_mul_block["right"], y_mid), arrowprops=arrow_props)
    mid_c_x = (second_mul_block["right"] + split_x) / 2.0
    ax.text(mid_c_x, y_mid + 0.45, "C", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_c_x, y_mid - 0.55, "$c(t)$", ha="center", va="center", fontsize=11)

    split_circle = plt.Circle((split_x, y_mid), _SPLITTER_RADIUS, color=_EDGE_COLOR)
    ax.add_patch(split_circle)

    # Top branch: splitter to real block
    ax.plot([split_x, split_x], [y_mid, top_y], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.annotate("", xy=(real_block["left"], top_y), xytext=(split_x, top_y), arrowprops=arrow_props)

    # Bottom branch: splitter to imag block
    ax.plot([split_x, split_x], [y_mid, bot_y], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.annotate("", xy=(imag_block["left"], bot_y), xytext=(split_x, bot_y), arrowprops=arrow_props)

    # Real block to adder (letter D)
    adder_radius = adder_block.get("radius", 0.36)
    real_route_x = real_block["right"] + 0.6
    connector_points = [
        (real_block["right"], top_y),
        (real_route_x, top_y),
        (real_route_x, y_mid + adder_radius + 0.12),
        (adder_block["centre"], y_mid + adder_radius + 0.12),
        (adder_block["centre"], y_mid + adder_radius * 0.95),
    ]
    _draw_connector(ax, connector_points, arrow_props=arrow_props)
    mid_d_x = (real_route_x + adder_block["centre"]) / 2.0
    ax.text(mid_d_x, y_mid + adder_radius + 0.45, "D", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_d_x, y_mid + adder_radius - 0.2, "$d(t)$", ha="center", va="center", fontsize=11)

    # Imag block to adder (letter E)
    imag_route_x = imag_block["right"] + 0.6
    connector_points = [
        (imag_block["right"], bot_y),
        (imag_route_x, bot_y),
        (imag_route_x, y_mid - adder_radius - 0.12),
        (adder_block["centre"], y_mid - adder_radius - 0.12),
        (adder_block["centre"], y_mid - adder_radius * 0.95),
    ]
    _draw_connector(ax, connector_points, arrow_props=arrow_props)
    mid_e_x = (imag_route_x + adder_block["centre"]) / 2.0
    ax.text(mid_e_x, y_mid - adder_radius - 0.45, "E", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_e_x, y_mid - adder_radius - 1.0, "$e(t)$", ha="center", va="center", fontsize=11)

    # Adder to sampling (letter F)
    ax.annotate("", xy=(sampling_block["left"], y_mid), xytext=(adder_block["right"], y_mid), arrowprops=arrow_props)
    mid_f_x = (adder_block["right"] + sampling_block["left"]) / 2.0
    ax.text(mid_f_x, y_mid + 0.45, "F", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_f_x, y_mid - 0.55, "$f(t)$", ha="center", va="center", fontsize=11)

    # Sampling to filter (letter G)
    ax.annotate("", xy=(filter_block["left"], y_mid), xytext=(sampling_block["right"], y_mid), arrowprops=arrow_props)
    mid_g_x = (sampling_block["right"] + filter_block["left"]) / 2.0
    ax.text(mid_g_x, y_mid + 0.45, "G", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_g_x, y_mid - 0.55, "$g(t)$", ha="center", va="center", fontsize=11)

    # Filter to output (letter H)
    ax.annotate("", xy=(output_x, y_mid), xytext=(filter_block["right"], y_mid), arrowprops=arrow_props)
    mid_h_x = (filter_block["right"] + output_x) / 2.0
    ax.text(mid_h_x, y_mid + 0.45, "H", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_h_x, y_mid - 0.55, "$y(t)$", ha="center", va="center", fontsize=11)

    ax.text(output_x + 0.75, y_mid, "$y(t)$", ha="center", va="center", fontsize=12)

    ax.set_xlim(0.3, output_x + 0.9)
    ax.set_ylim(bot_y - 0.7, top_y + 0.8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=230, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")



def _draw_hard_diagram_real_imag_sampling(
    mul_param: str | None,
    top_mul_param: str | None,
    bottom_mul_param: str | None,
    sample_param: str | None,
    filter_param: str | None,
) -> str:
    """Draw the additional HARD layout with real/imag branches followed by sampling."""

    fig, ax = plt.subplots(figsize=(13.0, 4.9))
    ax.axis("off")

    arrow_props = _arrow_props()

    y_mid = 1.9
    top_y = 3.2
    bot_y = 0.6
    input_x = 0.9
    mul_x = 2.4
    split_x = 3.6
    top_real_x = 5.3
    top_mul_x = 7.3
    bottom_imag_x = 5.3
    bottom_mul_x = 7.3
    adder_x = 9.6
    sampling_x = 11.6
    filter_x = 13.6
    output_x = 15.0

    def prepare_block(info: Dict[str, object], centre: float, y_pos: float) -> Dict[str, object]:
        info = dict(info)
        info["centre"] = centre
        info["y"] = y_pos
        if info["shape"] == "circle":
            radius = info["radius"]
            info["left"] = centre - radius
            info["right"] = centre + radius
            info["top"] = y_pos + radius
            info["bottom"] = y_pos - radius
        else:
            width = info["width"]
            height = info["height"]
            info["left"] = centre - width / 2.0
            info["right"] = centre + width / 2.0
            info["top"] = y_pos + height / 2.0
            info["bottom"] = y_pos - height / 2.0
        return info

    mul_block = prepare_block(_block_render_info("Multiplication", mul_param), mul_x, y_mid)
    real_block = prepare_block(_block_render_info("Real", None), top_real_x, top_y)
    top_mul_block = prepare_block(_block_render_info("Multiplication", top_mul_param), top_mul_x, top_y)
    imag_block = prepare_block(_block_render_info("Imag", None), bottom_imag_x, bot_y)
    bottom_mul_block = prepare_block(_block_render_info("Multiplication", bottom_mul_param), bottom_mul_x, bot_y)
    adder_block = prepare_block(_block_render_info("Addition", None), adder_x, y_mid)
    sampling_block = prepare_block(_block_render_info("Sampling", sample_param), sampling_x, y_mid)
    filter_block = prepare_block(_block_render_info("Filter", filter_param), filter_x, y_mid)

    blocks = [
        mul_block,
        real_block,
        top_mul_block,
        imag_block,
        bottom_mul_block,
        adder_block,
        sampling_block,
        filter_block,
    ]

    def draw_block(block: Dict[str, object]) -> None:
        if block["shape"] == "circle":
            circle = plt.Circle((block["centre"], block["y"]), block["radius"] * 1.0, fill=False, lw=_BLOCK_LINEWIDTH, color=_EDGE_COLOR)
            ax.add_patch(circle)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                block["y"],
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 16),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                param_text_y = block["y"] + block["radius"] + 0.5
                ax.text(
                    block["centre"],
                    param_text_y,
                    param_label,
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                )
                if block.get("param_connector") == "top":
                    arrow_tail_y = param_text_y - 0.18
                    arrow_head_y = block["y"] + block["radius"] * 0.98
                    ax.annotate(
                        "",
                        xy=(block["centre"], arrow_head_y),
                        xytext=(block["centre"], arrow_tail_y),
                        arrowprops=arrow_props,
                    )
        else:
            rect = plt.Rectangle(
                (block["left"], block["bottom"]),
                block["width"],
                block["height"],
                fill=False,
                lw=_BLOCK_LINEWIDTH,
                color=_EDGE_COLOR,
                joinstyle="round",
            )
            ax.add_patch(rect)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                block["y"] + block.get("label_y_offset", 0.18),
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 11),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                ax.text(
                    block["centre"],
                    block["y"] - block.get("param_y_offset", 0.18),
                    param_label,
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    for block in blocks:
        draw_block(block)

    ax.annotate("", xy=(mul_block["left"], y_mid), xytext=(input_x, y_mid), arrowprops=arrow_props)
    ax.text((input_x + mul_block["left"]) / 2.0, y_mid + 0.5, "$x(t)$", ha="center", va="center", fontsize=12)

    ax.annotate("", xy=(split_x, y_mid), xytext=(mul_block["right"], y_mid), arrowprops=arrow_props)
    mid_a_x = (mul_block["right"] + split_x) / 2.0
    ax.text(mid_a_x, y_mid + 0.45, "A", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_a_x, y_mid - 0.6, "$a(t)$", ha="center", va="center", fontsize=11)

    split_circle = plt.Circle((split_x, y_mid), _SPLITTER_RADIUS, color=_EDGE_COLOR)
    ax.add_patch(split_circle)

    ax.plot([split_x, split_x], [y_mid, top_y - 0.25], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.plot([split_x, split_x], [y_mid, bot_y + 0.25], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)

    ax.annotate("", xy=(real_block["left"], top_y), xytext=(split_x, top_y), arrowprops=arrow_props)
    ax.annotate("", xy=(imag_block["left"], bot_y), xytext=(split_x, bot_y), arrowprops=arrow_props)

    ax.annotate("", xy=(top_mul_block["left"], top_y), xytext=(real_block["right"], top_y), arrowprops=arrow_props)
    mid_b_x = (real_block["right"] + top_mul_block["left"]) / 2.0
    ax.text(mid_b_x, top_y + 0.4, "B", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_b_x, top_y - 0.6, "$b(t)$", ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(bottom_mul_block["left"], bot_y), xytext=(imag_block["right"], bot_y), arrowprops=arrow_props)
    mid_d_x = (imag_block["right"] + bottom_mul_block["left"]) / 2.0
    ax.text(mid_d_x, bot_y + 0.6, "D", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_d_x, bot_y - 0.35, "$d(t)$", ha="center", va="center", fontsize=11)

    join_top_y = y_mid + 0.95
    join_bot_y = y_mid - 0.95

    top_route_x = top_mul_block["right"] + 0.55
    connector_points = [
        (top_mul_block["right"], top_y),
        (top_route_x, top_y),
        (top_route_x, join_top_y),
        (adder_block["centre"], join_top_y),
        (adder_block["centre"], y_mid + adder_block["radius"] * 0.95),
    ]
    _draw_connector(ax, connector_points, arrow_props=arrow_props)
    mid_c_x = (top_route_x + adder_block["centre"]) / 2.0
    ax.text(mid_c_x, join_top_y + 0.32, "C", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_c_x, join_top_y - 0.38, "$c(t)$", ha="center", va="center", fontsize=11)

    bottom_route_x = bottom_mul_block["right"] + 0.55
    connector_points = [
        (bottom_mul_block["right"], bot_y),
        (bottom_route_x, bot_y),
        (bottom_route_x, join_bot_y),
        (adder_block["centre"], join_bot_y),
        (adder_block["centre"], y_mid - adder_block["radius"] * 0.95),
    ]
    _draw_connector(ax, connector_points, arrow_props=arrow_props)
    mid_e_x = (bottom_route_x + adder_block["centre"]) / 2.0
    ax.text(mid_e_x, join_bot_y + 0.33, "E", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_e_x, join_bot_y - 0.38, "$e(t)$", ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(sampling_block["left"], y_mid), xytext=(adder_block["right"], y_mid), arrowprops=arrow_props)
    mid_f_x = (adder_block["right"] + sampling_block["left"]) / 2.0
    ax.text(mid_f_x, y_mid + 0.5, "F", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_f_x, y_mid - 0.6, "$f(t)$", ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(filter_block["left"], y_mid), xytext=(sampling_block["right"], y_mid), arrowprops=arrow_props)
    mid_g_x = (sampling_block["right"] + filter_block["left"]) / 2.0
    ax.text(mid_g_x, y_mid + 0.5, "G", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_g_x, y_mid - 0.6, "$g(t)$", ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(output_x, y_mid), xytext=(filter_block["right"], y_mid), arrowprops=arrow_props)
    mid_h_x = (filter_block["right"] + output_x) / 2.0
    ax.text(mid_h_x, y_mid + 0.5, "H", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_h_x, y_mid - 0.6, "$y(t)$", ha="center", va="center", fontsize=11)

    ax.text(output_x + 0.75, y_mid, "$y(t)$", ha="center", va="center", fontsize=12)

    ax.set_xlim(0.3, output_x + 1.0)
    ax.set_ylim(bot_y - 0.7, top_y + 0.7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _draw_diagram(sequence: Tuple[str, str, str], params: Dict[str, str | None]) -> str:
    """
    Draw a block diagram that mirrors the processing chain presentation.

    Multiplication is shown as a circle with a multiplication symbol and the
    intermediate signals are labelled x(t) → a(t) → b(t) → c(t) along the
    arrows to match the training flow.
    """
    fig, ax = plt.subplots(figsize=(9.8, 3.0))
    ax.axis("off")

    # Position the three blocks evenly on the canvas
    n_blocks = len(sequence)
    start = 1.8
    spacing = 2.7
    block_centres = [start + i * spacing for i in range(n_blocks)]
    y_pos = 1.3

    signal_labels = ["x(t)"] + [f"{chr(ord('a') + i)}(t)" for i in range(n_blocks)]
    letter_labels = [chr(ord("A") + i) for i in range(n_blocks)]

    arrow_props = _arrow_props()

    block_geometries = []
    for centre, op_name in zip(block_centres, sequence):
        op_params = params.get(op_name)
        block_info = _block_render_info(op_name, op_params)
        block_info["centre"] = centre
        if block_info["shape"] == "circle":
            radius = block_info["radius"]
            block_info["left"] = centre - radius
            block_info["right"] = centre + radius
        else:
            half_width = block_info["width"] / 2.0
            block_info["left"] = centre - half_width
            block_info["right"] = centre + half_width
        block_geometries.append(block_info)

    input_x = block_geometries[0]["left"] - 0.9
    ax.annotate("", xy=(block_geometries[0]["left"], y_pos), xytext=(input_x, y_pos), arrowprops=arrow_props)
    ax.text((input_x + block_geometries[0]["left"]) / 2.0, y_pos + 0.5, f"${signal_labels[0]}$", ha="center", va="center", fontsize=12)

    for block in block_geometries:
        if block["shape"] == "circle":
            circle = plt.Circle((block["centre"], y_pos), block["radius"], fill=False, lw=_BLOCK_LINEWIDTH, color=_EDGE_COLOR)
            ax.add_patch(circle)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                y_pos,
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 16),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                param_text_y = y_pos + block["radius"] + 0.5
                ax.text(
                    block["centre"],
                    param_text_y,
                    param_label,
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                )
                if block.get("param_connector") == "top":
                    arrow_tail_y = param_text_y - 0.18
                    arrow_head_y = y_pos + block["radius"] * 0.98
                    ax.annotate(
                        "",
                        xy=(block["centre"], arrow_head_y),
                        xytext=(block["centre"], arrow_tail_y),
                        arrowprops=arrow_props,
                    )
        else:
            rect = plt.Rectangle(
                (block["left"], y_pos - block["height"] / 2.0),
                block["width"],
                block["height"],
                fill=False,
                lw=_BLOCK_LINEWIDTH,
                color=_EDGE_COLOR,
                joinstyle="round",
            )
            ax.add_patch(rect)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                y_pos + block.get("label_y_offset", 0.18),
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 11),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                ax.text(
                    block["centre"],
                    y_pos - block.get("param_y_offset", 0.18),
                    param_label,
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    for idx in range(len(block_geometries) - 1):
        left_block = block_geometries[idx]
        right_block = block_geometries[idx + 1]
        ax.annotate(
            "",
            xy=(right_block["left"], y_pos),
            xytext=(left_block["right"], y_pos),
            arrowprops=arrow_props,
        )
        mid_x = (left_block["right"] + right_block["left"]) / 2.0
        ax.text(mid_x, y_pos + 0.5, letter_labels[idx], ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(mid_x, y_pos - 0.6, f"${signal_labels[idx + 1]}$", ha="center", va="center", fontsize=11)

    last_block = block_geometries[-1]
    output_x = last_block["right"] + 0.9
    ax.annotate("", xy=(output_x, y_pos), xytext=(last_block["right"], y_pos), arrowprops=arrow_props)
    mid_out = (last_block["right"] + output_x) / 2.0
    ax.text(mid_out, y_pos + 0.5, letter_labels[-1], ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_out, y_pos - 0.6, f"${signal_labels[-1]}$", ha="center", va="center", fontsize=11)
    ax.text(output_x + 0.6, y_pos, "$y(t)$", ha="center", va="center", fontsize=12)

    max_x = start + (n_blocks - 1) * spacing + 2.4
    ax.set_xlim(0, max_x)
    ax.set_ylim(0.1, 2.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=210, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _draw_medium_diagram_sampling(
    branch_ops: List[Tuple[str, str | None]],
    post_op: str,
    post_param: str | None,
    filter_param: str,
) -> str:
    """Draw the original MEDIUM layout with sampling/multiplication branches."""

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")

    arrow_props = _arrow_props()

    y_mid = 1.9
    top_y = 3.1
    bot_y = 0.7
    input_x = 0.8
    split_x = 1.7
    branch_x = 3.5
    adder_x = 5.9
    post_x = 8.0
    filter_x = 10.1
    output_x = 11.5

    def prepare_block(info: Dict[str, object], centre: float, y_pos: float) -> Dict[str, object]:
        info = dict(info)
        info["centre"] = centre
        info["y"] = y_pos
        if info["shape"] == "circle":
            radius = info["radius"]
            info["left"] = centre - radius
            info["right"] = centre + radius
            info["top"] = y_pos + radius
            info["bottom"] = y_pos - radius
        else:
            width = info["width"]
            height = info["height"]
            info["left"] = centre - width / 2.0
            info["right"] = centre + width / 2.0
            info["top"] = y_pos + height / 2.0
            info["bottom"] = y_pos - height / 2.0
        return info

    top_block = prepare_block(_block_render_info(*branch_ops[0]), branch_x, top_y)
    bot_block = prepare_block(_block_render_info(*branch_ops[1]), branch_x, bot_y)
    adder_block = prepare_block(_block_render_info("Addition", None), adder_x, y_mid)
    post_block = prepare_block(_block_render_info(post_op, post_param), post_x, y_mid)
    filter_block = prepare_block(_block_render_info("Filter", filter_param), filter_x, y_mid)

    blocks = [top_block, bot_block, adder_block, post_block, filter_block]

    def draw_block(block: Dict[str, object]) -> None:
        if block["shape"] == "circle":
            circle = plt.Circle((block["centre"], block["y"]), block["radius"], fill=False, lw=_BLOCK_LINEWIDTH, color=_EDGE_COLOR)
            ax.add_patch(circle)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                block["y"],
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 16),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                param_text_y = block["y"] + block["radius"] + 0.5
                ax.text(
                    block["centre"],
                    param_text_y,
                    param_label,
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                )
                if block.get("param_connector") == "top":
                    arrow_tail_y = param_text_y - 0.18
                    arrow_head_y = block["y"] + block["radius"] * 0.98
                    ax.annotate(
                        "",
                        xy=(block["centre"], arrow_head_y),
                        xytext=(block["centre"], arrow_tail_y),
                        arrowprops=arrow_props,
                    )
        else:
            rect = plt.Rectangle(
                (block["left"], block["bottom"]),
                block["width"],
                block["height"],
                fill=False,
                lw=_BLOCK_LINEWIDTH,
                color=_EDGE_COLOR,
                joinstyle="round",
            )
            ax.add_patch(rect)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(
                block["centre"],
                block["y"] + block.get("label_y_offset", 0.18),
                label,
                ha="center",
                va="center",
                fontsize=block.get("label_fontsize", 11),
                fontweight="bold",
            )
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                ax.text(
                    block["centre"],
                    block["y"] - block.get("param_y_offset", 0.18),
                    param_label,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
    for block in blocks:
        draw_block(block)

    # Input and splitter
    ax.annotate("", xy=(split_x, y_mid), xytext=(input_x, y_mid), arrowprops=arrow_props)
    ax.text((input_x + split_x) / 2.0, y_mid + 0.5, "$x(t)$", ha="center", va="center", fontsize=12)
    split_circle = plt.Circle((split_x, y_mid), _SPLITTER_RADIUS, color=_EDGE_COLOR)
    ax.add_patch(split_circle)

    # Branch connections
    ax.plot([split_x, split_x], [y_mid, top_block["y"] - 0.25], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.plot([split_x, split_x], [y_mid, bot_block["y"] + 0.25], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.annotate("", xy=(top_block["left"], top_block["y"]), xytext=(split_x, top_block["y"]), arrowprops=arrow_props)
    ax.annotate("", xy=(bot_block["left"], bot_block["y"]), xytext=(split_x, bot_block["y"]), arrowprops=arrow_props)

    # Branch outputs into adder
    adder_radius = adder_block.get("radius", 0.45)
    top_target = (adder_block["centre"], y_mid + adder_radius * 0.95)
    bot_target = (adder_block["centre"], y_mid - adder_radius * 0.95)

    top_knee_x = (top_block["right"] + adder_block["left"]) / 2.0
    bot_knee_x = (bot_block["right"] + adder_block["left"]) / 2.0

    join_top_y = y_mid + 1.0
    join_bot_y = y_mid - 1.0

    top_connector = [
        (top_block["right"], top_block["y"]),
        (top_knee_x, top_block["y"]),
        (top_knee_x, join_top_y),
        (adder_block["centre"], join_top_y),
        top_target,
    ]
    _draw_connector(ax, top_connector, arrow_props=arrow_props)

    bottom_connector = [
        (bot_block["right"], bot_block["y"]),
        (bot_knee_x, bot_block["y"]),
        (bot_knee_x, join_bot_y),
        (adder_block["centre"], join_bot_y),
        bot_target,
    ]
    _draw_connector(ax, bottom_connector, arrow_props=arrow_props)

    # Labels for branches
    mid_top_x = (top_knee_x + top_target[0]) / 2.0
    mid_top_y = (join_top_y + top_block["y"]) / 2.0
    ax.text(mid_top_x, mid_top_y + 0.28, "A", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_top_x, mid_top_y - 0.38, "$a(t)$", ha="center", va="center", fontsize=11)

    mid_bot_x = (bot_knee_x + bot_target[0]) / 2.0
    mid_bot_y = (join_bot_y + bot_block["y"]) / 2.0
    ax.text(mid_bot_x, mid_bot_y + 0.28, "B", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_bot_x, mid_bot_y - 0.38, "$b(t)$", ha="center", va="center", fontsize=11)

    # Adder to post block
    ax.annotate("", xy=(post_block["left"], y_mid), xytext=(adder_block["right"], y_mid), arrowprops=arrow_props)
    mid_c_x = (adder_block["right"] + post_block["left"]) / 2.0
    ax.text(mid_c_x, y_mid + 0.45, "C", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_c_x, y_mid - 0.6, "$c(t)$", ha="center", va="center", fontsize=11)

    # Post block to filter
    ax.annotate("", xy=(filter_block["left"], y_mid), xytext=(post_block["right"], y_mid), arrowprops=arrow_props)
    mid_d_x = (post_block["right"] + filter_block["left"]) / 2.0
    ax.text(mid_d_x, y_mid + 0.45, "D", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_d_x, y_mid - 0.6, "$d(t)$", ha="center", va="center", fontsize=11)

    # Filter to output
    ax.annotate("", xy=(output_x, y_mid), xytext=(filter_block["right"], y_mid), arrowprops=arrow_props)
    mid_e_x = (filter_block["right"] + output_x) / 2.0
    ax.text(mid_e_x, y_mid + 0.45, "E", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_e_x, y_mid - 0.6, "$e(t)$", ha="center", va="center", fontsize=11)
    ax.text(output_x + 0.7, y_mid, "$y(t)$", ha="center", va="center", fontsize=12)

    ax.set_xlim(0.2, output_x + 1.1)
    ax.set_ylim(bot_y - 0.7, top_y + 0.8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.2)

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _draw_medium_diagram_multiplication_split(
    mul_param: str | None,
    branch_ops: List[Tuple[str, str | None]],
    filter_param: str,
) -> str:
    """Draw the MEDIUM layout with an initial multiplication before branching."""

    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.axis("off")

    arrow_props = _arrow_props()

    y_mid = 1.9
    top_y = 3.1
    bot_y = 0.7
    input_x = 0.8
    mul_x = 2.4
    split_x = 3.7
    branch_x = 5.5
    adder_x = 7.7
    filter_x = 9.9
    output_x = 11.3

    def prepare_block(info: Dict[str, object], centre: float, y_pos: float) -> Dict[str, object]:
        info = dict(info)
        info["centre"] = centre
        info["y"] = y_pos
        if info["shape"] == "circle":
            radius = info["radius"]
            info["left"] = centre - radius
            info["right"] = centre + radius
            info["top"] = y_pos + radius
            info["bottom"] = y_pos - radius
        else:
            width = info["width"]
            height = info["height"]
            info["left"] = centre - width / 2.0
            info["right"] = centre + width / 2.0
            info["top"] = y_pos + height / 2.0
            info["bottom"] = y_pos - height / 2.0
        return info

    mul_block = prepare_block(_block_render_info("Multiplication", mul_param), mul_x, y_mid)
    top_block = prepare_block(_block_render_info(*branch_ops[0]), branch_x, top_y)
    bot_block = prepare_block(_block_render_info(*branch_ops[1]), branch_x, bot_y)
    adder_block = prepare_block(_block_render_info("Addition", None), adder_x, y_mid)
    filter_block = prepare_block(_block_render_info("Filter", filter_param), filter_x, y_mid)

    blocks = [mul_block, top_block, bot_block, adder_block, filter_block]

    def draw_block(block: Dict[str, object]) -> None:
        if block["shape"] == "circle":
            circle = plt.Circle((block["centre"], block["y"]), block["radius"], fill=False, lw=_BLOCK_LINEWIDTH, color=_EDGE_COLOR)
            ax.add_patch(circle)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(block["centre"], block["y"], label, ha="center", va="center", fontsize=16, fontweight="bold")
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                param_text_y = block["y"] + block["radius"] + 0.5
                ax.text(
                    block["centre"],
                    param_text_y,
                    param_label,
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                )
                if block.get("param_connector") == "top":
                    arrow_tail_y = param_text_y - 0.18
                    arrow_head_y = block["y"] + block["radius"] * 0.98
                    ax.annotate(
                        "",
                        xy=(block["centre"], arrow_head_y),
                        xytext=(block["centre"], arrow_tail_y),
                        arrowprops=arrow_props,
                    )
        else:
            rect = plt.Rectangle(
                (block["left"], block["bottom"]),
                block["width"],
                block["height"],
                fill=False,
                lw=_BLOCK_LINEWIDTH,
                color=_EDGE_COLOR,
                joinstyle="round",
            )
            ax.add_patch(rect)
            label = block.get("label_latex") or block.get("label")
            if block.get("label_latex"):
                label = f"${label}$"
            ax.text(block["centre"], block["y"] + 0.18, label, ha="center", va="center", fontsize=11, fontweight="bold")
            param_label = None
            if block.get("param_latex"):
                param_label = f"${block['param_latex']}$"
            elif block.get("param"):
                param_label = block["param"]
            if param_label:
                ax.text(block["centre"], block["y"] - 0.18, param_label, ha="center", va="center", fontsize=10)

    for block in blocks:
        draw_block(block)

    # Input to multiplication
    ax.annotate("", xy=(mul_block["left"], y_mid), xytext=(input_x, y_mid), arrowprops=arrow_props)
    ax.text((input_x + mul_block["left"]) / 2.0, y_mid + 0.45, "$x(t)$", ha="center", va="center", fontsize=12)

    # Multiplication to splitter
    ax.annotate("", xy=(split_x, y_mid), xytext=(mul_block["right"], y_mid), arrowprops=arrow_props)
    mid_a_x = (mul_block["right"] + split_x) / 2.0
    ax.text(mid_a_x, y_mid + 0.45, "A", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_a_x, y_mid - 0.6, "$a(t)$", ha="center", va="center", fontsize=11)

    split_circle = plt.Circle((split_x, y_mid), _SPLITTER_RADIUS, color=_EDGE_COLOR)
    ax.add_patch(split_circle)

    # Branch connections
    ax.plot([split_x, split_x], [y_mid, top_block["y"] - 0.25], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.plot([split_x, split_x], [y_mid, bot_block["y"] + 0.25], color=_EDGE_COLOR, lw=_CONNECTOR_LINEWIDTH)
    ax.annotate("", xy=(top_block["left"], top_block["y"]), xytext=(split_x, top_block["y"]), arrowprops=arrow_props)
    ax.annotate("", xy=(bot_block["left"], bot_block["y"]), xytext=(split_x, bot_block["y"]), arrowprops=arrow_props)

    # Branch outputs into adder
    adder_radius = adder_block.get("radius", 0.45)
    top_target = (adder_block["centre"], y_mid + adder_radius * 0.95)
    bot_target = (adder_block["centre"], y_mid - adder_radius * 0.95)

    top_knee_x = (top_block["right"] + adder_block["left"]) / 2.0
    bot_knee_x = (bot_block["right"] + adder_block["left"]) / 2.0

    join_top_y = y_mid + 1.0
    join_bot_y = y_mid - 1.0

    top_connector = [
        (top_block["right"], top_block["y"]),
        (top_knee_x, top_block["y"]),
        (top_knee_x, join_top_y),
        (adder_block["centre"], join_top_y),
        top_target,
    ]
    _draw_connector(ax, top_connector, arrow_props=arrow_props)

    bottom_connector = [
        (bot_block["right"], bot_block["y"]),
        (bot_knee_x, bot_block["y"]),
        (bot_knee_x, join_bot_y),
        (adder_block["centre"], join_bot_y),
        bot_target,
    ]
    _draw_connector(ax, bottom_connector, arrow_props=arrow_props)

    # Labels for branches
    mid_top_x = (top_knee_x + top_target[0]) / 2.0
    mid_top_y = (join_top_y + top_block["y"]) / 2.0
    ax.text(mid_top_x, mid_top_y + 0.28, "B", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_top_x, mid_top_y - 0.38, "$b(t)$", ha="center", va="center", fontsize=11)

    mid_bot_x = (bot_knee_x + bot_target[0]) / 2.0
    mid_bot_y = (join_bot_y + bot_block["y"]) / 2.0
    ax.text(mid_bot_x, mid_bot_y + 0.28, "C", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_bot_x, mid_bot_y - 0.38, "$c(t)$", ha="center", va="center", fontsize=11)

    # Adder to filter
    ax.annotate("", xy=(filter_block["left"], y_mid), xytext=(adder_block["right"], y_mid), arrowprops=arrow_props)
    mid_d_x = (adder_block["right"] + filter_block["left"]) / 2.0
    ax.text(mid_d_x, y_mid + 0.45, "D", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_d_x, y_mid - 0.6, "$d(t)$", ha="center", va="center", fontsize=11)

    # Filter to output
    ax.annotate("", xy=(output_x, y_mid), xytext=(filter_block["right"], y_mid), arrowprops=arrow_props)
    mid_e_x = (filter_block["right"] + output_x) / 2.0
    ax.text(mid_e_x, y_mid + 0.45, "E", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_e_x, y_mid - 0.6, "$e(t)$", ha="center", va="center", fontsize=11)
    ax.text(output_x + 0.7, y_mid, "$y(t)$", ha="center", va="center", fontsize=12)

    ax.set_xlim(0.2, output_x + 1.1)
    ax.set_ylim(bot_y - 0.7, top_y + 0.8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")



def _block_render_info(op_name: str, param: str | None) -> Dict[str, object]:
    """Return geometry and labels for rendering a block."""

    if op_name == "Multiplication":
        radius = 0.43
        return {
            "shape": "circle",
            "radius": radius,
            "left": None,  # filled in later
            "right": None,
            "label": "×",
            "label_latex": r"\times",
            "param": _describe_multiplication_param(param),
            "param_latex": _describe_multiplication_param_latex(param),
            "label_fontsize": 16,
            "param_connector": "top",
        }

    if op_name == "Addition":
        radius = 0.36
        return {
            "shape": "circle",
            "radius": radius,
            "left": None,
            "right": None,
            "label": "+",
            "label_latex": r"+",
            "param": None,
            "param_latex": None,
            "label_fontsize": 16,
        }

    width = 1.65
    height = 0.68
    label_fontsize = 11
    label_y_offset = 0.12
    param_y_offset = 0.26
    if op_name == "Filter":
        label = "Filter"
        param_text = _describe_filter_param(param)
        param_text_latex = _describe_filter_param_latex(param)
        width = 2.2
        param_y_offset = 0.22
    elif op_name == "Hilbert":
        label = "Hilbert"
        param_text = None
        param_text_latex = None
        label_fontsize = 17
        label_y_offset = 0.0
        param_y_offset = 0.22
    elif op_name == "Sampling":
        label = "Sampling"
        param_text = _describe_sampling_param(param)
        param_text_latex = _describe_sampling_param_latex(param)
        width = 1.8
    elif op_name == "Derivative":
        label = "Derivative"
        param_text = None
        param_text_latex = None
        label_fontsize = 17
        label_y_offset = 0.0
        param_y_offset = 0.22
    elif op_name == "Real":
        label = "Re"
        param_text = None
        param_text_latex = None
        label_fontsize = 17
        label_y_offset = 0.0
        param_y_offset = 0.22
    elif op_name == "Imag":
        label = "Im"
        param_text = None
        param_text_latex = None
        label_fontsize = 17
        label_y_offset = 0.0
        param_y_offset = 0.22
    else:
        label = op_name
        param_text = None
        param_text_latex = None


    return {
        "shape": "rect",
        "width": width,
        "height": height,
        "left": None,
        "right": None,
        "label": label,
        "label_latex": _operation_name_latex(op_name),
        "param": param_text,
        "param_latex": param_text_latex,
        "label_fontsize": label_fontsize,
        "label_y_offset": label_y_offset,
        "param_y_offset": param_y_offset,
    }

def _ensure_option_diversity(options: List[np.ndarray], w: np.ndarray) -> List[np.ndarray]:
    """Ensure the spectra shown to the user are pairwise distinguishable."""

    adjusted: List[np.ndarray] = []
    for idx, spec in enumerate(options):
        candidate = spec.copy()
        if idx == 0:
            adjusted.append(candidate)
            continue

        attempt = 0
        while any(_spectra_are_close(candidate, other) for other in adjusted):
            candidate = _nudge_spectrum(candidate, w, attempt)
            attempt += 1
            if attempt > 6:
                break
        adjusted.append(candidate)
    return adjusted


def _spectra_are_close(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True when two spectra are visually indistinguishable."""

    return np.allclose(a, b, rtol=1e-3, atol=1e-5)


def _nudge_spectrum(sig: np.ndarray, w: np.ndarray, attempt: int) -> np.ndarray:
    """Perturb ``sig`` slightly so it no longer matches another option."""

    if np.allclose(sig, 0.0, atol=1e-8):
        width = 1.2 + 0.4 * attempt
        scale = 0.2 + 0.05 * attempt
        return scale * np.exp(-0.5 * (w / (width + 1e-6)) ** 2)

    real_part = np.real(sig)
    imag_part = np.imag(sig)
    real_max = np.max(np.abs(real_part)) if np.any(real_part) else 0.0
    imag_max = np.max(np.abs(imag_part)) if np.any(imag_part) else 0.0
    overall_scale = max(real_max, imag_max, 1e-3)

    mostly_real = imag_max < 0.1 * max(real_max, 1e-6)

    if attempt % 2 == 0:
        amp_factor = 1.0 + 0.12 * (attempt + 1)
    else:
        amp_factor = 1.0 - 0.1 * (attempt + 1) / (attempt + 2)

    # Add a gentle tilt plus a localised bump so that the shape remains
    # believable but visibly different from the original
    tilt_strength = 0.03 * (attempt + 1)
    tilt = tilt_strength * (w / 5.0)

    bump_centres = [0.0, 2.5, -2.5, 1.5, -1.5, 3.5]
    centre = bump_centres[attempt % len(bump_centres)]
    width = 1.1 + 0.3 * (attempt % 3)
    bump = np.exp(-0.5 * ((w - centre) / width) ** 2)
    bump_strength = 0.15 * (attempt + 1) * overall_scale

    if mostly_real:
        new_real = amp_factor * real_part + bump_strength * bump + tilt * overall_scale
        new_imag = amp_factor * imag_part
        return new_real + 1j * new_imag

    # For real complex spectra adjust the magnitude slightly and apply a
    # tiny phase shift -
    mag = np.abs(sig)
    phase = np.angle(sig)
    mag *= amp_factor
    phase += 0.08 * (-1) ** attempt
    complex_perturbation = bump_strength * bump * np.exp(1j * phase)
    return mag * np.exp(1j * phase) + complex_perturbation



def _operation_parameter_label(op_name: str, param: str | None) -> str | None:
    """Return a text label describing an operation's parameter."""

    if op_name == "Multiplication":
        return _describe_multiplication_param(param)
    if op_name == "Filter":
        return _describe_filter_param(param)
    if op_name == "Sampling":
        return _describe_sampling_param(param)
    return None

def _operation_name_latex(op_name: str) -> str:
    """Return a LaTeX representation for an operation name."""

    if op_name == "Multiplication":
        return r"\text{Multiplication}"
    if op_name == "Hilbert":
        return r"\mathcal{H}"
    if op_name == "Sampling":
        return r"\text{Sampling}"
    if op_name == "Derivative":
        return r"\frac{d}{dt}"
    if op_name == "Addition":
        return r"\text{Adder}"
    if op_name == "Filter":
        return r"\text{Filter}"
    if op_name == "Real":
        return r"\Re"
    if op_name == "Imag":
        return r"\Im"
    return rf"\text{{{op_name}}}"


def _operation_parameter_label_latex(op_name: str, param: str | None) -> str | None:
    """Return a LaTeX representation for an operation parameter."""

    if op_name == "Multiplication":
        return _describe_multiplication_param_latex(param)
    if op_name == "Filter":
        return _describe_filter_param_latex(param)
    if op_name == "Sampling":
        return _describe_sampling_param_latex(param)
    return None

def _build_summary_latex(letter: str, signal_latex: str, op_name: str, param: str | None) -> str:
    """Return a LaTeX summary string for diagram metadata entries."""

    name_latex = _operation_name_latex(op_name)
    param_latex = _operation_parameter_label_latex(op_name, param)
    summary = rf"\mathbf{{{letter}}}:\; {signal_latex}\; \text{{after}}\; {name_latex}"
    if param_latex:
        summary += rf"\;({param_latex})"
    return summary




def _describe_multiplication_param(param: str | None) -> str | None:
    """Return a display string for the multiplication parameter."""

    if not param:
        return None

    raw = param.strip()
    lower = raw.lower()

    if lower.startswith("constant:"):
        value = _format_number(raw.split(":", 1)[1])
        return value

    if lower.startswith("imaginary"):
        parts = raw.split(":", 1)
        if len(parts) == 2:
            coeff = _format_number(parts[1])
            if coeff == "1":
                return "j"
            return f"{coeff}·j"
        return "j"

    if lower.startswith("linear:"):
        value = _format_number(raw.split(":", 1)[1])
        if value == "1":
            return "ω"
        return f"{value}·ω"

    if lower.startswith("sin:") or lower.startswith("cos:"):
        func = "sin" if lower.startswith("sin:") else "cos"
        _, rest = raw.split(":", 1)
        tokens = [t.strip() for t in rest.split(",") if t.strip()]
        jflag = False
        if tokens and tokens[0].lower() == "j":
            jflag = True
            tokens = tokens[1:]
        amp = _format_number(tokens[0]) if tokens else "1"
        freq = _format_number(tokens[1]) if len(tokens) > 1 else "1"
        amp_prefix = "" if amp == "1" else f"{amp}·"
        base = f"{amp_prefix}{func}({freq}·t)"
        return f"j·{base}" if jflag else base
    
    if lower.startswith("exponential:"):
        _, rest = raw.split(":", 1)
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        if len(parts) == 3:
            coeff, sign, freq = parts
        elif len(parts) == 2:
            coeff, sign, freq = "1", parts[0], parts[1]
        else:
            coeff, sign, freq = "1", "+", "1"
        sign_symbol = "+" if sign.startswith("+") else "-"
        coeff_fmt = _format_number(coeff)
        freq_fmt = _format_number(freq)
        coeff_prefix = "" if coeff_fmt == "1" else f"{coeff_fmt}·"
        return f"{coeff_prefix}exp({sign_symbol}j{freq_fmt}t)"

    return raw

def _describe_multiplication_param_latex(param: str | None) -> str | None:
    """Return a LaTeX display string for the multiplication parameter."""

    if not param:
        return None

    raw = param.strip()
    lower = raw.lower()

    if lower.startswith("constant:"):
        value = _format_number(raw.split(":", 1)[1])
        return _format_number_latex(value)

    if lower.startswith("imaginary"):
        parts = raw.split(":", 1)
        if len(parts) == 2:
            coeff = _format_number(parts[1])
            coeff_fmt = _format_number_latex(coeff)
            if coeff_fmt == "1":
                return "j"
            return rf"{coeff_fmt}\,j"
        return "j"

    if lower.startswith("linear:"):
        value = _format_number(raw.split(":", 1)[1])
        value_fmt = _format_number_latex(value)
        if value_fmt == "1":
            return r"\omega"
        return rf"{value_fmt}\,\omega"

    if lower.startswith("sin:") or lower.startswith("cos:"):
        func = r"\sin" if lower.startswith("sin:") else r"\cos"
        _, rest = raw.split(":", 1)
        tokens = [t.strip() for t in rest.split(",") if t.strip()]
        jflag = False
        if tokens and tokens[0].lower() == "j":
            jflag = True
            tokens = tokens[1:]
        amp = _format_number(tokens[0]) if tokens else "1"
        freq = _format_number(tokens[1]) if len(tokens) > 1 else "1"
        amp_fmt = _format_number_latex(amp)
        freq_fmt = _format_number_latex(freq)
        amp_prefix = "" if amp_fmt == "1" else rf"{amp_fmt}\,"
        base = rf"{amp_prefix}{func}({freq_fmt}\,t)"
        return rf"j\,{base}" if jflag else base
    
    if lower.startswith("exponential:"):
        _, rest = raw.split(":", 1)
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        if len(parts) == 3:
            coeff, sign, freq = parts
        elif len(parts) == 2:
            coeff, sign, freq = "1", parts[0], parts[1]
        else:
            coeff, sign, freq = "1", "+", "1"
        sign_symbol = "+" if sign.startswith("+") else "-"
        coeff_fmt = _format_number_latex(_format_number(coeff))
        freq_fmt = _format_number_latex(_format_number(freq))
        coeff_prefix = "" if coeff_fmt == "1" else rf"{coeff_fmt}\,"
        return rf"{coeff_prefix}e^{{{sign_symbol} j {freq_fmt}\,t}}"

    safe_raw = raw.replace("\\", r"\textbackslash ")
    return rf"\text{{{safe_raw}}}"



def _describe_filter_param(param: str | None) -> str | None:
    """Return a readable filter description."""

    if not param:
        return None

    raw = param.strip()
    lower = raw.lower()

    if lower.startswith("lowpass:"):
        value = _format_number(raw.split(":", 1)[1])
        return f"lowpass (ω_c={value})"

    if lower.startswith("highpass:"):
        value = _format_number(raw.split(":", 1)[1])
        return f"highpass (ω_c={value})"

    if lower.startswith("bandpass:"):
        _, rest = raw.split(":", 1)
        try:
            lo, hi = [part.strip() for part in rest.split(",", 1)]
        except ValueError:
            lo, hi = rest, ""
        lo_fmt = _format_number(lo)
        hi_fmt = _format_number(hi) if hi else ""
        return f"bandpass ({lo_fmt} – {hi_fmt})"

    return raw


def _describe_filter_param_latex(param: str | None) -> str | None:
    """Return a LaTeX string describing a filter parameter."""

    if not param:
        return None

    raw = param.strip()
    lower = raw.lower()

    if lower.startswith("lowpass:"):
        value = _format_number(raw.split(":", 1)[1])
        value_fmt = _format_number_latex(value)
        return rf"\text{{lowpass}}\; (\omega_c = {value_fmt})"

    if lower.startswith("highpass:"):
        value = _format_number(raw.split(":", 1)[1])
        value_fmt = _format_number_latex(value)
        return rf"\text{{highpass}}\; (\omega_c = {value_fmt})"

    if lower.startswith("bandpass:"):
        _, rest = raw.split(":", 1)
        try:
            lo, hi = [part.strip() for part in rest.split(",", 1)]
        except ValueError:
            lo, hi = rest, ""
        lo_fmt = _format_number_latex(_format_number(lo))
        hi_fmt = _format_number_latex(_format_number(hi)) if hi else ""
        if hi_fmt:
            return rf"\text{{bandpass}}\; ({lo_fmt} \leq |\omega| \leq {hi_fmt})"
        return rf"\text{{bandpass}}\; (|\omega| \geq {lo_fmt})"

    safe_raw = raw.replace("\\", r"\textbackslash ")
    return rf"\text{{{safe_raw}}}"


def _describe_sampling_param(param: str | None) -> str | None:
    """Return a readable description for a sampling parameter."""

    if not param:
        return "sampling (T=1)"

    raw = param.strip()
    lower = raw.lower()
    if lower.startswith("sampling:"):
        value = _format_number(raw.split(":", 1)[1])
        return f"sampling (T={value})"
    return raw


def _describe_sampling_param_latex(param: str | None) -> str | None:
    """Return a LaTeX version of a sampling parameter description."""

    if not param:
        return r"\text{sampling}\; (T = 1)"

    raw = param.strip()
    lower = raw.lower()
    if lower.startswith("sampling:"):
        value = _format_number_latex(_format_number(raw.split(":", 1)[1]))
        return rf"\text{{sampling}}\; (T = {value})"
    safe_raw = raw.replace("\\", r"\textbackslash ")
    return rf"\text{{{safe_raw}}}"


def _format_number(value: str) -> str:
    """Format a numeric string without unnecessary decimal places."""

    try:
        num = float(value)
    except (TypeError, ValueError):
        return value.strip()
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f"{num:g}"


def _format_number_latex(value: str) -> str:
    """Return a LaTeX-safe version of a formatted numeric string."""

    # ``_format_number`` already normalises decimal places, so we only need to
    # ensure surrounding whitespace is removed.
    return _format_number(value)