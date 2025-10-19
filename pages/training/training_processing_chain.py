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
        else:
            return jsonify({"error": "Only EASY and MEDIUM difficulties are implemented."}), 400
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

    # Branch operations: Multiplication or Sampling on each split
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

    # Distractor helpers for each stage
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

    # Diagram metadata
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

    diagram_img = _draw_medium_diagram(branch_ops, post_op, post_param, filter_param)
    input_plot = _plot_spectrum(w, x_sig, title=r"$X(j\omega)$")

    return {
        "diagram": diagram_img,
        "diagramOperations": diagram_ops,
        "inputExpression": input_expr,
        "inputPlot": input_plot,
        "operations": [
            {"type": op, "param": param} for op, param in branch_ops
        ]
        + [
            {"type": "Addition", "param": None},
            {"type": post_op, "param": post_param},
            {"type": "Filter", "param": filter_param},
        ],
        "letters": letter_results,
        "inputExpressionLatex": input_expr_latex,
    }

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
        tokens = p.split(":")[1].split(",")
        A = float(tokens[0]) if tokens and tokens[0] else 1.0
        # parse the frequency shift value (use ASCII name w0 instead of unicode omega)
        w0 = float(tokens[1]) if len(tokens) > 1 else 1.0
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
        return y
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


def _draw_diagram(sequence: Tuple[str, str, str], params: Dict[str, str | None]) -> str:
    """
    Draw a block diagram that mirrors the processing chain presentation.

    Multiplication is shown as a circle with a multiplication symbol and the
    intermediate signals are labelled x(t) → a(t) → b(t) → c(t) along the
    arrows to match the training flow.
    """
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")

    # Position the three blocks evenly on the canvas
    n_blocks = len(sequence)
    start = 1.8
    spacing = 2.7
    block_centres = [start + i * spacing for i in range(n_blocks)]
    y_pos = 1.3

    signal_labels = ["x(t)"] + [f"{chr(ord('a') + i)}(t)" for i in range(n_blocks)]
    letter_labels = [chr(ord("A") + i) for i in range(n_blocks)]

    arrow_props = dict(arrowstyle="->", lw=1.4, color="#111111")

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
            circle = plt.Circle((block["centre"], y_pos), block["radius"], fill=False, lw=1.6, color="#111111")
            ax.add_patch(circle)
            ax.text(block["centre"], y_pos, block["label"], ha="center", va="center", fontsize=16, fontweight="bold")
            if block.get("param"):
                param_text_y = y_pos + block["radius"] + 0.6
                ax.annotate(
                    block["param"],
                    xy=(block["centre"], y_pos + block["radius"] * 0.35),
                    xytext=(block["centre"], param_text_y),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#111111"),
                )
        else:
            rect = plt.Rectangle(
                (block["left"], y_pos - block["height"] / 2.0),
                block["width"],
                block["height"],
                fill=False,
                lw=1.6,
                color="#111111",
                joinstyle="round",
            )
            ax.add_patch(rect)
            ax.text(block["centre"], y_pos + 0.18, block["label"], ha="center", va="center", fontsize=11, fontweight="bold")
            if block.get("param"):
                ax.text(
                    block["centre"],
                    y_pos - 0.18,
                    block["param"],
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
    ax.set_ylim(0.2, 2.4)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _draw_medium_diagram(
    branch_ops: List[Tuple[str, str | None]],
    post_op: str,
    post_param: str | None,
    filter_param: str,
) -> str:
    """Draw the MEDIUM difficulty diagram with two-branch structure."""

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.axis("off")

    arrow_props = dict(arrowstyle="->", lw=1.4, color="#111111")

    y_mid = 1.6
    top_y = 2.6
    bot_y = 0.6
    input_x = 0.8
    split_x = 1.6
    branch_x = 3.2
    adder_x = 5.2
    post_x = 7.2
    filter_x = 9.1
    output_x = 10.4

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
            circle = plt.Circle((block["centre"], block["y"]), block["radius"], fill=False, lw=1.6, color="#111111")
            ax.add_patch(circle)
            ax.text(block["centre"], block["y"], block["label"], ha="center", va="center", fontsize=16, fontweight="bold")
            if block.get("param"):
                ax.annotate(
                    block["param"],
                    xy=(block["centre"], block["y"] + block["radius"] * 0.4),
                    xytext=(block["centre"], block["y"] + block["radius"] + 0.6),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#111111"),
                )
        else:
            rect = plt.Rectangle(
                (block["left"], block["bottom"]),
                block["width"],
                block["height"],
                fill=False,
                lw=1.6,
                color="#111111",
                joinstyle="round",
            )
            ax.add_patch(rect)
            ax.text(block["centre"], block["y"] + 0.18, block["label"], ha="center", va="center", fontsize=11, fontweight="bold")
            if block.get("param"):
                ax.text(block["centre"], block["y"] - 0.18, block["param"], ha="center", va="center", fontsize=10)

    for block in blocks:
        draw_block(block)

    # Input and splitter
    ax.annotate("", xy=(split_x, y_mid), xytext=(input_x, y_mid), arrowprops=arrow_props)
    ax.text((input_x + split_x) / 2.0, y_mid + 0.45, "$x(t)$", ha="center", va="center", fontsize=12)
    split_circle = plt.Circle((split_x, y_mid), 0.06, color="#111111")
    ax.add_patch(split_circle)

    # Branch connections
    ax.plot([split_x, split_x], [y_mid, top_block["y"]], color="#111111", lw=1.4)
    ax.plot([split_x, split_x], [y_mid, bot_block["y"]], color="#111111", lw=1.4)
    ax.annotate("", xy=(top_block["left"], top_block["y"]), xytext=(split_x, top_block["y"]), arrowprops=arrow_props)
    ax.annotate("", xy=(bot_block["left"], bot_block["y"]), xytext=(split_x, bot_block["y"]), arrowprops=arrow_props)

    # Branch outputs into adder
    adder_radius = adder_block.get("radius", 0.45)
    top_target = (adder_block["centre"], y_mid + adder_radius / 1.2)
    bot_target = (adder_block["centre"], y_mid - adder_radius / 1.2)

    top_knee_x = (top_block["right"] + adder_block["left"]) / 2.0
    bot_knee_x = (bot_block["right"] + adder_block["left"]) / 2.0

    # Draw Manhattan style connections from branches into the adder
    ax.plot([top_block["right"], top_knee_x], [top_block["y"], top_block["y"]], color="#111111", lw=1.4)
    ax.plot([top_knee_x, top_knee_x], [top_block["y"], top_target[1]], color="#111111", lw=1.4)
    ax.annotate("", xy=top_target, xytext=(top_knee_x, top_target[1]), arrowprops=arrow_props)

    ax.plot([bot_block["right"], bot_knee_x], [bot_block["y"], bot_block["y"]], color="#111111", lw=1.4)
    ax.plot([bot_knee_x, bot_knee_x], [bot_block["y"], bot_target[1]], color="#111111", lw=1.4)
    ax.annotate("", xy=bot_target, xytext=(bot_knee_x, bot_target[1]), arrowprops=arrow_props)

    # Labels for branches
    mid_top_x = (top_knee_x + top_target[0]) / 2.0
    mid_top_y = (top_target[1] + top_block["y"]) / 2.0
    ax.text(mid_top_x, mid_top_y + 0.25, "A", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_top_x, mid_top_y - 0.35, "$a(t)$", ha="center", va="center", fontsize=11)

    mid_bot_x = (bot_knee_x + bot_target[0]) / 2.0
    mid_bot_y = (bot_target[1] + bot_block["y"]) / 2.0
    ax.text(mid_bot_x, mid_bot_y + 0.25, "B", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_bot_x, mid_bot_y - 0.35, "$b(t)$", ha="center", va="center", fontsize=11)

    # Adder to post block
    ax.annotate("", xy=(post_block["left"], y_mid), xytext=(adder_block["right"], y_mid), arrowprops=arrow_props)
    mid_c_x = (adder_block["right"] + post_block["left"]) / 2.0
    ax.text(mid_c_x, y_mid + 0.4, "C", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_c_x, y_mid - 0.55, "$c(t)$", ha="center", va="center", fontsize=11)

    # Post block to filter
    ax.annotate("", xy=(filter_block["left"], y_mid), xytext=(post_block["right"], y_mid), arrowprops=arrow_props)
    mid_d_x = (post_block["right"] + filter_block["left"]) / 2.0
    ax.text(mid_d_x, y_mid + 0.4, "D", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_d_x, y_mid - 0.55, "$d(t)$", ha="center", va="center", fontsize=11)

    # Filter to output
    ax.annotate("", xy=(output_x, y_mid), xytext=(filter_block["right"], y_mid), arrowprops=arrow_props)
    mid_e_x = (filter_block["right"] + output_x) / 2.0
    ax.text(mid_e_x, y_mid + 0.4, "E", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(mid_e_x, y_mid - 0.55, "$e(t)$", ha="center", va="center", fontsize=11)
    ax.text(output_x + 0.6, y_mid, "$y(t)$", ha="center", va="center", fontsize=12)

    ax.set_xlim(0.2, output_x + 1.2)
    ax.set_ylim(bot_y - 0.6, top_y + 0.9)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")



def _block_render_info(op_name: str, param: str | None) -> Dict[str, object]:
    """Return geometry and labels for rendering a block."""

    if op_name == "Multiplication":
        radius = 0.45
        return {
            "shape": "circle",
            "radius": radius,
            "left": None,  # filled in later
            "right": None,
            "label": "×",
            "param": _describe_multiplication_param(param),
        }

    if op_name == "Addition":
        radius = 0.45
        return {
            "shape": "circle",
            "radius": radius,
            "left": None,
            "right": None,
            "label": "+",
            "param": None,
        }

    width = 1.6
    height = 0.9
    if op_name == "Filter":
        label = "Filter"
        param_text = _describe_filter_param(param)
    elif op_name == "Hilbert":
        label = "Hilbert"
        param_text = None
    elif op_name == "Sampling":
        label = "Sampling"
        param_text = _describe_sampling_param(param)
    elif op_name == "Derivative":
        label = "Derivative"
        param_text = None
    else:
        label = op_name
        param_text = None

    return {
        "shape": "rect",
        "width": width,
        "height": height,
        "left": None,
        "right": None,
        "label": label,
        "param": param_text,
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
        tokens = [t.strip() for t in rest.split(",")]
        amp = _format_number(tokens[0]) if tokens and tokens[0] else "1"
        freq = _format_number(tokens[1]) if len(tokens) > 1 else "1"
        amp_prefix = "" if amp == "1" else f"{amp}·"
        return f"{amp_prefix}{func}({freq}·t)"

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
        tokens = [t.strip() for t in rest.split(",")]
        amp = _format_number(tokens[0]) if tokens and tokens[0] else "1"
        freq = _format_number(tokens[1]) if len(tokens) > 1 else "1"
        amp_fmt = _format_number_latex(amp)
        freq_fmt = _format_number_latex(freq)
        amp_prefix = "" if amp_fmt == "1" else rf"{amp_fmt}\,"
        return rf"{amp_prefix}{func}({freq_fmt}\,t)"

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