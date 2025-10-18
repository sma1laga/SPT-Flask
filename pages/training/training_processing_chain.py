"""
Blueprint and problem generator for the processing‑chain training module.

This module exposes a simple Flask blueprint that serves a template for the
processing chain training page and provides endpoints to generate practice
problems on demand.  Problems are generated at three difficulty levels, but
for now only the ``EASY`` level is implemented.  Each generated problem
consists of a small block diagram with three processing operations
(multiplication, Hilbert transform and filtering) arranged in one of three
predefined layouts.  A random input spectrum (rectangular or triangular) is
chosen and the operations are parameterised randomly.  The server then
computes the correct frequency‑domain output after each block as well as
plausible distractors by varying one parameter at a time.  Results are
returned to the client as base64‑encoded PNG images along with the correct
answer index for each connection.

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

    The client sends JSON with a ``difficulty`` field.  For the ``EASY``
    difficulty a random layout and random operation parameters are selected.
    The returned JSON has the following structure:

    ``diagram`` – a base64‑encoded PNG image of the block diagram.
    ``letters`` – a list of objects, one per connection (A, B, C).  Each
      object contains the letter name, a list of three base64 image strings
      and the index (0,1,2) of the correct image in that list.

    If an unsupported difficulty is requested an error message is returned.
    """
    data = request.get_json(force=True) or {}
    difficulty: str = str(data.get("difficulty", "EASY")).upper()

    if difficulty != "EASY":
        return jsonify({"error": "Only EASY difficulty is implemented."}), 400

    try:
        problem = _create_easy_problem()
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
    input_expr = f"{amp}*{input_shape}(w/{width})"

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
    cur_sig = x_sig.copy()
    for i, op_name in enumerate(op_sequence):
        # Apply correct operation
        param = params[op_name]
        cur_sig = _apply_operation(cur_sig, op_name, param, w)
        correct = cur_sig.copy()

        # Generate two distractors by varying the current operation only
        distractors = []
        if op_name == "Multiplication":
            # two new multiplication params
            for _ in range(2):
                alt_param = _random_multiplication_param(exclude=mul_param)
                alt_sig = _apply_operation(x_sig if i == 0 else letter_results[-1]["correct_sig"], op_name, alt_param, w)
                distractors.append(alt_sig)
        elif op_name == "Hilbert":
            # distractor 1: skip Hilbert
            alt_sig1 = (cur_sig.copy() if i == 0 else letter_results[-1]["correct_sig"].copy())
            # distractor 2: apply Hilbert on the input rather than at this point
            alt_sig2 = _apply_operation(x_sig, "Hilbert", None, w)
            distractors.extend([alt_sig1, alt_sig2])
        elif op_name == "Filter":
            # generate two different filters
            for _ in range(2):
                alt_param = _random_filter_param(exclude=fil_param)
                alt_sig = _apply_operation(letter_results[-1]["correct_sig"], "Filter", alt_param, w)
                distractors.append(alt_sig)
        else:
            # fallback: copy signal twice
            distractors.extend([correct.copy(), correct.copy()])

        # Shuffle options and determine correct index
        options = [correct] + distractors
        indices = list(range(3))
        random.shuffle(indices)
        shuffled = [options[j] for j in indices]
        correct_index = indices.index(0)

        # Encode each option as base64 image
        encoded_imgs = [
            _plot_spectrum(w, sig, title=f"{chr(ord('A')+i)}(jω)") for sig in shuffled
        ]
        letter_results.append(
            {
                "letter": chr(ord("A") + i),
                "images": encoded_imgs,
                "correctIndex": correct_index,
                "correct_sig": correct,  # keep for next stage computation
            }
        )

    # Draw block diagram
    diagram_img = _draw_diagram(op_sequence)

    # Strip internal signal objects before serialising
    for item in letter_results:
        item.pop("correct_sig", None)

    return {
        "diagram": diagram_img,
        "inputExpression": input_expr,
        # convert tuple to list for JSON serialisation
        "operations": list(op_sequence),
        "letters": letter_results,
    }


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
    elif op_name == "Hilbert":
        return _apply_hilbert(signal, w)
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


def _apply_hilbert(signal: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply a Hilbert transform in the frequency domain."""
    # Hilbert transform: multiply by -j*sign(w)
    sign_w = np.sign(w)
    return -1j * sign_w * signal


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
    ax.set_xlabel("Frequency ω")
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
    ax.grid(True, which="both", ls=":", lw=0.5)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _draw_diagram(sequence: Tuple[str, str, str]) -> str:
    """
    Draw a simple block diagram for the given sequence of operations.

    ``sequence`` is a tuple of three operation names corresponding to the
    blocks.  The diagram is drawn horizontally with labellsed arrows for
    connections A, B and C.  The returned value is a base64 PNG
    """
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.axis("off")
    # Coordinates for blocks
    x_positions = [0.5, 2.5, 4.5]
    block_labels = list(sequence)
    # Draw input label
    ax.text(0.0, 0.5, "$x(t)$", ha="center", va="center", fontsize=12)
    # Draw blocks and arrows
    for i, (x, lbl) in enumerate(zip(x_positions, block_labels)):
        # rectangle
        rect = plt.Rectangle((x - 0.4, 0.3), 0.8, 0.4, fill=False, lw=1.5)
        ax.add_patch(rect)
        ax.text(x, 0.5, lbl, ha="center", va="center", fontsize=10)
        # arrow from previous
        if i == 0:
            ax.annotate("", xy=(x - 0.8, 0.5), xytext=(0.2, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
            # label letter on this connection is after this block, handled below
        else:
            ax.annotate("", xy=(x - 0.8, 0.5), xytext=(x_positions[i - 1] + 0.4, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
        # letter label above arrow (A, B, C)
        letter = chr(ord("A") + i)  # A for first connection, etc.
        # Place letter halfway between this block and previous
        x_letter = (x_positions[i - 1] + 0.4 + (x - 0.8)) / 2.0 if i > 0 else (0.2 + (x - 0.8)) / 2.0
        ax.text(x_letter, 0.7, letter, ha="center", va="center", fontsize=12, fontweight="bold")
    # final arrow to output
    x_end = x_positions[-1] + 0.4
    ax.annotate("", xy=(x_end + 0.8, 0.5), xytext=(x_end, 0.5),
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(x_end + 1.0, 0.5, "$y(t)$", ha="center", va="center", fontsize=12)
    # Save diagram
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
