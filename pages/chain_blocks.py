# ------------------------- chain_blocks.py -------------------------
import io, base64
import matplotlib.pyplot as plt
import numpy as np
from .chain_transforms import (
    apply_addition, apply_subtraction, apply_multiplication,
    apply_derivative, apply_hilbert, apply_filter,
    apply_generic, no_op, rect, tri
)

def interpret_chain(chain_data, until_block=None):
    """
    Computes the process chain in the frequency domain and plots the output Y(jω),
    or up to `until_block` if specified (for partial plots on arrow dbl-click).
    """
    # (1) Evaluate the input function X(jω)
    input_expr = chain_data.get("input", "0")
    try:
        W = 10
        w = np.linspace(-W, W, 1024)
        safe_dict = {"np": np, "rect": rect, "tri": tri, "w": w}
        signal = eval(input_expr, {"__builtins__": {}}, safe_dict)
    except Exception as e:
        raise Exception(f"Error evaluating input expression '{input_expr}': {e}")

    # (2) Extract blocks and connection lines
    blocks = chain_data.get("blocks", [])
    lines  = chain_data.get("lines", [])
    if not blocks:
        raise Exception("No blocks provided in the process chain.")

    # (3) Identify the output block
    output_block = next((b for b in blocks
                         if b.get("label")=="y(t)" or b.get("type")=="Output"), None)
    if output_block is None:
        raise Exception("Process chain is incomplete: no output block (y(t)) found.")

    # Build lookup dictionaries
    blocks_by_id = {b["id"]: b for b in blocks}
    incoming     = {b["id"]: [] for b in blocks}
    outgoing     = {b["id"]: [] for b in blocks}
    for line in lines:
        fr = line.get("fromId"); to = line.get("toId")
        if fr in blocks_by_id and to in blocks_by_id:
            outgoing[fr].append(to)
            incoming[to].append(fr)

    # (4) Initialize signals at each node: start‐blocks get the input spectrum
    from collections import deque

    start_blocks = [bid for bid, inc in incoming.items() if len(inc) == 0]
    if not start_blocks:
        raise Exception("No starting block found; ensure at least one block has no incoming connections.")

    signals = {}
    for bid in start_blocks:
        signals[bid] = signal

    # (5) Topological sort to ensure each block is processed after its inputs
    in_deg = {bid: len(inc) for bid, inc in incoming.items()}
    queue  = deque(start_blocks)
    topo   = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in outgoing.get(u, []):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    # (6) Propagate through every block
    for bid in topo:
        if bid in start_blocks:
            continue

        # sum all incoming signals
        in_sigs = [signals[i] for i in incoming[bid]]
        x_sig   = sum(in_sigs)

        # apply this block’s transform
        blk   = blocks_by_id[bid]
        typ   = blk.get("type","")
        p     = blk.get("param")
        if   typ == "Addition":      y = apply_addition   (x_sig, p, w)
        elif typ == "Subtraction":   y = apply_subtraction(x_sig, p, w)
        elif typ == "Multiplication":y = apply_multiplication(x_sig, p, w)
        elif typ in ["Derivative","d/dt"]:
                                      y = apply_derivative (x_sig, p, w)
        elif typ == "Hilbert":       y = apply_hilbert    (x_sig, p, w)
        elif typ == "Filter":        y = apply_filter     (x_sig, p, w)
        else:                        y = apply_generic    (x_sig, p, w)

        signals[bid] = y

        # now if user requested a partial plot, stop here
        if until_block is not None and bid == until_block:
            break

        # if user requested a partial plot, stop here
        if until_block is not None and bid == until_block:
            signals[bid] = x_sig
            break

        blk   = blocks_by_id[bid]
        p     = blk.get("param")
        typ   = blk.get("type", "")

        if   typ == "Addition":      y = apply_addition   (x_sig, p, w)
        elif typ == "Subtraction":   y = apply_subtraction(x_sig, p, w)
        elif typ == "Multiplication":y = apply_multiplication(x_sig, p, w)
        elif typ in ["Derivative","d/dt"]:
                                     y = apply_derivative (x_sig, p, w)
        elif typ == "Hilbert":       y = apply_hilbert    (x_sig, p, w)
        elif typ == "Filter":        y = apply_filter     (x_sig, p, w)
        elif typ == "Re":            y = apply_generic    (x_sig, p, w)  # or apply_real?
        elif typ == "Im":            y = apply_generic    (x_sig, p, w)  # or apply_imag?
        else:                        y = apply_generic    (x_sig, p, w)

        signals[bid] = y

    # (7) Choose output: either the full chain or the partial point
    if until_block is not None:
        out_sig = signals.get(until_block)
        if out_sig is None:
            raise Exception(f"Block id {until_block} not found in propagation.")
    else:
        out_sig = signals[output_block["id"]]

    # (8) Plot real and imaginary parts
    realY = np.real(out_sig)
    imagY = np.imag(out_sig)

    plt.figure()
    plt.plot(w, realY, label="Real part of Y(jω)")
    if np.any(np.abs(imagY) > 1e-10):
        plt.plot(w, imagY, linestyle="dotted", color="orange",
                 label="Imaginary part of Y(jω)")
    plt.xlabel("Frequency ω")
    plt.ylabel("Amplitude")
    plt.title("Output Y(jω): Real and Imaginary Parts")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return encoded
# ------------------------- End of chain_blocks.py -------------------------
