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
    Computes the process chain in the frequency domain and plots the output spectrum.
    If `until_block` is provided, plots the intermediate output at that block.
    The plot title updates to show the block's connection letter (e.g. 'A(jω)').
    """
    # (1) Prepare frequency axis
    W = 10
    w = np.linspace(-W, W, 1024)

    # (2) Evaluate input X(jω)
    input_expr = chain_data.get("input", "0")
    try:
        safe_dict = {"np": np, "rect": rect, "tri": tri, "w": w}
        signal = eval(input_expr, {"__builtins__": {}}, safe_dict)
    except Exception as e:
        raise Exception(f"Error evaluating input expression '{input_expr}': {e}")

    # (3) Extract blocks and connections
    blocks = chain_data.get("blocks", [])
    lines = chain_data.get("lines", [])
    if not blocks:
        raise Exception("No blocks provided in the process chain.")

    # map blocks and build adjacency
    blocks_by_id = {b["id"]: b for b in blocks}
    incoming = {b["id"]: [] for b in blocks}
    outgoing = {b["id"]: [] for b in blocks}
    for line in lines:
        fr = line.get("fromId"); to = line.get("toId")
        if fr in blocks_by_id and to in blocks_by_id:
            outgoing[fr].append(to)
            incoming[to].append(fr)

    # (4) Find starting blocks (no incoming)
    start_blocks = [bid for bid, inc in incoming.items() if not inc]
    if not start_blocks:
        raise Exception("No starting block found; ensure at least one block has no incoming connections.")

    # initialize signals
    signals = {bid: signal for bid in start_blocks}

    # (5) Topological sort
    from collections import deque
    in_deg = {bid: len(incoming[bid]) for bid in blocks_by_id}
    queue = deque(start_blocks)
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in outgoing.get(u, []):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    # (6) Propagate
    for bid in topo:
        if bid in start_blocks:
            continue
        in_sigs = [signals[i] for i in incoming[bid]]
        x_sig = sum(in_sigs)
        blk = blocks_by_id[bid]
        typ = blk.get("type","")
        p = blk.get("param")
        # apply block transform
        if   typ == "Addition":       y = apply_addition(x_sig, p, w)
        elif typ == "Subtraction":    y = apply_subtraction(x_sig, p, w)
        elif typ == "Multiplication": y = apply_multiplication(x_sig, p, w)
        elif typ in ["Derivative","d/dt"]: y = apply_derivative(x_sig, p, w)
        elif typ == "Hilbert":        y = apply_hilbert(x_sig, p, w)
        elif typ == "Filter":         y = apply_filter(x_sig, p, w)
        else:                          y = apply_generic(x_sig, p, w)
        signals[bid] = y
        if until_block is not None and bid == until_block:
            break

    # (7) Determine output signal and title
    if until_block is not None:
        out_sig = signals.get(until_block)
        # find connection letter for this block
        letter = None
        for line in lines:
            if line.get("fromId") == until_block:
                letter = line.get("letter")
                break
        title = f"{letter.upper()}(jω)" if letter else "Y(jω)"
    else:
        # full chain output
        output_block = next((b for b in blocks if b.get("label")=="y(t)" or b.get("type")=="Output"), None)
        if output_block is None:
            raise Exception("No output block found for full-chain computation.")
        out_sig = signals.get(output_block["id"])
        title = "Y(jω)"

    # (8) Plot
    realY = np.real(out_sig)
    imagY = np.imag(out_sig)
    plt.figure()
    plt.plot(w, realY, label="Real part")
    if np.any(np.abs(imagY) > 1e-10):
        plt.plot(w, imagY, linestyle="dotted", label="Imaginary part")
    plt.xlabel("Frequency ω")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()

    # (9) Encode image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return encoded
# ------------------------- End of chain_blocks.py -------------------------
