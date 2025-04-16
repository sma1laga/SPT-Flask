# ------------------------- chain_blocks.py -------------------------
import io, base64
import matplotlib.pyplot as plt
import numpy as np
from .chain_transforms import (apply_addition, apply_subtraction, apply_multiplication,
                               apply_derivative, apply_hilbert, apply_filter,
                               apply_generic, no_op, rect, tri)

def interpret_chain(chain_data):
    """
    Computes the process chain in the frequency domain and plots the output Y(jω).
    
    The chain_data is expected to have the structure:
      {
        "input": "frequency domain expression, e.g. rect(w)" or "tri(w)",
        "blocks": [
           {"id": 1, "type": "Input", "label": "x(t)"},
           {"id": 2, "type": "Addition", "label": "+", "param": "0.5"},
           {"id": 3, "type": "Hilbert", "label": "Hilbert"},
           {"id": 4, "type": "Derivative", "label": "d/dt"},
           {"id": 5, "type": "Filter", "label": "Filter", "param": "lowpass:2"},
           ...,
           {"id": N, "type": "Output", "label": "y(t)"}
        ],
        "lines": [
           {"fromId": 1, "toId": 2},
           {"fromId": 2, "toId": 3},
           {"fromId": 3, "toId": 4},
           {"fromId": 4, "toId": 5},
           ...
        ]
      }
    
    This function:
      1. Evaluates the user-supplied input frequency function X(jω) over a frequency vector w.
      2. Constructs the connectivity graph from blocks and connection lines.
      3. Finds a valid path from a starting block (no incoming connections) to the output block ("y(t)").
      4. Sequentially applies each block’s transformation to the frequency-domain signal.
      5. Plots the output Y(jω) by drawing the real part as a solid line and the imaginary part (if nonzero) 
         as an orange dotted line.
      6. Returns the plot as a base64-encoded PNG image.
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
    
    # (2) Extract blocks and connection lines.
    blocks = chain_data.get("blocks", [])
    lines = chain_data.get("lines", [])
    if not blocks:
        raise Exception("No blocks provided in the process chain.")
    
    # (3) Identify the output block.
    output_block = None
    for b in blocks:
        if b.get("label") == "y(t)" or b.get("type") == "Output":
            output_block = b
            break
    if output_block is None:
        raise Exception("Process chain is incomplete: no output block (y(t)) found.")
    
    # Build lookup dictionaries.
    blocks_by_id = {b["id"]: b for b in blocks}
    incoming = {b["id"]: [] for b in blocks}
    outgoing = {b["id"]: [] for b in blocks}
    for line in lines:
        fr = line.get("fromId")
        to = line.get("toId")
        if fr in blocks_by_id and to in blocks_by_id:
            outgoing[fr].append(to)
            incoming[to].append(fr)
    
    # (4) Identify starting blocks (those with no incoming connections).
    start_blocks = [bid for bid, inc in incoming.items() if len(inc) == 0]
    if not start_blocks:
        raise Exception("No starting block found; ensure at least one block has no incoming connections.")
    
    # (5) Find a path from a starting block to the output block using BFS.
    from collections import deque
    start_block_id = start_blocks[0]  # Pick the first starting block.
    queue = deque([start_block_id])
    visited = {start_block_id}
    parent = {start_block_id: None}
    found = False
    while queue:
        current = queue.popleft()
        if current == output_block["id"]:
            found = True
            break
        for neighbor in outgoing.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    if not found:
        raise Exception("Process chain is incomplete: the input is not connected to output (y(t)).")
    
    # Reconstruct the path from starting block to output block.
    path = []
    node = output_block["id"]
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path = path[::-1]
    
    # (6) Process the signal through each block in the chain path.
    current_signal = signal
    for bid in path:
        block = blocks_by_id[bid]
        btype = block.get("type", "")
        param = block.get("param")
        if btype == "Addition":
            current_signal = apply_addition(current_signal, param, w)
        elif btype == "Subtraction":
            current_signal = apply_subtraction(current_signal, param, w)
        elif btype == "Multiplication":
            current_signal = apply_multiplication(current_signal, param, w)
        elif btype in ["Derivative", "d/dt"]:
            current_signal = apply_derivative(current_signal, param, w)
        elif btype == "Hilbert":
            current_signal = apply_hilbert(current_signal, param, w)
        elif btype == "Filter":
            current_signal = apply_filter(current_signal, param, w)
        elif btype == "Block":
            current_signal = apply_generic(current_signal, param, w)
        else:
            current_signal = no_op(current_signal, param, w)
    
    # (7) Plot the resulting Y(jω).
    realY = np.real(current_signal)
    imagY = np.imag(current_signal)
    
    plt.figure()
    plt.plot(w, realY, label="Real part of Y(jω)")
    # Plot the imaginary part as an orange dotted line, if it is non-negligible.
    if np.any(np.abs(imagY) > 1e-10):
        plt.plot(w, imagY, linestyle="dotted", color="orange", label="Imaginary part of Y(jω)")
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
