# pages/chain_blocks.py
import io, base64
import matplotlib.pyplot as plt
import numpy as np

def interpret_chain(chain_data):
    """
    TEMPORARY STUB:
    Given the user's chain_data from the front end:
      {
        "input": "some input expression, e.g. sin(t)",
        "blocks": [...list of blocks with type, param, x,y...],
        "lines": [...list of connections...]
      }
    we apply the 'chain' logic in frequency domain and produce a final result.

    For now, we do a dummy result: plot a simple parabola, return it as base64.
    """

    # (1) parse chain_data["input"], chain_data["blocks"], chain_data["lines"]
    # (2) do real logic. For now, just do a dummy plot.
    x = np.linspace(0, 1, 100)
    y = x**2

    # Make a quick plot
    plt.figure()
    plt.plot(x, y, label="Placeholder")
    plt.title("Dummy Frequency Spectrum")
    plt.legend()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return encoded
