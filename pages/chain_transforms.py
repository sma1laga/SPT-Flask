import numpy as np

def rect(w, width=1.0):
    """
    Rectangular function (ideal lowpass shape):
    Returns 1 for |w| <= width/2 and 0 otherwise.
    """
    return np.where(np.abs(w) <= width/2, 1.0, 0.0)

def tri(w, width=1.0):
    """
    Triangular function:
    Maximum of 1 at w=0 and linearly decreases to 0 at |w| = width/2.
    """
    return np.where(np.abs(w) <= width/2, 1 - np.abs(w) / (width/2), 0.0)

def apply_addition(signal, param, w):
    try:
        value = float(param) if param is not None else 0.0
    except:
        value = 0.0
    return signal + value

def apply_subtraction(signal, param, w):
    try:
        value = float(param) if param is not None else 0.0
    except:
        value = 0.0
    return signal - value

def apply_multiplication(signal, param, w):
    """
    Multiply the signal by a user-specified factor.
    
    The user can input:
      - A constant (e.g., "3")
      - The imaginary unit "j" (e.g., "j")
      - A linear function (e.g., "2*w")
      - sin(w) or cos(w)
      - An exponential, e.g., "exp(-j*4*w)"
      - Sampling: if the parameter begins with "sampling:", the user can choose the sampling interval.
        For example, "sampling:2" creates a comb function (impulse train) in the frequency domain.
    """
    try:
        param = param.strip()
    except:
        param = ""
    # Check if sampling is requested.
    if param.lower().startswith("sampling:"):
        try:
            parts = param.split(":")
            sample_interval = float(parts[1])
            # Set a tolerance relative to the sampling interval.
            tol = sample_interval / 10.0
            # Create an impulse train: ones at frequencies that are multiples of sample_interval.
            factor = np.where(np.abs(w - np.round(w / sample_interval) * sample_interval) < tol, 1.0, 0.0)
        except Exception as e:
            raise Exception(f"Error parsing sampling interval in multiplication block: {e}")
    else:
        try:
            safe_dict = {
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "exp": np.exp,
                "log": np.log,
                "sqrt": np.sqrt,
                "rect": rect,
                "tri": tri,
                "w": w,
                "j": 1j
            }
            factor = eval(param, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            raise Exception(f"Error in multiplication block evaluating param '{param}': {e}")
    return signal * factor

def apply_derivative(signal, param, w):
    """
    In the Fourier domain, the derivative in time corresponds to multiplication by (j*w).
    """
    return 1j * w * signal

def apply_hilbert(signal, param, w):
    """
    The Hilbert transform in the frequency domain multiplies by -j*sgn(w).
    """
    return -1j * np.sign(w) * signal

def apply_filter(signal, param, w):
    """
    Ideal filter block.
    
    If no parameter is provided, defaults to a lowpass filter with cutoff 1.
    Valid parameter formats:
      "lowpass:cutoff"      e.g., "lowpass:2"
      "highpass:cutoff"     e.g., "highpass:3"
      "bandpass:low,high"   e.g., "bandpass:1,3"
    """
    if param is None or param.strip() == "":
        cutoff = 1.0
        mask = np.abs(w) <= cutoff
        return signal * mask
    else:
        s = param.strip().lower()
        if s.startswith("lowpass"):
            parts = s.split(":")
            try:
                cutoff = float(parts[1])
            except:
                cutoff = 1.0
            mask = np.abs(w) <= cutoff
            return signal * mask
        elif s.startswith("highpass"):
            parts = s.split(":")
            try:
                cutoff = float(parts[1])
            except:
                cutoff = 1.0
            mask = np.abs(w) >= cutoff
            return signal * mask
        elif s.startswith("bandpass"):
            parts = s.split(":")
            try:
                freqs = parts[1].split(",")
                low_cut = float(freqs[0])
                high_cut = float(freqs[1])
            except:
                low_cut = 0.5
                high_cut = 2.0
            mask = (np.abs(w) >= low_cut) & (np.abs(w) <= high_cut)
            return signal * mask
        else:
            cutoff = 1.0
            mask = np.abs(w) <= cutoff
            return signal * mask

def apply_generic(signal, param, w):
    if param is not None and param.strip() != "":
        try:
            safe_dict = {"np": np, "x": signal, "w": w, "rect": rect, "tri": tri}
            return eval(param, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            raise Exception(f"Error in generic block evaluating '{param}': {e}")
    return signal

def no_op(signal, param, w):
    return signal
