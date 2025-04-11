import numpy as np

def rect(t):
    """Rectangular function: Returns 1 for |t| < 0.5 and 0 otherwise."""
    return np.where(np.abs(t) < 0.5, 1.0, 0.0)

def tri(t):
    """Triangular function: Returns 1 - |t| for |t| <= 1 and 0 otherwise."""
    return np.where(np.abs(t) <= 1, 1 - np.abs(t), 0.0)

def step(t):
    """Step function (Heaviside): 1 for t >= 0, 0 for t < 0."""
    return np.heaviside(t, 1)

def cos(t):
    """Cosine function."""
    return np.cos(t)

def sin(t):
    """Sine function."""
    return np.sin(t)

def sign(t):
    """Sign function: -1 for t < 0, 1 for t > 0, 0 for t = 0."""
    return np.sign(t)

def delta(t):
    """Approximate delta function using a very narrow Gaussian."""
    epsilon = 1e-3  # Narrow width for approximation
    return np.exp(-t**2 / epsilon) / (np.sqrt(np.pi * epsilon))

def exp_iwt(t, omega_0=1.0):
    """Complex exponential: e^(j * ω₀ * t)."""
    return np.exp(1j * omega_0 * t)

def inv_t(t):
    """1/t function."""
    return np.where(t != 0, 1 / t, 0.0)

def si(t):
    """Sinc function: sin(πt) / (πt)."""
    return np.sinc(t)  # NumPy's sinc is normalized to π
