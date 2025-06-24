import numpy as np

# ------------------------------------------------------------------
# Continuous‑time helpers (unchanged)
# ------------------------------------------------------------------

def rect(t):
    """Continuous rectangular: 1 for |t| < 0.5 else 0."""
    return np.where(np.abs(t) < 0.5, 1.0, 0.0)

def tri(t):
    """Continuous triangular: 1−|t| for |t| ≤ 1 else 0."""
    return np.where(np.abs(t) <= 1, 1 - np.abs(t), 0.0)

def step(t):
    """Heaviside step (1 for t ≥ 0)."""
    return np.heaviside(t, 1)

def cos(t, t_norm: float = 1.):
    return np.cos(t_norm * t)

def sin(t, t_norm: float = 1.):
    return np.sin(t_norm * t)

def sign(t):
    return np.sign(t)

def delta(t):
    """Approximate Dirac delta via narrow Gaussian."""
    eps = 1e-3
    return np.exp(-t**2 / eps) / np.sqrt(np.pi * eps)

def exp_iwt(t, omega_0=1.0):
    return np.exp(1j * omega_0 * t)

def inv_t(t):
    return np.where(t != 0, 1 / t, 0.0)

def si(t):
    return np.sinc(t)

def dsi(t):
    """Un-normalized sinc: sin(t)/t, with si(0)=1."""
    return np.where(t == 0, 1.0, np.sin(t) / t)
