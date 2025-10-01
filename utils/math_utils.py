import numpy as np

# ------------------------------------------------------------------
# Continuous‑time helpers
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
def delta_train(t, spacing: float = 1.0, count: int = 6):
    """Finite train of equally spaced deltas

    Args:
        t (np.ndarray): Time samples where the train is evaluated
        spacing (float): Separation between2 deltas
        count (int): Number of deltas in the train (clipped to >=1).

    Returns:
        np.ndarray: Sum of ``count`` shifted delta approximations whose
            individual peaks match :func:`delta`.
    """

    count = int(max(1, count))
    spacing = float(spacing)

    # Center the train around 0 so it fits well in the plotting window.
    offsets = (np.arange(count) - (count - 1) / 2.0) * spacing

    # Reuse the same narrow Gaussian shape as ``delta`` for each impulse.
    eps = 1e-3
    t = np.asarray(t)
    diff = t[..., None] - offsets
    train = np.exp(-diff**2 / eps) / np.sqrt(np.pi * eps)
    return train.sum(axis=-1)
def exp_iwt(t, omega_0=1.0):
    return np.exp(1j * omega_0 * t)

def inv_t(t):
    return np.where(t != 0, 1 / t, 0.0)

def si(t):
    return np.sinc(t)

def dsi(t):
    """Un-normalized sinc: sin(t)/t, with si(0)=1."""
    return np.where(t == 0, 1.0, np.sin(t) / t)

def delta_n(n):
    """Kronecker delta: 1 when n==0 else 0."""
    return np.where(np.isclose(n.round(8), 0.0), 1.0, 0.0)

# ------------------------------------------------------------------
# Discrete-time helpers
# ------------------------------------------------------------------

def rect_N(k:np.ndarray, N:int) -> np.ndarray:
    """Length N discrete rectangle: 1 for 0≤k≤N-1, else 0.
    
    Arguments:
        k (np.ndarray): Time array (only integer timesteps will be nonzero).
        N (int): Length of the rectangle.
    Returns:
        np.ndarray: Rectangular sequence."""
    return np.where((k >= 0) & (k <= N - 1), 1.0, 0.0)

def tri_N(k:np.ndarray, N:int) -> np.ndarray:
    """Length 2N-1 triangular sequence: 0,1,...,N-1, N, N-1,...,1,0 around k=0.
    Arguments:
        k (np.ndarray): Time array (only integer timesteps will be nonzero).
        N (int): length of the positive triangle part.
    Returns:
        np.ndarray: Triangular sequence.
    """
    abs_k = np.abs(k).round(8)
    return np.where(abs_k <= N, N - abs_k, 0.)
