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

def delta(t, eps=1e-4, normalize: bool = False):
    """Approximate Dirac delta via narrow Gaussian and
    normalize maximum to 1 if `normalize` is True."""
    delta_out = np.exp(-t**2 / eps) / np.sqrt(np.pi * eps)
    if normalize:
        abs_max = np.max(np.abs(delta_out))
        if abs_max != 0:
            delta_out /= abs_max
    return delta_out

def delta_train(t, spacing: float = 1.0, count: int = 17) -> np.ndarray:
    """Finite train of equally spaced deltas: ∑ δ(t - k*spacing) for k=-((count-1)//2) ... count//2
    Args:
        t (np.ndarray): Time samples where the train is evaluated
        spacing (float): Time spacing between deltas
        count (int): Number of deltas (odd number recommended, at least 1).
    Returns:
        np.ndarray: Sum of ``count`` shifted delta approximations.
    """
    count = int(max(1, count))

    offsets = (np.arange(count) - (count-1)//2) * spacing
    t = np.asarray(t)
    train = delta(t[..., None] - offsets)
    return train.sum(axis=-1)

def exp_iwt(t, omega_0=1.0):
    return np.exp(1j * omega_0 * t)

def inv_t(t):
    return np.where(t != 0, 1 / t, 0.0)

def si(t):
    """Un-normalized sinc: sin(t)/t, with si(0)=1."""
    return np.where(t == 0, 1.0, np.sin(t) / t)

def sinc(t):
    return np.sinc(t)

def delta_n(n):
    """Kronecker delta: 1 when n==0 else 0."""
    return np.where(np.isclose(n.round(8), 0.0), 1.0, 0.0)

# example
def delta_train_n(k, spacing: int = 6, count: int = 9) -> np.ndarray:
    """Finite delta train centred around the origin
    Args:
        k (np.ndarray): Discrete time samples where the train is evaluated
        spacing (int): Separation between adjacent impulses.  Values are rounded to the nearest integer to keep impulses aligned with the integer grid.
        count (int): Number of impulses to include in the train (min. 1).
    Returns:
        np.ndarray: Sum of `count` shifted Kronecker deltas.
    """
    k = np.asarray(k)
    count = int(max(1, count))
    spacing = int(max(1, round(spacing)))

    offsets = (np.arange(count) - (count - 1)//2) * spacing
    train = np.zeros_like(k, dtype=float)
    for shift in offsets:
        train += delta_n(k - shift)
    return train
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
