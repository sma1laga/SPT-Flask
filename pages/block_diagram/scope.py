import numpy as np
import control

MAX_POINTS = 4000
DEFAULT_HORIZON = 10.0
MIN_HORIZON = 1.0
MAX_HORIZON = 120.0
SLOW_POLE_EPS = 1e-6


def decimate(t, y, max_points=MAX_POINTS):
    """Downsample ``t`` and ``y`` to at most ``max_points`` samples."""
    if len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points, dtype=int)
        return t[idx], y[idx]
    return t, y


def quick_stats(y: np.ndarray) -> dict:
    """Return min/max/mean/RMS of a signal array."""
    return {
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "mean": float(np.mean(y)),
        "rms": float(np.sqrt(np.mean(np.square(y)))),
    }


def control_metrics(sys: control.TransferFunction) -> dict:
    """Compute standard control metrics for a step response."""
    try:
        info = control.step_info(sys)
    except Exception:
        info = {}
    return {
        "rise_time": float(info.get("RiseTime", float("nan"))),
        "settling_time": float(info.get("SettlingTime", float("nan"))),
        "overshoot": float(info.get("Overshoot", float("nan"))),
        "peak_time": float(info.get("PeakTime", float("nan"))),
        "final_value": float(info.get("SteadyStateValue", float("nan"))),
    }


def time_axis_for_step(sys: control.TransferFunction,
                       min_duration: float = MIN_HORIZON,
                       max_duration: float = MAX_HORIZON,
                       base_points: int = 500) -> np.ndarray:
    """Derive a simulation horizon that adapts to the system dynamics."""

    horizon = DEFAULT_HORIZON

    try:
        poles = control.pole(sys)
    except Exception:
        poles = []

    stable_real = [abs(np.real(p))
                   for p in poles
                   if np.real(p) < -SLOW_POLE_EPS]

    if stable_real:
        slowest = min(stable_real)
        if slowest > 0:
            horizon = max(horizon, 7.0 / slowest)

    try:
        info = control.step_info(sys)
    except Exception:
        info = {}

    settling = info.get("SettlingTime") if info else None
    if settling is not None and np.isfinite(settling) and settling > 0:
        horizon = max(horizon, 1.2 * settling)

    horizon = float(np.clip(horizon, min_duration, max_duration))
    num_points = max(base_points, int(np.ceil(horizon * 50)))
    return np.linspace(0.0, horizon, num_points)


def simulate_tf(num, den, T=None):
    """Simulate step response of transfer function defined by ``num`` and ``den``."""
    sys = control.TransferFunction(num, den)
    if T is None:
        T = time_axis_for_step(sys)
    t, y = control.step_response(sys, T=T)
    return sys, t, y
