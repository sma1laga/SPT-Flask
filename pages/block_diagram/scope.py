import numpy as np
import control

MAX_POINTS = 4000


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


def simulate_tf(num, den, T):
    """Simulate step response of transfer function defined by ``num`` and ``den``."""
    sys = control.TransferFunction(num, den)
    t, y = control.step_response(sys, T=T)
    return sys, t, y