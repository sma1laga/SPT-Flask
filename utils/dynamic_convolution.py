from flask import jsonify
from functools import partial
import numpy as np
from scipy.signal import convolve
from utils.math_utils import rect, tri, step, cos, sin, delta, inv_t, si, delta_train
from utils.eval_helpers import safe_eval


def rect_half_edge(t, half_width=0.5, tol=1e-12):
    """
    rect(t): 1 for |t|<half_width, 0 for |t|>half_width, and 0.5 at |t|=half_width.
    This avoids single-sample spikesat exact boundaries
    """
    t = np.asarray(t)
    a = np.abs(t)
    out = np.zeros_like(a, dtype=float)
    out[a < half_width] = 1.0
    out[np.isclose(a, half_width, atol=tol)] = 0.5
    return out

def _delta_impulse(t_shifted: np.ndarray, *, dt: float, **_kwargs) -> np.ndarray:
    """Sampled ideal impulse with unit area: amplitude 1/dt at closest sample to 0."""
    t_shifted = np.asarray(t_shifted)
    out = np.zeros_like(t_shifted, dtype=float)
    idx = int(np.argmin(np.abs(t_shifted)))
    out[idx] = 1.0 / dt
    return out


def _delta_train_impulse(t: np.ndarray, *, dt: float, spacing: float = 1.0, count: int = 17, **_kwargs) -> np.ndarray:
    """Finite impulse train on sampled grid with unit area per impulse"""
    t = np.asarray(t)
    out = np.zeros_like(t, dtype=float)
    count = int(max(1, count))
    offsets = (np.arange(count) - (count - 1) // 2) * float(spacing)
    for off in offsets:
        idx = int(np.argmin(np.abs(t - off)))
        out[idx] += 1.0 / dt
    return out


def _coerce_1d(y, t, name: str):
    """Ensure y is a float 1D array same shape as t"""
    y = np.asarray(y)
    if y.ndim == 0:
        y = np.full_like(t, float(y), dtype=float)
    else:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != t.size:
            raise ValueError(f"{name} has wrong length: got {y.size}, expected {t.size}")
        y = y.reshape(t.shape)
    return y


functions = [
    ("rect(t)", "rect(t)"),
    ("tri(t)", "tri(t)"),
    ("step(t)", "step(t)"),
    ("sin(πt)·step(t)", "sin(t)*step(t)"),
    ("cos(πt)·step(t)", "cos(t)*step(t)"),
    ("delta(t)", "delta(t)"),
    ("delta(t-2)", "delta(t-2)"),
    ("delta_train(t)·step(t)", "delta_train(t)*step(t+0.1)"),  # shift +0.1 to avoid half delta at t=0
    ("exp(t)·step(-t)", "exp(t)*step(-t)"),
    ("exp(-t)·step(t)", "exp(-t)*step(t)"),
    ("inv_t(t)", "inv_t(t)"),
    ("si(πt)", "si(pi*t)"),
    ("si^2(πt)", "si(pi*t)**2"),
]


def conv_json(data: dict):
    """
    Expects JSON { func1, func2 }.
    Returns JSON { t, y1, y2, y_conv }.
    """
    try:
        f1_str = (data.get("func1", "") or "").strip()
        f2_str = (data.get("func2", "") or "").strip()

        DISP_RANGE = 3.5
        SLIDER_RANGE = 3.5

        # calc grid
        DT = 0.0025  # 1/400 -> aligns perfectly with spacing=1.0 and half-width=0.5
        t_calc = np.arange(-4*DISP_RANGE-0.1, 4*DISP_RANGE+0.1 + DT/2, DT)
        dt = DT

        # context for CALC: ideal sampled impulses (fixes delta-train notches)
        ctx_calc = {
            "t": t_calc, "pi": np.pi, "e": np.e,
            "rect": rect_half_edge, "tri": tri, "step": step,
            "cos": partial(cos, t_norm=np.pi), "sin": partial(sin, t_norm=np.pi),
            "delta": partial(_delta_impulse, dt=dt),
            "delta_train": partial(_delta_train_impulse, dt=dt),
            "inv_t": inv_t, "si": si, "exp": np.exp
        }

        y1_calc = safe_eval(f1_str, ctx_calc) if f1_str else np.zeros_like(t_calc)
        y2_calc = safe_eval(f2_str, ctx_calc) if f2_str else np.zeros_like(t_calc)
        y1_calc = _coerce_1d(y1_calc, t_calc, "y1_calc")
        y2_calc = _coerce_1d(y2_calc, t_calc, "y2_calc")

        y_conv_calc = convolve(y1_calc, y2_calc, mode="same") * dt

        # display grid
        t_disp = np.arange(-DISP_RANGE-SLIDER_RANGE-0.1, DISP_RANGE+SLIDER_RANGE+0.1 + DT/2, DT)

        # context for DISPLAY: gaussian deltas (look nicer)
        ctx_disp = ctx_calc.copy()
        ctx_disp["t"] = t_disp
        ctx_disp["delta"] = delta
        ctx_disp["delta_train"] = delta_train

        y1_disp = safe_eval(f1_str, ctx_disp) if f1_str else np.zeros_like(t_disp)
        y2_disp = safe_eval(f2_str, ctx_disp) if f2_str else np.zeros_like(t_disp)
        y1_disp = _coerce_1d(y1_disp, t_disp, "y1_disp")
        y2_disp = _coerce_1d(y2_disp, t_disp, "y2_disp")

        def _scale_delta(expr, arr):
            expr = (expr or "").strip()
            if "delta" in expr:
                m = float(np.max(np.abs(arr)))
                if m != 0:
                    return arr / m
            return arr

        y1_disp = _scale_delta(f1_str, y1_disp)
        y2_disp = _scale_delta(f2_str, y2_disp)

        y_conv_disp = np.interp(t_disp, t_calc, y_conv_calc)

        # if both are deltas, keep plot readable
        if ("delta" in f1_str) and ("delta" in f2_str):
            y_conv_disp = _scale_delta("delta", y_conv_disp)

        return jsonify({
            "t": t_disp.tolist(),
            "y1": y1_disp.tolist(),
            "y2": y2_disp.tolist(),
            "y_conv": y_conv_disp.tolist()
        })

    except Exception as e:
        # IMPORTANT: always return JSON so frontend never breaks on r.json()
        return jsonify(error=f"Internal server error in conv_json: {type(e).__name__}: {e}"), 500