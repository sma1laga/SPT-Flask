"""Interactive 2D normal distribution explorer."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from flask import Blueprint, jsonify, render_template, request


demos_normal2d_bp = Blueprint(
    "demos_normal2d", __name__, template_folder="../../templates"
)

DEFAULTS: Dict[str, float] = {
    "rho": 0.3,
    "mean_x": 0.0,
    "mean_y": 0.0,
    "sigma_x": 2.0,
    "sigma_y": 1.4,
}


_DEF_STEP = 120
_MEAN_MIN = -5.0
_MEAN_MAX = 5.0
_SIGMA_MAX = 4.0
_AXIS_PADDING_SIGMA = 4.0
_AXIS_MIN = _MEAN_MIN - _AXIS_PADDING_SIGMA * _SIGMA_MAX
_AXIS_MAX = _MEAN_MAX + _AXIS_PADDING_SIGMA * _SIGMA_MAX


def _sanitize(payload: Dict[str, float]) -> Dict[str, float]:
    rho = float(payload.get("rho", DEFAULTS["rho"]))
    rho = max(min(rho, 0.98), -0.98)
    sigma_x = max(float(payload.get("sigma_x", DEFAULTS["sigma_x"])), 0.1)
    sigma_y = max(float(payload.get("sigma_y", DEFAULTS["sigma_y"])), 0.1)

    return {
        "rho": rho,
        "mean_x": float(payload.get("mean_x", DEFAULTS["mean_x"])),
        "mean_y": float(payload.get("mean_y", DEFAULTS["mean_y"])),
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
    }


def _grid_limits(mean_x: float, mean_y: float, sigma_x: float, sigma_y: float):
    return (
        np.linspace(_AXIS_MIN, _AXIS_MAX, _DEF_STEP),
        np.linspace(_AXIS_MIN, _AXIS_MAX, _DEF_STEP),
    )


def _gaussian_2d(x: np.ndarray, y: np.ndarray, params: Dict[str, float]):
    rho = params["rho"]
    mx, my = params["mean_x"], params["mean_y"]
    sx, sy = params["sigma_x"], params["sigma_y"]

    x0 = x - mx
    y0 = y - my
    inv = 1.0 / (2 * (1 - rho**2))
    pref = 1.0 / (2 * math.pi * sx * sy * math.sqrt(1 - rho**2))
    exponent = -inv * (
        (x0**2) / (sx**2)
        - 2 * rho * (x0 * y0) / (sx * sy)
        + (y0**2) / (sy**2)
    )
    return pref * np.exp(exponent)


def _marginal(axis: np.ndarray, mean: float, sigma: float):
    return (1.0 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(
        -((axis - mean) ** 2) / (2 * sigma**2)
    )


def _build_payload(params: Dict[str, float]):
    x_axis, y_axis = _grid_limits(
        params["mean_x"], params["mean_y"], params["sigma_x"], params["sigma_y"]
    )
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    z_grid = _gaussian_2d(x_grid, y_grid, params)

    pdf_x = _marginal(x_axis, params["mean_x"], params["sigma_x"])
    pdf_y = _marginal(y_axis, params["mean_y"], params["sigma_y"])

    return {
        "x_grid": x_grid.tolist(),
        "y_grid": y_grid.tolist(),
        "z_grid": z_grid.tolist(),
        "x_range": [_AXIS_MIN, _AXIS_MAX],
        "y_range": [_AXIS_MIN, _AXIS_MAX],
        "marginal_x": {
            "x": x_axis.tolist(),
            "y": np.full_like(x_axis, y_axis.max()).tolist(),
            "z": pdf_x.tolist(),
        },
        "marginal_y": {
            "x": np.full_like(y_axis, x_axis.max()).tolist(),
            "y": y_axis.tolist(),
            "z": pdf_y.tolist(),
        },
        "z_peak": float(z_grid.max()),
    }


@demos_normal2d_bp.route("/", methods=["GET"])
def page():
    payload = _build_payload(DEFAULTS)
    return render_template("demos/normal2d.html", defaults=DEFAULTS, initial=payload)


@demos_normal2d_bp.route("/compute", methods=["POST"])
def compute():
    params = _sanitize(request.get_json(force=True) or {})
    payload = _build_payload(params)
    return jsonify({"data": payload, "params": params})