"""Interactive demo for mapping random variables through function"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from flask import Blueprint, jsonify, render_template, request


mapping_random_variables_bp = Blueprint(
    "mapping_random_variables", __name__, template_folder="../../templates"
)


@dataclass
class Distribution:
    name: str
    defaults: Dict[str, float]
    pdf: Callable[[Dict[str, float]], Callable[[np.ndarray], np.ndarray]]
    support: Callable[[Dict[str, float]], Tuple[float, float]]


def _uniform_pdf(x_min: float, x_max: float) -> Callable[[np.ndarray], np.ndarray]:
    if x_min >= x_max:
        raise ValueError("Uniform distribution requires X_min < X_max.")

    def _pdf(x: np.ndarray) -> np.ndarray:
        return np.where((x >= x_min) & (x <= x_max), 1.0 / (x_max - x_min), 0.0)

    return _pdf


def _normal_pdf(mean: float, variance: float) -> Callable[[np.ndarray], np.ndarray]:
    if variance <= 0:
        raise ValueError("Variance must be positive.")
    std = float(np.sqrt(variance))

    def _pdf(x: np.ndarray) -> np.ndarray:
        return (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * variance))

    return _pdf


def _exponential_pdf(lmbda: float) -> Callable[[np.ndarray], np.ndarray]:
    if lmbda <= 0:
        raise ValueError("Lambda must be positive.")

    def _pdf(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, lmbda * np.exp(-lmbda * x), 0.0)

    return _pdf


def _rayleigh_pdf(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    def _pdf(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, (x / sigma**2) * np.exp(-(x**2) / (2 * sigma**2)), 0.0)

    return _pdf


def _cauchy_pdf(loc: float, scale: float) -> Callable[[np.ndarray], np.ndarray]:
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    def _pdf(x: np.ndarray) -> np.ndarray:
        return (1.0 / (np.pi * scale)) * (scale**2 / ((x - loc) ** 2 + scale**2))

    return _pdf


def _support_all(_: Dict[str, float]) -> Tuple[float, float]:
    return (-10.0, 10.0)


def _support_positive(params: Dict[str, float]) -> Tuple[float, float]:
    limit = max(10.0, 8.0 / max(params.get("lambda", 0.1), params.get("sigma", 0.1)))
    return (0.0, limit)


DISTRIBUTIONS: Dict[str, Distribution] = {
    "uniform": Distribution(
        "Uniform",
        {"x_min": -2.0, "x_max": 2.0},
        lambda params: _uniform_pdf(params["x_min"], params["x_max"]),
        _support_all,
    ),
    "normal": Distribution(
        "Normal",
        {"mean": 0.0, "variance": 1.0},
        lambda params: _normal_pdf(params["mean"], params["variance"]),
        _support_all,
    ),
    "exponential": Distribution(
        "Exponential",
        {"lambda": 0.5},
        lambda params: _exponential_pdf(params["lambda"]),
        _support_positive,
    ),
    "rayleigh": Distribution(
        "Rayleigh",
        {"sigma": 1.0},
        lambda params: _rayleigh_pdf(params["sigma"]),
        _support_positive,
    ),
    "cauchy": Distribution(
        "Cauchy",
        {"loc": 0.0, "scale": 1.0},
        lambda params: _cauchy_pdf(params["loc"], params["scale"]),
        _support_all,
    ),
}


QUANTIZER_PRESETS = {
    "coarse": {
        "label": "2-level",
        "x_steps": [-10, 0, 0, 10],
        "y_steps": [-5, -5, 5, 5],
    },
    "medium": {
        "label": "4-level",
        "x_steps": [-10, -5, -5, 0, 0, 5, 5, 10],
        "y_steps": [-6, -6, -2, -2, 2, 2, 6, 6],
    },
    "fine": {
        "label": "6-level",
        "x_steps": [-10, -6.6, -6.6, -3.3, -3.3, 0, 0, 3.3, 3.3, 6.6, 6.6, 10],
        "y_steps": [-5, -5, -3, -3, -1, -1, 1, 1, 3, 3, 5, 5],
    },
}


DEFAULTS = {
    "distribution": "uniform",
    "mapping": "linear",
    "linear": {"a": 2.0, "b": 0.0},
    "quadratic": {"a": 1.0, "b": 0.0},
    "quantizer": {"preset": "coarse"},
    "custom": {"points": "-8:-6, -4:-2, 0:0, 4:2, 8:6"},
}
DEFAULTS.update({key: dist.defaults for key, dist in DISTRIBUTIONS.items()})


def _range_from_support(support: Tuple[float, float], margin: float = 2.0) -> np.ndarray:
    lo, hi = support
    span = hi - lo
    return np.linspace(lo - margin, hi + margin, 2000)


def _linear_mapping(a: float, b: float, x: np.ndarray, pdf_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a == 0:
        mass = np.trapz(pdf_x, x)
        return np.array([b]), np.array([mass])
    y = a * x + b
    pdf_y = pdf_x / abs(a)
    return y, pdf_y


def _quadratic_mapping(a: float, b: float, x: np.ndarray, pdf_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a <= 0:
        raise ValueError("Quadratic mapping requires a > 0 to stay invertible on |x|.")
    y = a * (x**2) + b
    y_sorted = np.sort(y)
    positive = y_sorted[y_sorted >= b]
    if positive.size == 0:
        return np.array([b]), np.array([0.0])

    # Inverse mapping for y >= b: x = ±sqrt((y-b)/a)
    y_grid = np.linspace(b, positive[-1], 1200)
    root = np.sqrt((y_grid - b) / a)
    pdf_pos = np.interp(root, x, pdf_x, left=0.0, right=0.0)
    pdf_neg = np.interp(-root, x, pdf_x, left=0.0, right=0.0)
    pdf_y = (pdf_pos + pdf_neg) / (2 * np.sqrt(a * (y_grid - b)))
    return y_grid, pdf_y


def _quantizer_mapping(
    x_steps: Iterable[float],
    y_steps: Iterable[float],
    x: np.ndarray,
    pdf_x: np.ndarray,
) -> Tuple[List[float], List[float]]:
    xs = list(x_steps)
    ys = list(y_steps)
    if len(xs) != len(ys):
        raise ValueError("Quantizer step definitions must have equal lengths.")

    y_positions: List[float] = []
    masses: List[float] = []
    for idx in range(len(xs) - 1):
        x_start, x_end = xs[idx], xs[idx + 1]
        y_val = ys[idx]
        mask = (x >= min(x_start, x_end)) & (x <= max(x_start, x_end))
        prob_mass = float(np.trapz(pdf_x[mask], x[mask]))
        if prob_mass > 0:
            y_positions.append(y_val)
            masses.append(prob_mass)
    return y_positions, masses


def _custom_points_mapping(points: str) -> Tuple[np.ndarray, np.ndarray]:
    pairs: List[Tuple[float, float]] = []
    for item in points.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError("Custom points must use the form x:y, separated by commas.")
        x_str, y_str = item.split(":", 1)
        pairs.append((float(x_str), float(y_str)))
    pairs.sort(key=lambda p: p[0])
    if len(pairs) < 2:
        raise ValueError("Provide at least two points for a custom mapping.")
    xs, ys = zip(*pairs)
    return np.array(xs), np.array(ys)


def _piecewise_mapping(
    xs: np.ndarray, ys: np.ndarray, x: np.ndarray, pdf_x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_grid = np.linspace(np.min(ys), np.max(ys), 1500)
    mapped_pdf = np.zeros_like(y_grid)
    for idx in range(len(xs) - 1):
        x0, x1 = xs[idx], xs[idx + 1]
        y0, y1 = ys[idx], ys[idx + 1]
        if x1 == x0:
            continue
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        if a == 0:
            mask_x = (x >= min(x0, x1)) & (x <= max(x0, x1))
            mass = float(np.trapz(pdf_x[mask_x], x[mask_x]))
            idx_y = np.argmin(np.abs(y_grid - y0))
            mapped_pdf[idx_y] += mass
            continue
        inv_x = (y_grid - b) / a
        mask_y = (y_grid >= min(y0, y1)) & (y_grid <= max(y0, y1))
        pdf_segment = np.interp(inv_x, x, pdf_x, left=0.0, right=0.0) / abs(a)
        mapped_pdf += np.where(mask_y, pdf_segment, 0.0)
    return y_grid, mapped_pdf


def _compute_mapping(payload: Dict) -> Dict:
    dist_key = (payload.get("distribution") or DEFAULTS["distribution"]).lower()
    mapping = (payload.get("mapping") or DEFAULTS["mapping"]).lower()
    if dist_key not in DISTRIBUTIONS:
        raise ValueError("Unknown distribution selected.")
    distribution = DISTRIBUTIONS[dist_key]

    params = {**distribution.defaults, **{k: float(v) for k, v in payload.items() if k in distribution.defaults}}
    support = distribution.support(params)
    x = _range_from_support(support)
    pdf_fn = distribution.pdf(params)
    pdf_x = pdf_fn(x)

    result: Dict[str, object] = {
        "x": x.tolist(),
        "pdf_x": pdf_x.tolist(),
        "mapping": mapping,
        "dist_label": distribution.name,
    }

    if mapping == "linear":
        a = float(payload.get("a", DEFAULTS["linear"]["a"]))
        b = float(payload.get("b", DEFAULTS["linear"]["b"]))
        y, pdf_y = _linear_mapping(a, b, x, pdf_x)
        result.update(
            {
                "mapping_points": [x.tolist(), (a * x + b).tolist()],
                "y": y.tolist(),
                "pdf_y": pdf_y.tolist(),
                "params": {"a": a, "b": b},
            }
        )
    elif mapping == "quadratic":
        a = float(payload.get("a", DEFAULTS["quadratic"]["a"]))
        b = float(payload.get("b", DEFAULTS["quadratic"]["b"]))
        y, pdf_y = _quadratic_mapping(a, b, x, pdf_x)
        result.update(
            {
                "mapping_points": [x.tolist(), (a * x**2 + b).tolist()],
                "y": y.tolist(),
                "pdf_y": pdf_y.tolist(),
                "params": {"a": a, "b": b},
            }
        )
    elif mapping == "quantizer":
        preset = payload.get("preset", DEFAULTS["quantizer"]["preset"])
        preset_cfg = QUANTIZER_PRESETS.get(preset)
        if not preset_cfg:
            raise ValueError("Unknown quantizer preset.")
        y_positions, masses = _quantizer_mapping(preset_cfg["x_steps"], preset_cfg["y_steps"], x, pdf_x)
        result.update(
            {
                "mapping_points": [preset_cfg["x_steps"], preset_cfg["y_steps"]],
                "y_positions": y_positions,
                "masses": masses,
                "preset": preset,
            }
        )
    elif mapping == "custom":
        points = str(payload.get("points") or DEFAULTS["custom"]["points"])
        xs, ys = _custom_points_mapping(points)
        y, pdf_y = _piecewise_mapping(xs, ys, x, pdf_x)
        result.update(
            {
                "mapping_points": [xs.tolist(), ys.tolist()],
                "y": y.tolist(),
                "pdf_y": pdf_y.tolist(),
                "points": points,
            }
        )
    else:
        raise ValueError("Unknown mapping mode selected.")

    return result


@mapping_random_variables_bp.route("/", methods=["GET"])
def page():
    return render_template(
        "demos/mapping_random_variables.html", defaults=DEFAULTS, quantizers=QUANTIZER_PRESETS
    )


@mapping_random_variables_bp.route("/compute", methods=["POST"])
def compute():
    try:
        payload = request.get_json(force=True) or {}
        result = _compute_mapping(payload)
        return jsonify(result)
    except Exception as exc:  
        return jsonify({"error": str(exc)}), 400