"""Central Limit Theorem demo iterative convolution toward Gaußian"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from flask import Blueprint, jsonify, render_template, request


demos_central_limit_theorem_bp = Blueprint(
    "demos_central_limit_theorem", __name__, template_folder="../../templates"
)


@dataclass
class StepResult:
    n: int
    x: np.ndarray
    pdf: np.ndarray
    mean: float
    variance: float


DEFAULTS = {
    "distribution": "laplace",
    "num_vars": 5,
    "laplace": {"variance": 1.0, "mean": 0.0},
    "uniform": {"x_max": 3.0, "x_min": -3.0},
    "rayleigh": {"sigma": 1.0},
    "exponential": {"lambda": 1.0},
    "gamma": {"shape": 1.4, "rate": 0.4},
}


@demos_central_limit_theorem_bp.route("/", methods=["GET"], endpoint="page")
def page():
    return render_template(
        "demos/central_limit_theorem.html",
        defaults=DEFAULTS,
    )


def _normalize_pdf(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    pdf = np.clip(pdf, 0.0, None)
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(pdf, x))
    else:
        area = float(np.trapz(pdf, x))
    if area <= 0:
        raise ValueError("PDF integrates to zero; check parameters.")
    return pdf / area


def _stats(x: np.ndarray, pdf: np.ndarray) -> Tuple[float, float]:
    dx = float(x[1] - x[0])
    mean = float(np.sum(x * pdf) * dx)
    variance = float(np.sum(((x - mean) ** 2) * pdf) * dx)
    return mean, variance


def _normal_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    if variance <= 0:
        return np.zeros_like(x)
    return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(
        -((x - mean) ** 2) / (2 * variance)
    )


def _laplace_grid(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    variance = float(params.get("variance", DEFAULTS["laplace"]["variance"]))
    mean = float(params.get("mean", DEFAULTS["laplace"]["mean"]))
    if variance <= 0:
        raise ValueError("Laplace variance must be positive.")
    b = np.sqrt(variance / 2.0)
    span = max(8.0 * b, 10.0)
    x = np.linspace(mean - span, mean + span, 1201)
    pdf = (1.0 / (2 * b)) * np.exp(-np.abs(x - mean) / b)
    return x, pdf


def _uniform_grid(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    x_max = float(params.get("x_max", DEFAULTS["uniform"]["x_max"]))
    x_min = float(params.get("x_min", DEFAULTS["uniform"]["x_min"]))
    if not x_min < x_max:
        raise ValueError("Uniform distribution needs X_min < X_max.")
    padding = (x_max - x_min) * 0.5 + 4.0
    x = np.linspace(x_min - padding, x_max + padding, 1201)
    pdf = np.where((x >= x_min) & (x <= x_max), 1.0 / (x_max - x_min), 0.0)
    return x, pdf


def _rayleigh_grid(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    sigma = float(params.get("sigma", DEFAULTS["rayleigh"]["sigma"]))
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")
    x = np.linspace(0.0, 10.0 * sigma, 1201)
    pdf = (x / (sigma**2)) * np.exp(-(x**2) / (2 * sigma**2))
    return x, pdf


def _exponential_grid(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    lmbda = float(params.get("lambda", DEFAULTS["exponential"]["lambda"]))
    if lmbda <= 0:
        raise ValueError("Lambda must be positive.")
    x = np.linspace(0.0, 12.0 / lmbda, 1201)
    pdf = lmbda * np.exp(-lmbda * x)
    return x, pdf


def _gamma_grid(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    shape = float(params.get("shape", DEFAULTS["gamma"]["shape"]))
    rate = float(params.get("rate", DEFAULTS["gamma"]["rate"]))
    if shape <= 0 or rate <= 0:
        raise ValueError("Gamma parameters must be positive.")
    theta = 1.0 / rate
    span = max(8.0 * shape * theta, 12.0)
    x = np.linspace(0.0, span, 1401)
    pdf = (x ** (shape - 1) * np.exp(-x / theta)) / (math.gamma(shape) * (theta**shape))
    return x, pdf


def _base_grid(distribution: str, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    if distribution == "laplace":
        x, pdf = _laplace_grid(params)
    elif distribution == "uniform":
        x, pdf = _uniform_grid(params)
    elif distribution == "rayleigh":
        x, pdf = _rayleigh_grid(params)
    elif distribution == "exponential":
        x, pdf = _exponential_grid(params)
    elif distribution == "gamma":
        x, pdf = _gamma_grid(params)
    else:
        raise ValueError("Unsupported distribution.")

    pdf = _normalize_pdf(x, pdf)
    return x, pdf


def _convolution_sequence(distribution: str, params: Dict[str, float], num_vars: int) -> List[StepResult]:
    if num_vars < 1:
        raise ValueError("Number of variables must be at least 1.")

    base_x, base_pdf = _base_grid(distribution, params)
    dx = float(base_x[1] - base_x[0])

    steps: List[StepResult] = []
    current_x = base_x
    current_pdf = base_pdf

    for n in range(1, num_vars + 1):
        mean, variance = _stats(current_x, current_pdf)
        steps.append(StepResult(n=n, x=current_x, pdf=current_pdf, mean=mean, variance=variance))
        if n == num_vars:
            break
        conv = np.convolve(current_pdf, base_pdf) * dx
        start = current_x[0] + base_x[0]
        end = current_x[-1] + base_x[-1]
        current_x = np.linspace(start, end, len(conv))
        current_pdf = _normalize_pdf(current_x, conv)

    return steps


def _serialize_step(step: StepResult):
    normal = _normal_pdf(step.x, step.mean, step.variance)
    return {
        "n": step.n,
        "x": step.x.tolist(),
        "pdf": step.pdf.tolist(),
        "mean": step.mean,
        "variance": step.variance,
        "normal_pdf": normal.tolist(),
    }


@demos_central_limit_theorem_bp.route("/compute", methods=["POST"], endpoint="compute")
def compute():
    try:
        payload = request.get_json(force=True) or {}
        distribution = payload.get("distribution", DEFAULTS["distribution"]).strip().lower()
        num_vars = int(payload.get("num_vars", DEFAULTS["num_vars"]))
        params = payload.get(distribution, {}) if isinstance(payload, dict) else {}

        steps = _convolution_sequence(distribution, params, num_vars)
        return jsonify(
            {
                "distribution": distribution,
                "steps": [_serialize_step(step) for step in steps],
            }
        )
    except Exception as exc: 
        return jsonify({"error": str(exc)}), 400