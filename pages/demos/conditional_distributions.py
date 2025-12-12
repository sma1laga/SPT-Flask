"""Conditional distribution demo: uniform normal exponential"""
from flask import Blueprint, render_template, request, jsonify
import numpy as np
from scipy.special import erf


demos_conditional_distributions_bp = Blueprint(
    "demos_conditional_distributions", __name__, template_folder="../../templates"
)

DEFAULTS = {
    "distribution": "uniform",
    "x_max": 5.0,
    "x_min": -5.0,
    "a_max": 2.0,
    "a_min": -2.0,
    "mean": 0.0,
    "variance": 4.0,
    "lambda": 0.3,
}


@demos_conditional_distributions_bp.route("/", methods=["GET"])
def page():
    return render_template("demos/conditional_distributions.html", defaults=DEFAULTS)


def _conditional_curves(x: np.ndarray, pdf: np.ndarray, cdf: np.ndarray, a_min: float, a_max: float):
    cdf_at_min = float(np.interp(a_min, x, cdf))
    cdf_at_max = float(np.interp(a_max, x, cdf))
    cond_prob = cdf_at_max - cdf_at_min
    if cond_prob <= 0:
        raise ValueError("Condition probability is zero; choose A_min < A_max with non-zero mass.")

    cond_pdf = np.where((x >= a_min) & (x <= a_max), pdf / cond_prob, 0.0)
    base_cdf = (cdf - cdf_at_min) / cond_prob
    cond_cdf = np.where(x < a_min, 0.0, np.where(x > a_max, 1.0, base_cdf))
    return cond_prob, cond_pdf, cond_cdf


def _uniform_curves(x_min: float, x_max: float):
    if not x_min < x_max:
        raise ValueError("Uniform distribution requires X_min < X_max.")
    x = np.linspace(x_min - 2.0, x_max + 2.0, 1200)
    pdf = np.where((x >= x_min) & (x <= x_max), 1.0 / (x_max - x_min), 0.0)
    cdf = np.where(x < x_min, 0.0, np.where(x <= x_max, (x - x_min) / (x_max - x_min), 1.0))
    return x, pdf, cdf


def _normal_curves(mean: float, variance: float):
    if variance <= 0:
        raise ValueError("Variance must be positive.")
    std = float(np.sqrt(variance))
    spread = max(4.0 * std, 8.0)
    x = np.linspace(mean - spread, mean + spread, 1500)
    pdf = (1.0 / (np.sqrt(2 * np.pi * variance))) * np.exp(-((x - mean) ** 2) / (2 * variance))
    cdf = 0.5 * (1 + erf((x - mean) / (np.sqrt(2 * variance))))
    return x, pdf, cdf


def _exponential_curves(lmbda: float):
    if lmbda <= 0:
        raise ValueError("Lambda must be positive.")
    x = np.linspace(0.0, 20.0, 1200)
    pdf = lmbda * np.exp(-lmbda * x)
    cdf = 1.0 - np.exp(-lmbda * x)
    return x, pdf, cdf


def _plot_curves(x, pdf, cdf, cond_pdf, cond_cdf, a_min, a_max, title_prefix: str):
    return {
        "x": x.tolist(),
        "pdf": pdf.tolist(),
        "cdf": cdf.tolist(),
        "cond_pdf": cond_pdf.tolist(),
        "cond_cdf": cond_cdf.tolist(),
        "a_min": a_min,
        "a_max": a_max,
        "title": title_prefix,
    }


def _prepare_curves(distribution: str, params: dict):
    if distribution == "uniform":
        x_min = float(params.get("x_min", DEFAULTS["x_min"]))
        x_max = float(params.get("x_max", DEFAULTS["x_max"]))
        title = "Uniform"
        x, pdf, cdf = _uniform_curves(x_min, x_max)
    elif distribution == "normal":
        mean = float(params.get("mean", DEFAULTS["mean"]))
        variance = float(params.get("variance", DEFAULTS["variance"]))
        title = "Normal"
        x, pdf, cdf = _normal_curves(mean, variance)
    elif distribution == "exponential":
        lmbda = float(params.get("lambda", DEFAULTS["lambda"]))
        title = "Exponential"
        x, pdf, cdf = _exponential_curves(lmbda)
    else:
        raise ValueError("Unknown distribution selected.")

    a_min = float(params.get("a_min", DEFAULTS["a_min"]))
    a_max = float(params.get("a_max", DEFAULTS["a_max"]))
    if a_min >= a_max:
        raise ValueError("A_min must be smaller than A_max.")

    cond_prob, cond_pdf, cond_cdf = _conditional_curves(x, pdf, cdf, a_min, a_max)
    return x, pdf, cdf, cond_pdf, cond_cdf, a_min, a_max, title, cond_prob


@demos_conditional_distributions_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        distribution = (data.get("distribution") or DEFAULTS["distribution"]).strip().lower()

        x, pdf, cdf, cond_pdf, cond_cdf, a_min, a_max, title, cond_prob = _prepare_curves(distribution, data)

        payload = _plot_curves(x, pdf, cdf, cond_pdf, cond_cdf, a_min, a_max, title_prefix=title)
        payload["condition_prob"] = float(cond_prob)
        return jsonify(payload)
    except Exception as exc:  
        return jsonify({"error": str(exc)}), 400