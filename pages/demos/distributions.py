"""Interactive distribution explorer demo"""

from math import factorial
from typing import Dict, Tuple

import numpy as np
from flask import Blueprint, jsonify, render_template, request
from scipy.special import erf, gamma as gamma_fn, gammainc


demos_distributions_bp = Blueprint(
    "demos_distributions", __name__, template_folder="../../templates"
)

S = 0.01

DEFAULTS: Dict[str, float] = {
    "distribution": "normal",
    "mean": 0.0,
    "sigma": 1.0,
    "laplace_sigma": 1.0,
    "rayleigh_sigma": 1.0,
    "lambda": 1.0,
    "cauchy_a": 0.0,
    "cauchy_b": 1.0,
    "binom_p": 0.5,
    "binom_n": 60,
    "geom_p": 0.5,
    "poisson_a": 20.0,
    "gamma_a": 2.0,
    "gamma_lambda": 0.2,
    "erlang_n": 2,
    "erlang_lambda": 0.2,
    "chi_b": 2.0,
    "uniform_max": 5.0,
    "uniform_min": -5.0,
    "fixed_limits": True,
}


FORMULAS: Dict[str, Tuple[str, str]] = {
    "normal": (
        r"f_X(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
        r"F_X(x) = \tfrac{1}{2}\left(1 + \operatorname{erf}\left(\tfrac{x-\mu}{\sqrt{2}\sigma}\right)\right)",
    ),
    "laplace": (
        r"f_X(x) = \frac{1}{\sqrt{2}\sigma} e^{-\frac{\sqrt{2}|x-\mu|}{\sigma}}",
        r"F_X(x) = \tfrac{1}{2}\bigl(1 + \operatorname{sgn}(x-\mu)(1-e^{-\frac{|x-\mu|}{\sigma/\sqrt{2}}})\bigr)",
    ),
    "rayleigh": (
        r"f_X(x) = \frac{x}{\sigma^2} e^{-\frac{x^2}{2\sigma^2}}\; \epsilon(x)",
        r"F_X(x) = 1 - e^{-\frac{x^2}{2\sigma^2}}\; \epsilon(x)",
    ),
    "exponential": (
        r"f_X(x) = \lambda e^{-\lambda x}\; \epsilon(x)",
        r"F_X(x) = (1-e^{-\lambda x})\; \epsilon(x)",
    ),
    "cauchy": (
        r"f_X(x) = \frac{b}{\pi(b^2 + (x-a)^2)}",
        r"F_X(x) = \tfrac{1}{2} + \tfrac{1}{\pi} \tan^{-1}\left(\tfrac{x-a}{b}\right)",
    ),
    "binomial": (
        r"f_X(k) = \binom{N}{k} p^k (1-p)^{N-k}",
        r"F_X(k) = \sum_{i=0}^k \binom{N}{i} p^i (1-p)^{N-i}",
    ),
    "geometric": (
        r"f_X(k) = p (1-p)^k,\; k \ge 0",
        r"F_X(k) = 1 - (1-p)^{k+1}",
    ),
    "poisson": (
        r"f_X(k) = e^{-\lambda} \frac{\lambda^k}{k!}",
        r"F_X(k) = e^{-\lambda} \sum_{i=0}^k \frac{\lambda^i}{i!}",
    ),
    "gamma": (
        r"f_X(x) = \frac{\lambda^a x^{a-1} e^{-\lambda x}}{\Gamma(a)}\; \epsilon(x)",
        r"F_X(x) = 1 - \frac{\Gamma(a, \lambda x)}{\Gamma(a)}",
    ),
    "erlang": (
        r"f_X(x) = \frac{\lambda^n x^{n-1} e^{-\lambda x}}{(n-1)!}\; \epsilon(x)",
        r"F_X(x) = 1 - \frac{\Gamma(n, \lambda x)}{\Gamma(n)}",
    ),
    "chi_square": (
        r"f_X(x) = \frac{x^{b/2 - 1} e^{-x/2}}{2^{b/2} \Gamma(b/2)}\; \epsilon(x)",
        r"F_X(x) = \frac{\gamma\left(\tfrac{b}{2}, \tfrac{x}{2}\right)}{\Gamma(\tfrac{b}{2})}",
    ),
    "uniform": (
        r"f_X(x) = \frac{1}{x_{\max} - x_{\min}}\; \text{for}\; x \in [x_{\min}, x_{\max}] ",
        r"F_X(x) = \frac{x - x_{\min}}{x_{\max} - x_{\min}}\; \text{on support}",
    ),
}


@demos_distributions_bp.route("/", methods=["GET"])
def page():
    return render_template("demos/distributions.html", defaults=DEFAULTS)


def _step_cdf(k: np.ndarray, pdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cdf = np.cumsum(pdf)
    x_cdf = np.concatenate(([k[0] - 1], k))
    y_cdf = np.concatenate(([0.0], cdf))
    return x_cdf, np.clip(y_cdf, 0.0, 1.0)


def _normal(mean: float, sigma: float, fixed_limits: bool):
    sigma = max(sigma, 1e-4)
    lim = 10.0 if fixed_limits else max(10.0 * sigma, 10.0)
    x = np.arange(-lim, lim + S, S)
    pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mean) ** 2) / (2 * sigma**2))
    cdf = 0.5 * (1 + erf((x - mean) / (np.sqrt(2) * sigma)))
    variance = sigma**2
    return x, pdf, cdf, mean, variance


def _laplace(mean: float, sigma: float, fixed_limits: bool):
    sigma = max(sigma, 1e-4)
    lim = 10.0 if fixed_limits else max(20.0 * sigma, 10.0)
    x = np.arange(-lim, lim + S, S)
    pdf = (1 / (np.sqrt(2) * sigma)) * np.exp(-np.sqrt(2) * np.abs(x - mean) / sigma)
    cdf = 0.5 * (1 + np.sign(x - mean) * (1 - np.exp(-np.abs(x - mean) / (sigma / np.sqrt(2)))))
    variance = sigma**2
    return x, pdf, cdf, mean, variance


def _rayleigh(sigma: float, fixed_limits: bool):
    sigma = max(sigma, 1e-4)
    lim = 10.0 if fixed_limits else max(5.0 * sigma, 10.0)
    x = np.arange(0.0, lim + S, S)
    pdf = (x / (sigma**2)) * np.exp(-(x**2) / (2 * sigma**2))
    cdf = 1 - np.exp(-(x**2) / (2 * sigma**2))
    mean = sigma * np.sqrt(np.pi / 2)
    variance = (2 - np.pi / 2) * sigma**2
    return x, pdf, cdf, mean, variance


def _exponential(lmbda: float, fixed_limits: bool):
    lmbda = max(lmbda, 1e-6)
    lim = 10.0 if fixed_limits else max(20.0 / lmbda, 10.0)
    x = np.arange(0.0, lim + S, S)
    pdf = lmbda * np.exp(-lmbda * x)
    cdf = 1 - np.exp(-lmbda * x)
    mean = 1.0 / lmbda
    variance = 1.0 / (lmbda**2)
    return x, pdf, cdf, mean, variance


def _cauchy(a: float, b: float, fixed_limits: bool):
    b = max(b, 1e-6)
    lim = 10.0 if fixed_limits else max(40.0 * b, 10.0)
    x = np.arange(-lim, lim + S, S)
    pdf = b / (np.pi * (b**2 + (x - a) ** 2))
    cdf = 0.5 + (1 / np.pi) * np.arctan((x - a) / b)
    return x, pdf, cdf, None, None


def _binomial(p: float, n: int):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    n = max(1, min(int(round(n)), 100))
    ks = np.arange(0, n + 1)
    pdf = np.zeros_like(ks, dtype=float)
    pdf[0] = (1 - p) ** n
    for k in range(1, n + 1):
        pdf[k] = p * (n - k + 1) * pdf[k - 1] / ((1 - p) * k)
    x_cdf, cdf = _step_cdf(ks, pdf)
    mean = n * p
    variance = n * p * (1 - p)
    return ks, pdf, x_cdf, cdf, mean, variance


def _geometric(p: float):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ks = np.arange(0, 70)
    pdf = p * (1 - p) ** ks
    x_cdf, cdf = _step_cdf(ks, pdf)
    mean = (1 - p) / p
    variance = (1 - p) / (p**2)
    return ks, pdf, x_cdf, cdf, mean, variance


def _poisson(a: float):
    a = max(a, 1e-6)
    max_k = min(120, max(80, int(np.ceil(6 * a))))
    ks = np.arange(0, max_k + 1)
    pdf = np.exp(-a) * (a**ks) / np.array([factorial(int(k)) for k in ks], dtype=float)
    x_cdf, cdf = _step_cdf(ks, pdf)
    return ks, pdf, x_cdf, cdf, a, a


def _gamma(a: float, lmbda: float):
    a = max(a, 1e-6)
    lmbda = max(lmbda, 1e-6)
    x = np.arange(0.0, 60.0 + S, S)
    pdf = (lmbda**a) * (x ** (a - 1)) * np.exp(-lmbda * x) / gamma_fn(a)
    cdf = gammainc(a, lmbda * x)
    mean = a / lmbda
    variance = a / (lmbda**2)
    return x, pdf, cdf, mean, variance


def _erlang(n: int, lmbda: float):
    n = max(1, min(int(round(n)), 6))
    lmbda = max(lmbda, 1e-6)
    x = np.arange(0.0, 60.0 + S, S)
    pdf = (lmbda**n) * (x ** (n - 1)) * np.exp(-lmbda * x) / factorial(n - 1)
    cdf = gammainc(n, lmbda * x)
    mean = n / lmbda
    variance = n / (lmbda**2)
    return x, pdf, cdf, mean, variance


def _chi_square(b: float):
    b = max(b, 1e-6)
    x = np.arange(0.0, 60.0 + S, S)
    pdf = (x ** (b / 2 - 1)) * np.exp(-x / 2) / (2 ** (b / 2) * gamma_fn(b / 2))
    cdf = gammainc(b / 2, x / 2)
    mean = b
    variance = 2 * b
    return x, pdf, cdf, mean, variance


def _uniform(x_min: float, x_max: float, fixed_limits: bool):
    if x_min >= x_max:
        raise ValueError("Uniform distribution requires X_min < X_max.")
    lim = 10.0 if fixed_limits else max(abs(x_min), abs(x_max), 10.0) * 1.2
    x = np.arange(-lim, lim + S, S)
    support = (x >= x_min) & (x <= x_max)
    pdf = np.where(support, 1.0 / (x_max - x_min), 0.0)
    raw_cdf = (x - x_min) / (x_max - x_min)
    cdf = np.where(x < x_min, 0.0, np.where(x > x_max, 1.0, raw_cdf))
    mean = (x_min + x_max) / 2
    variance = (x_max - x_min) ** 2 / 12
    return x, pdf, cdf, mean, variance


def _response_payload(
    dist_key: str,
    kind: str,
    x_pdf: np.ndarray,
    pdf: np.ndarray,
    x_cdf: np.ndarray,
    cdf: np.ndarray,
    mean: float,
    variance: float,
    support: str,
):
    pdf_formula, cdf_formula = FORMULAS[dist_key]
    return {
        "kind": kind,
        "x_pdf": x_pdf.tolist(),
        "pdf": pdf.tolist(),
        "x_cdf": x_cdf.tolist(),
        "cdf": cdf.tolist(),
        "mean": None if mean is None else float(mean),
        "variance": None if variance is None else float(variance),
        "support": support,
        "pdf_formula": pdf_formula,
        "cdf_formula": cdf_formula,
        "title": dist_key,
    }


def _prepare_distribution(dist: str, params: Dict):
    fixed_limits = bool(params.get("fixed_limits", DEFAULTS["fixed_limits"]))

    if dist == "normal":
        mean = float(params.get("mean", DEFAULTS["mean"]))
        sigma = float(params.get("sigma", DEFAULTS["sigma"]))
        x, pdf, cdf, mean, variance = _normal(mean, sigma, fixed_limits)
        return _response_payload(
            dist, "continuous", x, pdf, x, cdf, mean, variance, r"(-\infty, \infty)"
        )

    if dist == "laplace":
        mean = float(params.get("mean", DEFAULTS["mean"]))
        sigma = float(params.get("laplace_sigma", DEFAULTS["laplace_sigma"]))
        x, pdf, cdf, mean, variance = _laplace(mean, sigma, fixed_limits)
        return _response_payload(
            dist, "continuous", x, pdf, x, cdf, mean, variance, r"(-\infty, \infty)"
        )

    if dist == "rayleigh":
        sigma = float(params.get("rayleigh_sigma", DEFAULTS["rayleigh_sigma"]))
        x, pdf, cdf, mean, variance = _rayleigh(sigma, fixed_limits)
        return _response_payload(
            dist, "continuous", x, pdf, x, cdf, mean, variance, r"[0, \infty)"
        )

    if dist == "exponential":
        lmbda = float(params.get("lambda", DEFAULTS["lambda"]))
        x, pdf, cdf, mean, variance = _exponential(lmbda, fixed_limits)
        return _response_payload(
            dist, "continuous", x, pdf, x, cdf, mean, variance, r"[0, \infty)"
        )

    if dist == "cauchy":
        a = float(params.get("cauchy_a", DEFAULTS["cauchy_a"]))
        b = float(params.get("cauchy_b", DEFAULTS["cauchy_b"]))
        x, pdf, cdf, mean, variance = _cauchy(a, b, fixed_limits)
        return _response_payload(
            dist, "continuous", x, pdf, x, cdf, mean, variance, "(-\infty, \infty)"
        )

    if dist == "binomial":
        p = float(params.get("binom_p", DEFAULTS["binom_p"]))
        n = params.get("binom_n", DEFAULTS["binom_n"])
        ks, pdf, x_cdf, cdf, mean, variance = _binomial(p, n)
        return _response_payload(
            dist,
            "discrete",
            ks,
            pdf,
            x_cdf,
            cdf,
            mean,
            variance,
            rf"k ∈ {{0, \ldots, {int(max(ks))}}}",
        )

    if dist == "geometric":
        p = float(params.get("geom_p", DEFAULTS["geom_p"]))
        ks, pdf, x_cdf, cdf, mean, variance = _geometric(p)
        return _response_payload(
            dist,
            "discrete",
            ks,
            pdf,
            x_cdf,
            cdf,
            mean,
            variance,
            r"k \ge 0",
        )

    if dist == "poisson":
        a = float(params.get("poisson_a", DEFAULTS["poisson_a"]))
        ks, pdf, x_cdf, cdf, mean, variance = _poisson(a)
        return _response_payload(
            dist,
            "discrete",
            ks,
            pdf,
            x_cdf,
            cdf,
            mean,
            variance,
            r"k \ge 0",
        )

    if dist == "gamma":
        a = float(params.get("gamma_a", DEFAULTS["gamma_a"]))
        lmbda = float(params.get("gamma_lambda", DEFAULTS["gamma_lambda"]))
        x, pdf, cdf, mean, variance = _gamma(a, lmbda)
        return _response_payload(
            dist, "continuous", x, pdf, x, cdf, mean, variance, r"[0, \infty)"
        )

    if dist == "erlang":
        n = params.get("erlang_n", DEFAULTS["erlang_n"])
        lmbda = float(params.get("erlang_lambda", DEFAULTS["erlang_lambda"]))
        x, pdf, cdf, mean, variance = _erlang(n, lmbda)
        return _response_payload(
            dist, "continuous", x, pdf, x, cdf, mean, variance, r"[0, \infty)"
        )

    if dist == "chi_square":
        b = float(params.get("chi_b", DEFAULTS["chi_b"]))
        x, pdf, cdf, mean, variance = _chi_square(b)
        return _response_payload(
            dist, "continuous", x, pdf, x, cdf, mean, variance, r"[0, \infty)"
        )

    if dist == "uniform":
        x_max = float(params.get("uniform_max", DEFAULTS["uniform_max"]))
        x_min = float(params.get("uniform_min", DEFAULTS["uniform_min"]))
        x, pdf, cdf, mean, variance = _uniform(x_min, x_max, fixed_limits)
        return _response_payload(
            dist,
            "continuous",
            x,
            pdf,
            x,
            cdf,
            mean,
            variance,
            f"[{x_min:.2f}, {x_max:.2f}]",
        )

    raise ValueError("Unknown distribution selected.")


@demos_distributions_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        distribution = (data.get("distribution") or DEFAULTS["distribution"]).strip().lower()
        payload = _prepare_distribution(distribution, data)
        return jsonify(payload)
    except Exception as exc:  
        return jsonify({"error": str(exc)}), 400