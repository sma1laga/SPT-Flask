from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

import control
import numpy as np
import sympy as sp
from flask import Blueprint, jsonify, render_template, request

loop_shaping_bp = Blueprint("loop_shaping", __name__, template_folder="../templates")


EXAMPLES = [
    {
        "id": "pt1",
        "name": "PT1 (well damped)",
        "expression": "1/(s+1)",
        "controller": {"type": "P", "params": {"k": 1.2}},
    },
    {
        "id": "int_pt1",
        "name": "Integrator + PT1",
        "expression": "1/(s*(s+1))",
        "controller": {"type": "PI", "params": {"k": 1.0, "ti": 1.5}},
    },
    {
        "id": "nmp",
        "name": "Non-minimum-phase (RHP zero)",
        "expression": "(1 - s)/(s+1)",
        "controller": {"type": "Lead", "params": {"k": 0.6, "wz": 0.5, "wp": 5.0}},
    },
    {
        "id": "delay",
        "name": "Time delay (Padé approximation)",
        "expression": "(1-0.5*s)/(1+0.5*s) * 1/(s+1)",
        "controller": {"type": "PI", "params": {"k": 0.8, "ti": 2.0}},
    },
    {
        "id": "tight",
        "name": "Near instability (multiple poles)",
        "expression": "10/(s*(s+2)*(s+5))",
        "controller": {"type": "Lead", "params": {"k": 0.5, "wz": 1.0, "wp": 8.0}},
    },
    {
        "id": "unstable",
        "name": "Unstable process",
        "expression": "1/(s-1)",
        "controller": {"type": "PID", "params": {"k": 1.2, "ti": 0.8, "td": 0.15, "n": 12}},
    },
]


@dataclass
class ParsedPlant:
    transfer: Optional[control.TransferFunction]
    frequencies: Optional[np.ndarray]
    response: Optional[np.ndarray]
    warnings: List[str]


def _parse_number(token: str) -> complex:
    token = token.replace("i", "j")
    try:
        return complex(token)
    except ValueError as exc:
        raise ValueError(f"Invalid number: '{token}'") from exc


def _parse_coeff_list(text: str) -> List[complex]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Coefficient list is empty.")

    if stripped.startswith("["):
        try:
            coeffs = literal_eval(stripped)
        except Exception as exc:
            raise ValueError("Coefficient list is invalid.") from exc
        if not isinstance(coeffs, list) or not coeffs:
            raise ValueError("Please provide a non-empty list.")
        parsed = [_parse_number(str(v)) for v in coeffs]
    else:
        parts = [p.strip() for p in stripped.split(",") if p.strip()]
        if not parts:
            raise ValueError("Please separate coefficients with commas.")
        parsed = [_parse_number(p) for p in parts]

    cleaned = _clean_coeffs(parsed)
    if all(abs(c) < 1e-12 for c in cleaned):
        raise ValueError("All coefficients are zero.")
    return cleaned


def _clean_coeffs(coeffs: List[complex]) -> List[float]:
    cleaned: List[float] = []
    for value in coeffs:
        if abs(value.imag) < 1e-9:
            cleaned.append(float(value.real))
        else:
            raise ValueError("Complex coefficients are not supported yet.")
    return cleaned


def _parse_transfer_expression(expr: str) -> control.TransferFunction:
    s = sp.symbols("s", complex=True)
    expr_fixed = expr.replace("j", "I")
    expr_fixed = expr_fixed.replace("^", "**")
    expr_fixed = sp.sympify(expr_fixed, locals={"s": s})
    num_expr, den_expr = sp.fraction(sp.simplify(expr_fixed))
    num_poly = sp.Poly(sp.expand(num_expr), s)
    den_poly = sp.Poly(sp.expand(den_expr), s)
    if den_poly.is_zero:
        raise ValueError("Denominator is zero.")
    num_coeffs = _clean_coeffs([complex(c.evalf()) for c in num_poly.all_coeffs()])
    den_coeffs = _clean_coeffs([complex(c.evalf()) for c in den_poly.all_coeffs()])
    return control.TransferFunction(num_coeffs, den_coeffs)


def _parse_transfer_function(expression: str, numerator: str, denominator: str) -> control.TransferFunction:
    if expression.strip():
        return _parse_transfer_expression(expression)
    if numerator.strip() or denominator.strip():
        if not numerator.strip() or not denominator.strip():
            raise ValueError("Please provide both numerator and denominator.")
        num_coeffs = _parse_coeff_list(numerator)
        den_coeffs = _parse_coeff_list(denominator)
        return control.TransferFunction(num_coeffs, den_coeffs)
    raise ValueError("Please provide a transfer function.")


def _parse_frequency_data(text: str, data_format: str) -> Tuple[np.ndarray, np.ndarray]:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError("Measured data is empty.")

    frequencies: List[float] = []
    responses: List[complex] = []

    for line in lines:
        parts = [p for p in re.split(r"[\s,;]+", line) if p]
        if data_format == "real_imag":
            if len(parts) < 3:
                raise ValueError("Measured data must include ω, Re, Im.")
            w = float(parts[0])
            real = float(parts[1])
            imag = float(parts[2])
            resp = real + 1j * imag
        else:
            if len(parts) < 3:
                raise ValueError("Measured data must include ω, magnitude (dB), phase (°).")
            w = float(parts[0])
            mag_db = float(parts[1])
            phase_deg = float(parts[2])
            mag = 10 ** (mag_db / 20)
            resp = mag * np.exp(1j * np.deg2rad(phase_deg))

        if w <= 0:
            raise ValueError("Frequencies must be > 0.")
        frequencies.append(w)
        responses.append(resp)

    freq_array = np.array(frequencies, dtype=float)
    response_array = np.array(responses, dtype=complex)
    order = np.argsort(freq_array)
    return freq_array[order], response_array[order]


def _infer_frequency_grid(sys: control.TransferFunction, points: int = 480) -> np.ndarray:
    poles = control.poles(sys)
    zeros = control.zeros(sys)
    values = np.concatenate([poles, zeros]) if zeros.size or poles.size else np.array([])
    magnitudes = np.abs(values[np.isfinite(values)])
    magnitudes = magnitudes[magnitudes > 0]

    if magnitudes.size == 0:
        w_min, w_max = 1e-2, 1e2
    else:
        w_min = max(np.min(magnitudes) / 10.0, 1e-3)
        w_max = max(np.max(magnitudes) * 10.0, w_min * 10)
    return np.logspace(np.log10(w_min), np.log10(w_max), points)


def _controller_transfer(controller_type: str, params: Dict[str, float]) -> control.TransferFunction:
    s = control.TransferFunction([1, 0], [1])
    k = float(params.get("k", 1.0))

    if controller_type == "P":
        return control.TransferFunction([k], [1])
    if controller_type == "PI":
        ti = float(params.get("ti", 1.0))
        if ti <= 0:
            raise ValueError("Ti must be > 0.")
        return k * (ti * s + 1) / (ti * s)
    if controller_type == "PD":
        td = float(params.get("td", 0.1))
        n = float(params.get("n", 10.0))
        if td <= 0 or n <= 0:
            raise ValueError("Td and N must be > 0.")
        return k * (td * s + 1) / (td * s / n + 1)
    if controller_type == "PID":
        ti = float(params.get("ti", 1.0))
        td = float(params.get("td", 0.1))
        n = float(params.get("n", 10.0))
        if ti <= 0 or td <= 0 or n <= 0:
            raise ValueError("Ti, Td, and N must be > 0.")
        pi = (ti * s + 1) / (ti * s)
        pd = (td * s) / (td * s / n + 1)
        return k * (1 + (pi - 1) + pd)
    if controller_type in {"Lead", "Lag"}:
        wz = float(params.get("wz", 1.0))
        wp = float(params.get("wp", 10.0))
        if wz <= 0 or wp <= 0:
            raise ValueError("wz and wp must be > 0.")
        return k * (s / wz + 1) / (s / wp + 1)
    if controller_type == "Lead-Lag":
        lead_wz = float(params.get("lead_wz", 1.0))
        lead_wp = float(params.get("lead_wp", 10.0))
        lag_wz = float(params.get("lag_wz", 0.1))
        lag_wp = float(params.get("lag_wp", 1.0))
        if min(lead_wz, lead_wp, lag_wz, lag_wp) <= 0:
            raise ValueError("Frequencies must be > 0.")
        lead = (s / lead_wz + 1) / (s / lead_wp + 1)
        lag = (s / lag_wz + 1) / (s / lag_wp + 1)
        return k * lead * lag
    if controller_type == "Notch":
        w0 = float(params.get("w0", 10.0))
        zeta_z = float(params.get("zeta_z", 0.05))
        zeta_p = float(params.get("zeta_p", 0.5))
        if w0 <= 0 or zeta_z <= 0 or zeta_p <= 0:
            raise ValueError("w0, zeta_z, and zeta_p must be > 0.")
        numerator = [1, 2 * zeta_z * w0, w0 ** 2]
        denominator = [1, 2 * zeta_p * w0, w0 ** 2]
        return k * control.TransferFunction(numerator, denominator)
    raise ValueError("Unknown controller type.")


def _frequency_response(sys: control.TransferFunction, w: np.ndarray) -> np.ndarray:
    response = control.frequency_response(sys, w)
    if hasattr(response, "complex"):
        data = np.asarray(response.complex)
    elif hasattr(response, "response"):
        data = np.asarray(response.response)
    else:
        data = np.asarray(response.fresp)
    if data.ndim > 1:
        data = np.squeeze(data)
    if data.ndim > 1:
        data = data.reshape(-1)
    return np.atleast_1d(data)


def _compute_margins_from_data(w: np.ndarray, response: np.ndarray) -> Dict[str, Optional[float]]:
    mag = np.abs(response)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    phase = np.unwrap(np.angle(response)) * 180 / np.pi

    def interpolate_crossing(x_vals: np.ndarray, y_vals: np.ndarray, target: float) -> Optional[float]:
        diff = y_vals - target
        for idx in range(len(diff) - 1):
            if diff[idx] == 0:
                return x_vals[idx]
            if diff[idx] * diff[idx + 1] < 0:
                x0, x1 = x_vals[idx], x_vals[idx + 1]
                y0, y1 = diff[idx], diff[idx + 1]
                return x0 + (x1 - x0) * (-y0) / (y1 - y0)
        return None

    wc = interpolate_crossing(w, mag_db, 0.0)
    pm = None
    if wc is not None:
        phase_wc = np.interp(wc, w, phase)
        pm = 180 + phase_wc

    gm = None
    gm_db = None
    phase_targets = []
    if phase.size:
        min_phase, max_phase = phase.min(), phase.max()
        for k in range(-5, 6):
            target = -180 + 360 * k
            if min_phase <= target <= max_phase:
                phase_targets.append(target)

    wgc = None
    for target in phase_targets:
        candidate = interpolate_crossing(w, phase, target)
        if candidate is not None:
            wgc = candidate
            break

    if wgc is not None:
        mag_at_wgc = np.interp(wgc, w, mag)
        if mag_at_wgc > 0:
            gm = 1 / mag_at_wgc
            gm_db = 20 * np.log10(gm)

    return {
        "pm": pm,
        "gm": gm,
        "gm_db": gm_db,
        "wc": wc,
        "wgc": wgc,
    }


def _estimate_time_vector(sys: control.TransferFunction, points: int = 400) -> np.ndarray:
    poles = control.poles(sys)
    stable_poles = poles[np.real(poles) < 0]
    if stable_poles.size:
        tau = -1 / np.min(np.real(stable_poles))
        t_end = max(6 * tau, 1.0)
    else:
        t_end = 10.0
    return np.linspace(0, t_end, points)


def _build_interpretation(
    controller_type: str,
    has_integrator: bool,
    pm: Optional[float],
    gm_db: Optional[float],
    wc: Optional[float],
    bandwidth: Optional[float],
    closed_loop_stable: Optional[bool],
    warnings: List[str],
) -> Dict[str, Any]:
    notes: List[str] = []
    badges: Dict[str, Dict[str, str]] = {}

    if pm is not None and np.isfinite(pm):
        if pm < 30:
            notes.append("PM < 30° → risky, overshoot is likely.")
        elif pm < 50:
            notes.append("PM between 30° and 50° → moderate, check damping.")
        else:
            notes.append("PM ≥ 50° → good damping and robustness.")
    else:
        notes.append("Phase margin could not be determined reliably.")

    if gm_db is not None and np.isfinite(gm_db):
        if gm_db < 6:
            notes.append("Gain margin < 6 dB → low robustness.")
        else:
            notes.append("Gain margin ≥ 6 dB → solid robustness.")
    else:
        notes.append("Gain margin could not be determined reliably.")

    if has_integrator:
        notes.append("Integrator boosts low-frequency gain → step error ≈ 0, but phase drops.")
    else:
        notes.append("Without an integrator, step steady-state error remains nonzero.")

    if wc is not None and np.isfinite(wc):
        notes.append(f"Crossover ωc ≈ {wc:.3g} rad/s → sets bandwidth/speed.")
    if bandwidth is not None and np.isfinite(bandwidth):
        notes.append(f"Estimated bandwidth ≈ {bandwidth:.3g} rad/s.")

    if closed_loop_stable is None:
        badges["stability"] = {"label": "Stability: —", "class": "warn"}
    elif closed_loop_stable:
        badges["stability"] = {"label": "Stability: OK", "class": "ok"}
    else:
        badges["stability"] = {"label": "Stability: critical", "class": "danger"}

    robustness_class = "warn"
    robustness_label = "Robustness: review"
    if pm is not None and gm_db is not None:
        if pm >= 45 and gm_db >= 6:
            robustness_class = "ok"
            robustness_label = "Robustness: good"
        elif pm < 30 or gm_db < 3:
            robustness_class = "danger"
            robustness_label = "Robustness: low"
    badges["robustness"] = {"label": robustness_label, "class": robustness_class}

    speed_class = "warn"
    speed_label = "Speed: moderate"
    if bandwidth is not None and np.isfinite(bandwidth):
        if bandwidth > 5:
            speed_class = "ok"
            speed_label = "Speed: high"
        elif bandwidth < 0.5:
            speed_class = "danger"
            speed_label = "Speed: low"
    badges["speed"] = {"label": speed_label, "class": speed_class}

    steady_state_label = "Steady-state error: small with integrator"
    if not has_integrator:
        steady_state_label = "Steady-state error: likely > 0"
    badges["steady_state"] = {"label": steady_state_label, "class": "warn" if not has_integrator else "ok"}

    if warnings:
        notes.extend(warnings)

    return {"notes": notes, "badges": badges, "controller": controller_type}


def _compute_step_metrics(time: Optional[np.ndarray], response: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    if time is None or response is None or len(time) == 0:
        return {"mp": None, "t_r": None, "e_inf": None}

    y_inf = float(response[-1])
    if not np.isfinite(y_inf):
        return {"mp": None, "t_r": None, "e_inf": None}

    y_max = float(np.max(response))
    if y_inf != 0:
        mp = (y_max - y_inf) / abs(y_inf) * 100
    else:
        mp = None

    e_inf = 1 - y_inf

    t_r = None
    lower = 0.1 * y_inf
    upper = 0.9 * y_inf
    if y_inf != 0:
        try:
            t_low = time[np.where(response >= lower)[0][0]]
            t_high = time[np.where(response >= upper)[0][0]]
            t_r = float(t_high - t_low)
        except IndexError:
            t_r = None

    return {"mp": float(mp) if mp is not None else None, "t_r": t_r, "e_inf": float(e_inf)}


def _normalize_targets(design_mode: str, targets: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    design_mode = design_mode or "general"
    targets_display: Dict[str, Any] = {
        "mode": design_mode,
        "wD": None,
        "phi_req": None,
        "wc": None,
        "pm": None,
        "gm": None,
        "einf_required": False,
        "mp": None,
        "pm_suggestion": None,
        "pm_mapping_label": None,
    }
    auto_targets = dict(targets)

    if design_mode == "exam":
        t_an = float(targets.get("t_an") or 0)
        mp = float(targets.get("mp") or 0)
        einf_required = bool(targets.get("einf_required"))
        w_d = 1.5 / t_an if t_an > 0 else None
        phi_req = 70 - mp if mp > 0 else None
        targets_display.update(
            {
                "wD": w_d,
                "phi_req": phi_req,
                "wc": w_d,
                "pm": phi_req,
                "einf_required": einf_required,
                "mp": mp if mp > 0 else None,
            }
        )
        if w_d is not None:
            auto_targets["wc"] = w_d
        if phi_req is not None:
            auto_targets["pm_min"] = phi_req
        if einf_required:
            auto_targets["steady_state"] = "step_zero"
    else:
        wc = float(targets.get("wc") or 0) or None
        pm = float(targets.get("pm_min") or 0) or None
        gm = float(targets.get("gm_min") or 0) or None
        einf_required = bool(targets.get("einf_required"))
        mp = float(targets.get("mp") or 0) or None
        mapping = targets.get("mp_mapping", "heuristic_70_minus")
        pm_suggestion = None
        mapping_label = None
        if mp is not None:
            if mapping == "heuristic_70_minus":
                pm_suggestion = 70 - mp
                mapping_label = "Lehrstuhl heuristic (70° - Mp)"
            else:
                pm_suggestion = 100 - 0.5 * mp
                mapping_label = "Rule of thumb (100° - 0.5·Mp)"
        targets_display.update(
            {
                "wc": wc,
                "pm": pm,
                "gm": gm,
                "einf_required": einf_required,
                "mp": mp,
                "pm_suggestion": pm_suggestion,
                "pm_mapping_label": mapping_label,
            }
        )
        if einf_required:
            auto_targets["steady_state"] = "step_zero"

    return targets_display, auto_targets


def _auto_tune_controller(
    plant: control.TransferFunction,
    targets: Dict[str, Any],
) -> Tuple[str, Dict[str, float], List[str]]:
    warnings: List[str] = []
    tr = float(targets.get("tr") or 0)
    ts = float(targets.get("ts") or 0)
    pm_min = float(targets.get("pm_min") or 45)
    steady_state = targets.get("steady_state", "none")

    wc_target_override = float(targets.get("wc") or 0)
    wc_candidates = []
    if tr > 0:
        wc_candidates.append(1.8 / tr)
    if ts > 0:
        wc_candidates.append(4 / ts)

    if wc_target_override > 0:
        wc_target = wc_target_override
    elif wc_candidates:
        wc_target = max(wc_candidates)
    else:
        w_grid = _infer_frequency_grid(plant)
        response = _frequency_response(plant, w_grid)
        mag_db = 20 * np.log10(np.maximum(np.abs(response), 1e-12))
        idx = np.argmin(np.abs(mag_db))
        wc_target = w_grid[idx]

    pm_target = max(pm_min + 10, 35)

    rhp_zeros = [z for z in control.zeros(plant) if np.real(z) > 0]
    if rhp_zeros:
        warnings.append("Non-minimum-phase: high bandwidth/PM are limited.")

    if steady_state == "ramp_zero":
        controller_type = "PID"
    elif steady_state == "step_zero":
        controller_type = "PI"
    elif pm_target >= 60:
        controller_type = "Lead"
    else:
        controller_type = "P"

    params: Dict[str, float] = {"k": 1.0}

    if controller_type == "P":
        response = _frequency_response(plant, np.array([wc_target]))[0]
        mag = abs(response)
        params["k"] = 1 / mag if mag > 0 else 1.0
    elif controller_type == "PI":
        ti = 4 / wc_target
        response = _frequency_response(plant, np.array([wc_target]))[0]
        mag = abs(response)
        mag_pi = np.sqrt(1 + (1 / (wc_target * ti)) ** 2)
        params.update({"ti": ti, "k": 1 / (mag * mag_pi)})
    elif controller_type == "Lead":
        response = _frequency_response(plant, np.array([wc_target]))[0]
        phase = np.angle(response, deg=True)
        needed = pm_target - (180 + phase)
        needed = np.clip(needed, 5, 60)
        phi = np.deg2rad(needed)
        alpha = (1 - np.sin(phi)) / (1 + np.sin(phi))
        wz = wc_target * np.sqrt(alpha)
        wp = wc_target / np.sqrt(alpha)
        mag = abs(response)
        params.update({"wz": wz, "wp": wp, "k": np.sqrt(alpha) / mag})
    elif controller_type == "PID":
        ti = 4 / wc_target
        td = 1 / (4 * wc_target)
        params.update({"ti": ti, "td": td, "n": 10.0})
        response = _frequency_response(plant, np.array([wc_target]))[0]
        mag = abs(response)
        mag_pi = np.sqrt(1 + (1 / (wc_target * ti)) ** 2)
        params["k"] = 1 / (mag * mag_pi) if mag > 0 else 1.0

    return controller_type, params, warnings


@loop_shaping_bp.route("/", methods=["GET"])
@loop_shaping_bp.route("", methods=["GET"])
def loop_shaping():
    return render_template("loop_shaping.html", examples=EXAMPLES)


@loop_shaping_bp.route("/api", methods=["POST"])
def loop_shaping_api():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "manual")
    design_mode = data.get("design_mode", "general")
    warnings: List[str] = []

    try:
        plant_cfg = data.get("plant", {})
        expression = plant_cfg.get("expression", "")
        numerator = plant_cfg.get("numerator", "")
        denominator = plant_cfg.get("denominator", "")
        freq_data = plant_cfg.get("frequency_data", "")
        data_format = plant_cfg.get("data_format", "mag_phase")

        parsed = ParsedPlant(transfer=None, frequencies=None, response=None, warnings=[])
        if freq_data.strip():
            freqs, resp = _parse_frequency_data(freq_data, data_format)
            parsed.frequencies = freqs
            parsed.response = resp
            warnings.append("Measured data active: time responses are disabled.")
        else:
            parsed.transfer = _parse_transfer_function(expression, numerator, denominator)

        controller_type = data.get("controller", {}).get("type", "P")
        controller_params = data.get("controller", {}).get("params", {})
        target_inputs = data.get("targets", {})
        targets_display, auto_targets = _normalize_targets(design_mode, target_inputs)

        if mode == "auto":
            if parsed.transfer is None:
                return ("Auto-tuning requires a transfer function.", 400)
            controller_type, controller_params, auto_warnings = _auto_tune_controller(
                parsed.transfer, auto_targets
            )
            warnings.extend(auto_warnings)

        controller = _controller_transfer(controller_type, controller_params)

        if parsed.transfer is None:
            w = parsed.frequencies
            plant_response = parsed.response
        else:
            w = _infer_frequency_grid(parsed.transfer)
            plant_response = _frequency_response(parsed.transfer, w)

        controller_response = _frequency_response(controller, w)
        loop_response = controller_response * plant_response

        mag = np.abs(loop_response)
        mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
        phase = np.unwrap(np.angle(loop_response)) * 180 / np.pi

        if parsed.transfer is None:
            margins = _compute_margins_from_data(w, loop_response)
            closed_loop_stable = None
            bandwidth = None
            time = None
            step_response = None
            control_response = None
        else:
            loop_tf = controller * parsed.transfer
            margins_values = control.margin(loop_tf)
            gm, pm, wg, wp = margins_values
            gm_db = 20 * np.log10(gm) if np.isfinite(gm) and gm > 0 else None
            margins = {
                "pm": float(pm) if np.isfinite(pm) else None,
                "gm": float(gm) if np.isfinite(gm) else None,
                "gm_db": float(gm_db) if gm_db is not None and np.isfinite(gm_db) else None,
                "wc": float(wp) if np.isfinite(wp) else None,
                "wgc": float(wg) if np.isfinite(wg) else None,
            }

            closed_loop = control.feedback(loop_tf, 1)
            closed_loop_stable = bool(np.all(np.real(control.poles(closed_loop)) < 0))

            try:
                bandwidth = float(control.bandwidth(closed_loop))
            except Exception:
                bandwidth = None

            t_vec = _estimate_time_vector(closed_loop)
            time, step_response = control.step_response(closed_loop, T=t_vec)
            control_tf = control.feedback(controller, parsed.transfer)
            _, control_response = control.step_response(control_tf, T=t_vec)

        sensitivity = 1 / (1 + loop_response)
        complementary = loop_response / (1 + loop_response)

        if mode == "auto":
            pm_min = float(auto_targets.get("pm_min") or 0)
            gm_min = float(auto_targets.get("gm_min") or 0)
            if pm_min and margins.get("pm") is not None and margins["pm"] < pm_min:
                warnings.append("Auto-tuning: PM target not met → reduce bandwidth or use lead.")
            if gm_min and margins.get("gm_db") is not None and margins["gm_db"] < gm_min:
                warnings.append("Auto-tuning: GM target not met → reduce gain.")

        has_integrator = controller_type in {"PI", "PID"}
        step_metrics = _compute_step_metrics(time, step_response)

        interpretation = _build_interpretation(
            controller_type=controller_type,
            has_integrator=has_integrator,
            pm=margins.get("pm"),
            gm_db=margins.get("gm_db"),
            wc=margins.get("wc"),
            bandwidth=bandwidth,
            closed_loop_stable=closed_loop_stable,
            warnings=warnings,
        )

        measured = {
            "wc": margins.get("wc"),
            "pm": margins.get("pm"),
            "gm": margins.get("gm_db"),
            "mp": step_metrics.get("mp"),
            "t_r": step_metrics.get("t_r"),
            "e_inf": step_metrics.get("e_inf"),
        }
        target_checks = {}
        if targets_display.get("wc") is not None and measured["wc"] is not None:
            target_checks["wc"] = abs(measured["wc"] - targets_display["wc"]) / max(
                targets_display["wc"], 1e-9
            ) <= 0.1
        if targets_display.get("pm") is not None and measured["pm"] is not None:
            target_checks["pm"] = measured["pm"] >= targets_display["pm"]
        if targets_display.get("gm") is not None and measured["gm"] is not None:
            target_checks["gm"] = measured["gm"] >= targets_display["gm"]
        if targets_display.get("mp") is not None and measured["mp"] is not None:
            target_checks["mp"] = measured["mp"] <= targets_display["mp"]
        if targets_display.get("einf_required") and measured["e_inf"] is not None:
            target_checks["e_inf"] = abs(measured["e_inf"]) <= 0.02

        return jsonify(
            {
                "frequency": w.tolist(),
                "mag_db": mag_db.tolist(),
                "phase_deg": phase.tolist(),
                "nyquist_re": loop_response.real.tolist(),
                "nyquist_im": loop_response.imag.tolist(),
                "nichols_phase": phase.tolist(),
                "nichols_mag_db": mag_db.tolist(),
                "sensitivity_mag_db": (20 * np.log10(np.maximum(np.abs(sensitivity), 1e-12))).tolist(),
                "complementary_mag_db": (
                    20 * np.log10(np.maximum(np.abs(complementary), 1e-12))
                ).tolist(),
                "time": time.tolist() if time is not None else None,
                "step_response": step_response.tolist() if step_response is not None else None,
                "control_response": control_response.tolist() if control_response is not None else None,
                "margins": margins,
                "controller": {"type": controller_type, "params": controller_params},
                "bandwidth": float(bandwidth) if bandwidth is not None and np.isfinite(bandwidth) else None,
                "interpretation": interpretation,
                "targets": targets_display,
                "measured": measured,
                "target_checks": target_checks,
                "warnings": warnings,
            }
        )
    except Exception as exc:
        return (str(exc), 400)