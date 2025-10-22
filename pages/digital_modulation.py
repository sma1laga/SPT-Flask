import math
import numpy as np
from flask import Blueprint, render_template, request, jsonify

dig_bp = Blueprint('digital_modulation', __name__, template_folder='templates')

# ---------- helpers ----------


def make_time(t_end=1.0, fs=1000):
    N = int(max(8, fs * t_end))
    return np.linspace(0.0, t_end, N, endpoint=False)


def spectrum_db(x: np.ndarray, fs: float):
    N = x.size
    win = np.hanning(N)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    P = (np.abs(X) ** 2) / (np.sum(win ** 2))
    P_db = 10 * np.log10(P + 1e-18)
    return f, P_db


def add_awgn(x: np.ndarray, snr_db):
    if snr_db is None or np.isinf(snr_db):
        return x
    p_sig = np.mean(x ** 2)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    p_noise = p_sig / snr_lin if snr_lin > 0 else 0.0
    n = np.random.normal(0.0, np.sqrt(p_noise), size=x.shape)
    return x + n


def _to_float_or_inf(s):
    if s is None:
        return None
    v = str(s).strip().lower()
    if v in ('inf', 'infinity'):
        return np.inf
    return float(s)

def _to_bool(s, default=True):
    if s is None:
        return default
    v = str(s).strip().lower()
    if v in ("0", "false", "off", "no"):
        return False
    if v in ("1", "true", "on", "yes"):
        return True
    return default


_erfc_vec = np.vectorize(math.erfc)


def qfunc(x):
    x_arr = np.asarray(x, dtype=float)
    return 0.5 * _erfc_vec(x_arr / np.sqrt(2.0))


def root_raised_cosine(beta: float, span: int, sps: int):
    beta = float(np.clip(beta, 0.0, 1.0))
    span = int(max(2, span))
    sps = int(max(2, sps))
    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps
    h = np.zeros_like(t)

    if beta == 0.0:
        h = np.sinc(t)
    else:
        for i, ti in enumerate(t):
            ti = float(ti)
            four_bt = 4 * beta * ti
            if abs(ti) < 1e-12:
                h[i] = 1.0 + beta * (4 / np.pi - 1)
            elif abs(abs(four_bt) - 1.0) < 1e-12:
                h[i] = (
                    beta
                    / np.sqrt(2)
                    * (
                        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                    )
                )
            else:
                numerator = (
                    np.sin(np.pi * ti * (1 - beta))
                    + four_bt * np.cos(np.pi * ti * (1 + beta))
                )
                denominator = np.pi * ti * (1 - four_bt**2)
                h[i] = numerator / denominator

    energy = np.sum(h**2)
    if energy <= 0:
        return h
    return h / np.sqrt(energy)


def pam_symbol_levels(M: int):
    M = int(M)
    levels = 2 * np.arange(M) - (M - 1)
    energy = np.mean(levels**2)
    norm = np.sqrt(energy)
    return levels / norm


def pam_theoretical_ber(M: int, ebn0_db):
    M = int(M)
    k = np.log2(M)
    if not float(k).is_integer() or M < 2:
        return np.nan
    k = int(k)
    ebn0_db = np.asarray(ebn0_db, dtype=float)
    ebn0_lin = 10 ** (ebn0_db / 10.0)
    const = 6 * k / (M**2 - 1)
    arg = np.sqrt(const * ebn0_lin)
    return 2 * (M - 1) / (M * k) * qfunc(arg)

# ----- passband helpers -----


def _digital_constellation(scheme: str, n_symbols: int, rng):
    scheme = scheme.upper()
    if scheme == 'BPSK':
        bits = rng.integers(0, 2, size=n_symbols)
        symbols = 2 * bits - 1
        return symbols.astype(complex), 1

    if scheme == 'QPSK':
        mapping = np.array(
            [
                1 + 1j,
                -1 + 1j,
                -1 - 1j,
                1 - 1j,
            ],
            dtype=complex,
        )
        mapping /= np.sqrt(2)
        idx = rng.integers(0, 4, size=n_symbols)
        return mapping[idx], 2

    if scheme in ('16QAM', '16-QAM', 'M-QAM'):
        bits = rng.integers(0, 2, size=(n_symbols, 4))
        gray_map = np.array([-3, -1, 3, 1], dtype=float)
        i_idx = (bits[:, 0] << 1) | bits[:, 1]
        q_idx = (bits[:, 2] << 1) | bits[:, 3]
        i_level = gray_map[i_idx]
        q_level = gray_map[q_idx]
        symbols = (i_level + 1j * q_level) / np.sqrt(10.0)
        return symbols, 4

    raise ValueError('Unsupported scheme: {}'.format(scheme))


def _costas_correct(symbols: np.ndarray, times: np.ndarray):
    if symbols.size < 2:
        return symbols.copy(), 0.0, 0.0

    phases = np.unwrap(np.angle(symbols))
    try:
        coeffs = np.polyfit(times, phases, 1)
    except np.linalg.LinAlgError:
        return symbols.copy(), 0.0, 0.0

    omega_est = float(coeffs[0])
    phase_est = float(coeffs[1])
    correction = np.exp(-1j * (omega_est * times + phase_est))
    corrected = symbols * correction
    return corrected, omega_est, phase_est


# ---------- simulation helpers ----------


def simulate_m_pam(
    *,
    M: int,
    sps: int,
    rolloff: float,
    span: int,
    snr_db: float,
    symbols: int,
    enable_tx_rrc: bool,
    enable_rx_rrc: bool,
    rng=None,
    include_waveforms: bool = True,
    include_eye: bool = True,
):
    if rng is None:
        rng = np.random.default_rng()

    M = int(max(2, M))
    if not float(np.log2(M)).is_integer():
        raise ValueError("M must be a power of 2 for M-PAM")
    bits_per_symbol = int(np.log2(M))
    symbols = int(max(16, symbols))
    sps = int(max(2, sps))
    span = int(max(2, span))
    rolloff = float(np.clip(rolloff, 0.0, 1.0))

    levels = pam_symbol_levels(M)
    sym_idx = rng.integers(0, M, size=symbols)
    sym_wave = levels[sym_idx]

    upsampled = np.zeros(symbols * sps)
    upsampled[::sps] = sym_wave

    tx_rrc = root_raised_cosine(rolloff, span, sps)
    if enable_tx_rrc:
        tx_wave = np.convolve(upsampled, tx_rrc, mode='full')
        tx_delay = (tx_rrc.size - 1) // 2
    else:
        tx_wave = upsampled.copy()
        tx_delay = 0

    if enable_rx_rrc:
        rx_rrc = tx_rrc
        rx_delay = (rx_rrc.size - 1) // 2
        rx_clean = np.convolve(tx_wave, rx_rrc, mode='full')
        noise_gain = float(np.sum(rx_rrc**2))
    else:
        rx_rrc = None
        rx_delay = 0
        rx_clean = tx_wave.copy()
        noise_gain = 1.0

    total_delay = tx_delay + rx_delay
    sample_idx = total_delay + np.arange(symbols) * sps
    sample_idx = sample_idx.astype(int)
    valid = sample_idx < rx_clean.size
    sample_idx = sample_idx[valid]
    signal_samples = rx_clean[sample_idx]
    sym_idx_valid = sym_idx[valid]
    sym_wave_valid = sym_wave[valid]

    snr_db = float(snr_db)
    ebn0_lin = 10 ** (snr_db / 10.0)
    if signal_samples.size == 0 or bits_per_symbol <= 0 or not np.isfinite(ebn0_lin):
        sigma = 0.0
    else:
        es = float(np.mean(signal_samples**2))
        if es <= 0.0:
            sigma = 0.0
        else:
            eb = es / bits_per_symbol
            if ebn0_lin <= 0.0:
                sigma = 0.0
            else:
                n0 = eb / ebn0_lin
                sigma = float(np.sqrt(max(n0 / 2.0, 0.0) / max(noise_gain, 1e-18)))

    noise = rng.normal(0.0, sigma, size=tx_wave.shape)
    rx_input = tx_wave + noise

    if enable_rx_rrc:
        rx_wave = np.convolve(rx_input, rx_rrc, mode='full')
    else:
        rx_wave = rx_input

    if sample_idx.size == 0:
        sym_idx = np.array([], dtype=int)
        sym_wave = np.array([], dtype=float)
        rx_symbols = np.array([], dtype=float)
    else:
        valid_rx = sample_idx < rx_wave.size
        sample_idx = sample_idx[valid_rx]
        sym_idx = sym_idx_valid[valid_rx]
        sym_wave = sym_wave_valid[valid_rx]
        rx_symbols = rx_wave[sample_idx]

    diffs = rx_symbols[:, None] - levels[None, :]
    decisions = np.argmin(diffs**2, axis=1)
    detected_levels = levels[decisions]

    tx_bits = np.unpackbits(sym_idx[:, None].astype(np.uint8), axis=1, bitorder='big')
    tx_bits = tx_bits[:, -bits_per_symbol:].reshape(-1)

    det_bits = np.unpackbits(decisions[:, None].astype(np.uint8), axis=1, bitorder='big')
    det_bits = det_bits[:, -bits_per_symbol:].reshape(-1)
    nbits = tx_bits.size
    bit_errors = int(np.sum(tx_bits != det_bits))
    ber = bit_errors / max(1, nbits)

    eye = None
    if include_eye:
        mf_wave = rx_wave
        eye_span = 2 * sps
        half = sps
        traces = []
        for center in sample_idx:
            if center - half < 0 or center + half > mf_wave.size:
                continue
            seg = mf_wave[center - half : center + half]
            if seg.size == eye_span:
                traces.append(seg)
            if len(traces) >= 40:
                break
        if traces:
            time_eye = (np.arange(eye_span) - half) / sps
            eye = {
                'time': time_eye,
                'traces': np.array(traces),
            }

    waveforms = None
    if include_waveforms:
        t_tx = np.arange(tx_wave.size) / sps
        t_rx = np.arange(rx_wave.size) / sps
        waveforms = {
            'tx_time': t_tx,
            'tx': tx_wave,
            'rx_time': t_rx,
            'rx': rx_wave,
        }

    return {
        'levels': levels,
        'tx_symbols': sym_wave,
        'rx_symbols': rx_symbols,
        'decisions': detected_levels,
        'sample_indices': sample_idx,
        'bit_errors': bit_errors,
        'ber': ber,
        'eye': eye,
        'waveforms': waveforms,
        'bits_per_symbol': bits_per_symbol,
        'symbols': int(sample_idx.size),
    }

@dig_bp.route('/api/passband')
def passband_api():
    params = {k: request.args.get(k) for k in request.args.keys()}

    scheme = (params.get('scheme') or 'BPSK').upper()
    try:
        symbol_rate = float(params.get('symbol_rate', 400.0))
        carrier = float(params.get('carrier', 2000.0))
        freq_offset = float(params.get('freq_offset', 0.0))
        phase_deg = float(params.get('phase_deg', 0.0))
        sps = int(params.get('sps', 16))
        n_symbols = int(params.get('symbols', 240))
    except ValueError as exc:
        return jsonify(error=f'Invalid parameter: {exc}'), 400

    sps = int(max(4, sps))
    n_symbols = int(max(16, n_symbols))
    symbol_rate = float(max(10.0, symbol_rate))
    carrier = float(max(1.0, carrier))

    seed_val = params.get('seed')
    if seed_val is not None:
        try:
            seed = int(seed_val)
        except ValueError:
            return jsonify(error='seed must be an integer'), 400
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(2024)

    try:
        tx_symbols, bits_per_symbol = _digital_constellation(scheme, n_symbols, rng)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    fs = symbol_rate * sps
    dt = 1.0 / fs
    t = np.arange(n_symbols * sps) * dt
    baseband = np.repeat(tx_symbols, sps)

    phase_rad = np.deg2rad(phase_deg)
    carrier_actual = carrier + freq_offset
    tx_passband = np.real(baseband * np.exp(1j * (2 * np.pi * carrier * t)))
    rx_passband = np.real(baseband * np.exp(1j * (2 * np.pi * carrier_actual * t + phase_rad)))

    # complex envelope after mixing with nominal carrier
    rx_complex = baseband * np.exp(1j * (2 * np.pi * freq_offset * t + phase_rad))

    center = sps // 2
    sample_idx = center + np.arange(n_symbols) * sps
    valid = sample_idx < rx_complex.size
    sample_idx = sample_idx[valid]
    symbol_times = sample_idx * dt
    rx_symbols = rx_complex[sample_idx]
    ideal_symbols = tx_symbols[: sample_idx.size]

    costas_on = _to_bool(params.get('costas', params.get('costas_on', '1')), default=True)
    corrected_symbols = np.array([], dtype=complex)
    corrected_wave = rx_complex.copy()
    omega_est = 0.0
    phase_est = 0.0
    if costas_on:
        corrected_symbols, omega_est, phase_est = _costas_correct(rx_symbols, symbol_times)
        correction_wave = np.exp(-1j * (omega_est * t + phase_est))
        corrected_wave = rx_complex * correction_wave
    else:
        corrected_wave = rx_complex

    info = {
        'scheme': scheme,
        'symbol_rate': symbol_rate,
        'carrier_hz': carrier,
        'freq_offset_hz': freq_offset,
        'phase_offset_deg': phase_deg,
        'bits_per_symbol': bits_per_symbol,
        'costas_enabled': bool(costas_on),
    }
    if costas_on:
        info['estimated_phase_deg'] = float(np.degrees(phase_est))
        info['estimated_freq_hz'] = float(omega_est / (2 * np.pi))

    constellation = {
        'ideal_i': ideal_symbols.real.tolist(),
        'ideal_q': ideal_symbols.imag.tolist(),
        'rx_i': rx_symbols.real.tolist(),
        'rx_q': rx_symbols.imag.tolist(),
        'corrected_i': corrected_symbols.real.tolist() if corrected_symbols.size else [],
        'corrected_q': corrected_symbols.imag.tolist() if corrected_symbols.size else [],
        'costas': bool(costas_on),
    }

    baseband_payload = {
        'time': t.tolist(),
        'i_tx': baseband.real.tolist(),
        'q_tx': baseband.imag.tolist(),
        'i_rx': rx_complex.real.tolist(),
        'q_rx': rx_complex.imag.tolist(),
    }
    if costas_on:
        baseband_payload['i_costas'] = corrected_wave.real.tolist()
        baseband_payload['q_costas'] = corrected_wave.imag.tolist()
    else:
        baseband_payload['i_costas'] = []
        baseband_payload['q_costas'] = []

    return jsonify(
        {
            'time': t.tolist(),
            'tx_passband': tx_passband.tolist(),
            'rx_passband': rx_passband.tolist(),
            'baseband': baseband_payload,
            'constellation': constellation,
            'info': info,
        }
    )




# ---------- UI page ----------
@dig_bp.route('/')
def digital_modulation():
    return render_template('digital_modulation.html')


# ---------- presets ----------

PRESETS = {
    'pam_voice': {'type': 'PAM', 'fs': 2000, 't_end': 1.0, 'prf': 50, 'fm': 5, 'snr_db': np.inf},
    'pwm_slow': {'type': 'PWM', 'fs': 2000, 't_end': 1.0, 'prf': 20, 'fm': 2, 'snr_db': np.inf},
    'ppm_demo': {'type': 'PPM', 'fs': 2000, 't_end': 1.0, 'prf': 40, 'fm': 3, 'snr_db': np.inf},
    'pcm_16lvl': {'type': 'PCM', 'fs': 2000, 't_end': 1.0, 'prf': 40, 'fm': 4, 'levels': 16, 'snr_db': np.inf},
}


@dig_bp.route('/api/presets')
def presets_api():
    return jsonify(list(PRESETS.keys()))


# ---------- modulation ----------

@dig_bp.route('/api/modulate')
def modulate_api():
    params = {k: request.args.get(k) for k in request.args.keys()}
    kind = (params.get('type') or 'PAM').upper()
    fs = float(params.get('fs', 2000))
    t_end = float(params.get('t_end', 1.0))
    snr_db = _to_float_or_inf(params.get('snr_db', None))

    preset = params.get('preset')
    if preset in PRESETS:
        for k, v in PRESETS[preset].items():
            params[k] = v
        kind = PRESETS[preset].get('type', kind).upper()
        fs = float(PRESETS[preset].get('fs', fs))
        t_end = float(PRESETS[preset].get('t_end', t_end))
        snr_db = _to_float_or_inf(PRESETS[preset].get('snr_db', snr_db))

    t = make_time(t_end=t_end, fs=fs)

    message = np.zeros_like(t)
    carrier = np.zeros_like(t)
    modulated = np.zeros_like(t)
    info = {'type': kind, 'fs': fs, 'duration_s': t_end}

    if kind == 'PAM':
        prf = float(params.get('prf', 50))
        fm = float(params.get('fm', 5))
        message = np.sin(2 * np.pi * fm * t)
        pulses = (np.mod(t, 1 / prf) < (1 / (2 * prf))).astype(float)
        carrier = pulses.copy()
        modulated = pulses * message
        info.update({'prf': prf, 'fm': fm})

    elif kind == 'PWM':
        prf = float(params.get('prf', 50))
        fm = float(params.get('fm', 5))
        message = np.sin(2 * np.pi * fm * t)
        duty = (message + 1) / 2
        tau = np.mod(t, 1 / prf)
        pulses = (tau < (duty / prf)).astype(float)
        carrier = pulses.copy()
        modulated = pulses
        info.update({'prf': prf, 'fm': fm})

    elif kind == 'PPM':
        prf = float(params.get('prf', 50))
        fm = float(params.get('fm', 5))
        message = np.sin(2 * np.pi * fm * t)
        shift = (message + 1) / 2 * (1 / (2 * prf))
        tau = np.mod(t, 1 / prf)
        width = 1 / (20 * prf)
        pulses = (np.abs(tau - shift) < width).astype(float)
        carrier = pulses.copy()
        modulated = pulses
        info.update({'prf': prf, 'fm': fm})

    elif kind == 'PCM':
        prf = float(params.get('prf', 40))
        fm = float(params.get('fm', 5))
        levels = int(params.get('levels', 8))
        levels = max(2, levels)
        message = np.sin(2 * np.pi * fm * t)
        q = np.round((message + 1) / 2 * (levels - 1))
        qnorm = q / (levels - 1) * 2 - 1
        sidx = np.floor(t * prf).astype(int)
        carrier = np.ones_like(t)
        modulated = qnorm[np.clip(sidx, 0, len(qnorm) - 1)]
        info.update({'prf': prf, 'fm': fm, 'levels': levels})

    else:
        return jsonify(error="Unknown modulation type"), 400
    
    mod_noisy = add_awgn(modulated, snr_db)
    if snr_db is not None:
        info['snr_db'] = float(snr_db) if not np.isinf(snr_db) else 'inf'
    f_mod, Pdb_mod = spectrum_db(mod_noisy, fs)

    return jsonify({
        't': t.tolist(),
        'message': message.tolist(),
        'carrier': carrier.tolist(),
        'modulated': mod_noisy.tolist(),
        'f': f_mod.tolist(),
        'P_db': Pdb_mod.tolist(),
        'info': info
    })


# ---------- demodulation ----------

@dig_bp.route('/api/demodulate')
def demodulate_api():
    params = {k: request.args.get(k) for k in request.args.keys()}
    kind = (params.get('type') or 'PAM').upper()
    fs = float(params.get('fs', 2000))
    t_end = float(params.get('t_end', 1.0))
    t = make_time(t_end=t_end, fs=fs)

    if kind in ['PAM', 'PWM', 'PPM', 'PCM']:
        prf = float(params.get('prf', 50))
        fm = float(params.get('fm', 5))
        msg = np.sin(2 * np.pi * fm * t)

        if kind == 'PAM':
            tx = (np.mod(t, 1 / prf) < 1 / (2 * prf)).astype(float) * msg
            demod = msg
        elif kind == 'PWM':
            duty = (msg + 1) / 2
            tau = np.mod(t, 1 / prf)
            tx = (tau < (duty / prf)).astype(float)
            demod = msg
        elif kind == 'PPM':
            shift = (msg + 1) / 2 * (1 / (2 * prf))
            tau = np.mod(t, 1 / prf)
            width = 1 / (20 * prf)
            tx = (np.abs(tau - shift) < width).astype(float)
            demod = msg
        elif kind == 'PCM':
            levels = int(params.get('levels', 8))
            q = np.round((msg + 1) / 2 * (levels - 1))
            demod = q / (levels - 1) * 2 - 1
            sidx = np.floor(t * prf).astype(int)
            tx = demod[np.clip(sidx, 0, len(demod) - 1)]
    else:
        return jsonify(error="Unknown demodulation type"), 400

    f_rx, Pdb_rx = spectrum_db(demod, fs)

    return jsonify({
        't': t.tolist(),
        'modulated': tx.tolist(),
        'demodulated': demod.tolist(),
        'f': f_rx.tolist(),
        'P_db': Pdb_rx.tolist()
    })

@dig_bp.route('/api/m_pam')
def m_pam_api():
    params = {k: request.args.get(k) for k in request.args.keys()}

    try:
        M = int(params.get('M', params.get('m', 4)))
        sps = int(params.get('sps', 8))
        rolloff = float(params.get('rolloff', 0.35))
        span = int(params.get('span', 8))
        snr_db = float(params.get('snr_db', 12.0))
        symbols = int(params.get('symbols', 2000))
    except ValueError as exc:
        return jsonify(error=f"Invalid parameter: {exc}"), 400

    seed_val = params.get('seed')
    curve_seed_base = None
    if seed_val is not None:
        try:
            seed_int = int(seed_val)
        except ValueError:
            return jsonify(error="seed must be an integer"), 400
        rng_main = np.random.default_rng(seed_int)
        curve_seed_base = seed_int + 1
    else:
        rng_main = np.random.default_rng()

    enable_tx_rrc = _to_bool(params.get('tx_filter', params.get('enable_tx_rrc', '1')), default=True)
    enable_rx_rrc = _to_bool(params.get('rx_filter', params.get('enable_rx_rrc', '1')), default=True)

    try:
        sim = simulate_m_pam(
            M=M,
            sps=sps,
            rolloff=rolloff,
            span=span,
            snr_db=snr_db,
            symbols=symbols,
            enable_tx_rrc=enable_tx_rrc,
            enable_rx_rrc=enable_rx_rrc,
            rng=rng_main,
            include_waveforms=False,
            include_eye=True,
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    levels = sim['levels']
    rx_symbols = sim['rx_symbols']
    tx_symbols = sim['tx_symbols']
    decisions = sim['decisions']
    sample_idx = sim['sample_indices']
    bits_per_symbol = sim['bits_per_symbol']
    symbol_count = sim['symbols']
    bits_total = symbol_count * bits_per_symbol

    eye_data = sim['eye']
    if eye_data is not None:
        eye_payload = {
            'time': eye_data['time'].tolist(),
            'traces': eye_data['traces'].tolist(),
        }
    else:
        eye_payload = {'time': [], 'traces': []}

    theory_current = float(pam_theoretical_ber(M, snr_db))

    snr_grid = np.arange(0, 21, 2)
    theory_curve = pam_theoretical_ber(M, snr_grid)
    sim_curve = []
    sim_curve_raw = []
    curve_symbols = max(800, min(4000, symbols))
    for idx, snr_test in enumerate(snr_grid):
        if curve_seed_base is not None:
            rng_loop = np.random.default_rng(curve_seed_base + idx)
        else:
            rng_loop = np.random.default_rng()
        sim_point = simulate_m_pam(
            M=M,
            sps=sps,
            rolloff=rolloff,
            span=span,
            snr_db=float(snr_test),
            symbols=curve_symbols,
            enable_tx_rrc=enable_tx_rrc,
            enable_rx_rrc=enable_rx_rrc,
            rng=rng_loop,
            include_waveforms=False,
            include_eye=False,
        )
        raw = sim_point['ber']
        min_floor = 1.0 / max(1, sim_point['symbols'] * sim_point['bits_per_symbol'])
        sim_curve_raw.append(raw)
        sim_curve.append(max(raw, min_floor))

    min_floor_current = 1.0 / max(1, bits_total)
    measured_point = max(sim['ber'], min_floor_current)

    return jsonify({
        'constellation': {
            'samples': rx_symbols.tolist(),
            'ideal': levels.tolist(),
            'decisions': decisions.tolist(),
            'tx': tx_symbols.tolist(),
        },
        'sample_indices': sample_idx.tolist(),
        'eye': eye_payload,
        'ber': {
            'curve_snr': snr_grid.tolist(),
            'curve_theory': theory_curve.tolist(),
            'curve_sim': sim_curve,
            'curve_sim_raw': sim_curve_raw,
            'point': {
                'snr_db': snr_db,
                'measured': float(sim['ber']),
                'measured_plot': measured_point,
                'theory': theory_current,
                'bit_errors': int(sim['bit_errors']),
                'bits': int(bits_total),
            },
        },
        'info': {
            'M': M,
            'bits_per_symbol': bits_per_symbol,
            'rolloff': rolloff,
            'span': span,
            'sps': sps,
            'symbols': symbol_count,
            'tx_rrc': bool(enable_tx_rrc),
            'rx_rrc': bool(enable_rx_rrc),
        },
    })



# ---------- legacy routes (for backwards compatibility) ----------

@dig_bp.route('/api/digital_modulate')
def legacy_modulate():
    return modulate_api()


@dig_bp.route('/api/digital_demodulate')
def legacy_demodulate():
    return demodulate_api()
